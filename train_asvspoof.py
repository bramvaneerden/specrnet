import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import SpecRNet
from config import get_specrnet_config
import warnings
import librosa

class ASVspoofLADataset(Dataset):
    # dataloader for dataset:
    # need init, getitem  len
    def __init__(self, protocol_file, audio_dir):
        """ASVspoof 2019 LA dataset
        Args:
            protocol_file: Path to protocol file (train/dev/eval)
            audio_dir: Directory containing audio files
        """
        self.metadata_df = pd.read_csv(
            protocol_file, 
            sep=' ', 
            header=None,
            names=['speaker_id', 'file_name', 'system_id', 'unused', 'key']
        )
        self.audio_dir = audio_dir
        
        # (bonafide=0, spoof=1)
        self.metadata_df['label'] = (self.metadata_df['key'] == 'spoof').astype(int)
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        try:
            file_path = os.path.join(self.audio_dir, f"{self.metadata_df.iloc[idx]['file_name']}.flac")
            label = self.metadata_df.iloc[idx]['label']
            
            try:
                waveform, sample_rate = torchaudio.load(file_path)
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
                return torch.zeros((1, 80, 404)), torch.tensor(label, dtype=torch.float32)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            #mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=80,
                f_min=20,
                f_max=8000,
                window_fn=torch.hann_window
            )
            
            mel_spectrogram = mel_transform(waveform)
            
            mel_spectrogram = torchaudio.transforms.AmplitudeToDB(
                top_db=80
            )(mel_spectrogram)
            
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (mel_spectrogram.std() + 1e-8)
            
            target_length = 404
            current_length = mel_spectrogram.shape[2]
            
            if current_length < target_length:
                mel_spectrogram = F.pad(mel_spectrogram, (0, target_length - current_length))
            elif current_length > target_length:
                mel_spectrogram = mel_spectrogram[:, :, :target_length]
                
            return mel_spectrogram, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            return torch.zeros((1, 80, 404)), torch.tensor(label, dtype=torch.float32)

def compute_eer(labels, scores):
    """Compute Equal Error Rate (EER)"""
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

class EarlyStoppingEER: # eearly stopping for EEr
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_eer = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, current_eer, model):
        if self.best_eer is None:
            self.best_eer = current_eer
            self.save_checkpoint(model)
        elif current_eer > self.best_eer - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_eer = current_eer
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def train_model_asvspoof(train_protocol,
                        train_dir,
                        dev_protocol,
                        dev_dir,
                        num_epochs=3,
                        batch_size=32,
                        learning_rate=0.0001,
                        save_dir='asvspoof_checkpoints'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = ASVspoofLADataset(train_protocol, train_dir)
    dev_dataset = ASVspoofLADataset(dev_protocol, dev_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model_config = get_specrnet_config(input_channels=1)
    model = SpecRNet(model_config, device=device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    
    training_history = []
    
    early_stopping = EarlyStoppingEER(patience=5)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nstarting training")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output.squeeze(), target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
            
            pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': 100.*train_correct/train_total
            })
        
        model.eval()
        dev_loss = 0
        dev_correct = 0
        dev_total = 0
        dev_scores = []
        dev_labels = []
        
        with torch.no_grad():
            for data, target in dev_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target)
                dev_loss += loss.item()
                
                scores = torch.sigmoid(output.squeeze()).cpu().numpy()
                labels = target.cpu().numpy()
                
                dev_scores.extend(scores)
                dev_labels.extend(labels)
                
                pred = (scores > 0.5).astype(float)
                dev_correct += (pred == labels).sum()
                dev_total += len(labels)
        
        dev_loss /= len(dev_loader)
        dev_accuracy = 100. * dev_correct / dev_total
        dev_eer = compute_eer(np.array(dev_labels), np.array(dev_scores))
        
        print(f'\nepoch {epoch+1}:')
        print(f'dev loss: {dev_loss:.4f}, dev accuracy: {dev_accuracy:.2f}%, dev EER: {dev_eer:.4f}')
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'train_accuracy': 100.*train_correct/train_total,
            'dev_loss': dev_loss,
            'dev_accuracy': dev_accuracy,
            'dev_eer': dev_eer,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_metrics)
        
        early_stopping(dev_eer, model)
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dev_eer': dev_eer,
                'config': model_config,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        scheduler.step(dev_eer)
        
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
        
        if early_stopping.early_stop:
            print("early stop")
            break
    
    model.load_state_dict(early_stopping.best_model_state)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'dev_eer': early_stopping.best_eer,
        'config': model_config,
    }, os.path.join(save_dir, 'best_model.pth'))
    print(f"Training don: dev EER: {early_stopping.best_eer:.4f}")
    
    return model, early_stopping.best_eer

if __name__ == "__main__":
    # paths
    train_protocol = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    train_dir = "./data/LA/ASVspoof2019_LA_train/flac/"
    dev_protocol = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" 
    dev_dir = "./data/LA/ASVspoof2019_LA_dev/flac/"
    
    model, best_eer = train_model_asvspoof(
        train_protocol=train_protocol,
        train_dir=train_dir,
        dev_protocol=dev_protocol,
        dev_dir=dev_dir,
        num_epochs=3,
        batch_size=32,
        learning_rate=0.0001,
        save_dir='asvspoof_checkpoints'
    )
    print(f"training done: best EER: {best_eer:.4f}")