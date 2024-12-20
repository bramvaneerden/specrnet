import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from model import SpecRNet
from config import get_specrnet_config
from train_asvspoof import ASVspoofLADataset, compute_eer

def set_seed(seed):
    """random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """training for one epoch"""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/3')
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
    
    return train_loss/len(train_loader), 100.*train_correct/train_total

def evaluate(model, data_loader, criterion, device):
    """Evaluation"""
    model.eval()
    total_loss = 0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target)
            total_loss += loss.item()
            
            scores = torch.sigmoid(output.squeeze()).cpu().numpy()
            labels = target.cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(data_loader)
    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    
    return avg_loss, eer

def run_multiple_trainings(
    train_protocol, 
    train_dir,
    val_protocol,
    val_dir,
    test_protocol,
    test_dir,
    num_runs=5,
    num_epochs=3,
    base_seed=42
):
    """Run multiple train runs"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    print("Loading ds")
    train_dataset = ASVspoofLADataset(train_protocol, train_dir)
    val_dataset = ASVspoofLADataset(val_protocol, val_dir)
    test_dataset = ASVspoofLADataset(test_protocol, test_dir)
    
    # data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    run_results = []
    best_overall_eer = float('inf')
    
    for run in range(num_runs):
        print(f'\nRun {run + 1}/{num_runs}')
        
        current_seed = base_seed + run
        set_seed(current_seed)
        
        # model
        model_config = get_specrnet_config(input_channels=1)
        model = SpecRNet(model_config, device=device).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=0.0001)
        
        # train loop
        run_metrics = []
        best_val_eer = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                              optimizer, device, epoch)
            
            val_loss, val_eer = evaluate(model, val_loader, criterion, device)
            
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                best_model_state = model.state_dict().copy()
            
            print(f'epoch {epoch + 1}:')
            print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%')
            print(f'val loss: {val_loss:.4f}, val EER: {val_eer:.4f}')
            
            run_metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_eer': val_eer
            })
        
        model.load_state_dict(best_model_state)
        test_loss, test_eer = evaluate(model, test_loader, criterion, device)
        
        if run == 0 or test_eer < best_overall_eer:
            best_overall_eer = test_eer
            os.makedirs('asvspoof_checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'config': model_config,
                'val_eer': best_val_eer,
                'test_eer': test_eer
            }, './asvspoof_checkpoints/best_model.pth')
        
        run_results.append({
            'run': run + 1,
            'seed': current_seed,
            'best_val_eer': best_val_eer,
            'test_eer': test_eer
        })
        
        run_df = pd.DataFrame(run_metrics)
        os.makedirs('training_results', exist_ok=True)
        run_df.to_csv(f'training_results/run_{run + 1}_metrics.csv', index=False)
    
    # final results
    results_df = pd.DataFrame(run_results)
    mean_test_eer = results_df['test_eer'].mean()
    std_test_eer = results_df['test_eer'].std()
    
    print('\nfinal results:')
    print('-------------')
    for idx, row in results_df.iterrows():
        print(f"run {row['run']}: val EER = {row['best_val_eer']:.4f}, test EER = {row['test_eer']:.4f}")
    print(f'\nmean test EER: {mean_test_eer:.4f} ± {std_test_eer:.4f}')
    
    results_df.to_csv('training_results/final_results.csv', index=False)
    
    return results_df, mean_test_eer, std_test_eer

if __name__ == "__main__":
    # paths
    train_protocol = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    train_dir = "./data/LA/ASVspoof2019_LA_train/flac/"
    val_protocol = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    val_dir = "./data/LA/ASVspoof2019_LA_dev/flac/"
    test_protocol = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    test_dir = "./data/LA/ASVspoof2019_LA_eval/flac/"
    
    results_df, mean_eer, std_eer = run_multiple_trainings(
        train_protocol=train_protocol,
        train_dir=train_dir,
        val_protocol=val_protocol,
        val_dir=val_dir,
        test_protocol=test_protocol,
        test_dir=test_dir,
        num_runs=5,
        num_epochs=3
    )