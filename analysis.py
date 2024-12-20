import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import SpecRNet
from train_asvspoof import ASVspoofLADataset
from torch.utils.data import DataLoader
import json
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from config import get_specrnet_config


# mapping
ATTACK_TYPES = {
    'A01': {'type': 'TTS', 'technique': 'neural waveform model'},
    'A02': {'type': 'TTS', 'technique': 'vocoder'},
    'A03': {'type': 'TTS', 'technique': 'vocoder'},
    'A04': {'type': 'TTS', 'technique': 'waveform concatenation'},
    'A05': {'type': 'VC', 'technique': 'vocoder'},
    'A06': {'type': 'VC', 'technique': 'spectral filtering'},
    'A07': {'type': 'TTS', 'technique': 'vocoder+GAN'},
    'A08': {'type': 'TTS', 'technique': 'neural waveform'},
    'A09': {'type': 'TTS', 'technique': 'vocoder'},
    'A10': {'type': 'TTS', 'technique': 'neural waveform'},
    'A11': {'type': 'TTS', 'technique': 'griffin lim'},
    'A12': {'type': 'TTS', 'technique': 'neural waveform'},
    'A13': {'type': 'TTS_VC', 'technique': 'waveform concatenation+filtering'},
    'A14': {'type': 'TTS_VC', 'technique': 'vocoder'},
    'A15': {'type': 'TTS_VC', 'technique': 'neural waveform'},
    'A16': {'type': 'TTS', 'technique': 'waveform concatenation'},
    'A17': {'type': 'VC', 'technique': 'waveform filtering'},
    'A18': {'type': 'VC', 'technique': 'vocoder'},
    'A19': {'type': 'VC', 'technique': 'spectral filtering'}
}

def compute_eer(labels, scores):
    """Compute EER"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold

def load_model(checkpoint_path, device):
    """load the best model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']

    model_config['filts'] = [1, [1, 20], [20, 64], [20, 64]]
    model = SpecRNet(model_config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model

def analyze_results(protocol_file, audio_dir, model_path, batch_size=32, device='cuda'):
    """detailed analysis of model performance"""
    
    model = load_model(model_path, device)
    
    protocol_df = pd.read_csv(protocol_file, sep=' ', header=None,
                             names=['speaker_id', 'file_name', 'unused', 'system_id', 'key'])
    
    dataset = ASVspoofLADataset(protocol_file, audio_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(loader)):
            data = data.to(device)
            outputs = model(data)
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            if isinstance(scores, np.float32):
                scores = np.array([scores])
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
    
    scores_array = np.array(all_scores)
    labels_array = np.array(all_labels)
        
    overall_eer, threshold = compute_eer(labels_array, scores_array)
    
    bonafide_mask = protocol_df['system_id'] == '-'
    bonafide_indices = protocol_df[bonafide_mask].index
    bonafide_scores = scores_array[bonafide_indices]
    bonafide_labels = labels_array[bonafide_indices]
    
    attack_results = {}
    
    for system_id in protocol_df['system_id'].unique():
        if system_id == '-':  # skip bonafide
            continue
            
        
        system_mask = protocol_df['system_id'] == system_id
        system_indices = protocol_df[system_mask].index
        
        
        if len(system_indices) == 0:
            continue
            
        attack_scores = np.concatenate([scores_array[system_indices], bonafide_scores])
        attack_labels = np.concatenate([labels_array[system_indices], bonafide_labels])
        
        attack_eer, _ = compute_eer(attack_labels, attack_scores)
        
        attack_results[system_id] = {
            'eer': float(attack_eer),
            'num_samples': len(system_indices),  # only count attack samples
            'type': ATTACK_TYPES[system_id]['type'],
            'technique': ATTACK_TYPES[system_id]['technique']
        }
            
    
    type_results = {}
    for attack_type in ['TTS', 'VC', 'TTS_VC']:
        type_attacks = [k for k, v in ATTACK_TYPES.items() if v['type'] == attack_type]
        type_indices = protocol_df[protocol_df['system_id'].isin(type_attacks)].index
        
        if len(type_indices) > 0:
            type_scores = np.concatenate([scores_array[type_indices], bonafide_scores])
            type_labels = np.concatenate([labels_array[type_indices], bonafide_labels])
            type_eer, _ = compute_eer(type_labels, type_scores)
            type_results[attack_type] = float(type_eer)

    
    predictions = (scores_array > threshold).astype(int)
    cm = confusion_matrix(labels_array, predictions)
    
    analysis_results = {
        'overall_metrics': {
            'eer': float(overall_eer),
            'threshold': float(threshold),
            'accuracy': float((predictions == labels_array).mean()),
            'confusion_matrix': cm.tolist()
        },
        'attack_results': attack_results,
        'type_results': type_results
    }
    
    return analysis_results

def visualize_results(results):
    """visualizations of the analysis results"""
    attack_types = list(results['type_results'].keys())
    eer_values = [results['type_results'][t] * 100 for t in attack_types] 
    
    if len(attack_types) > 0:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(attack_types, eer_values)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.title('Equal error rate (EER) by attack Type')
        plt.ylabel('EER (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('attack_type_eer.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # EER by specific attack
    attacks = sorted(results['attack_results'].keys())
    if len(attacks) > 0:
        attack_eers = [results['attack_results'][a]['eer'] * 100 for a in attacks]
        attack_types = [ATTACK_TYPES[a]['type'] for a in attacks]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(attacks)), attack_eers)
        
        colors = {'TTS': 'skyblue', 'VC': 'lightcoral', 'TTS_VC': 'lightgreen'}
        for idx, (bar, attack_type) in enumerate(zip(bars, attack_types)):
            bar.set_color(colors[attack_type])
            height = bar.get_height()
            plt.text(idx, height, f'{height:.2f}%',
                    ha='center', va='bottom')
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=label)
                          for label, color in colors.items()]
        plt.legend(handles=legend_elements)
        
        plt.title('Equal error rate (EER) by specific attack')
        plt.ylabel('EER (%)')
        plt.xlabel('Attack ID')
        plt.xticks(range(len(attacks)), attacks, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('specific_attack_eer.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    #confusion matrix
    cm = np.array(results['overall_metrics']['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bonafide', 'Spoof'],
                yticklabels=['Bonafide', 'Spoof'])
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    protocol_file = "./data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    audio_dir = "./data/LA/ASVspoof2019_LA_eval/flac"
    model_path = "./asvspoof_checkpoints/best_model.pth"
    
    results = analyze_results(protocol_file, audio_dir, model_path)
    
    with open('detailed_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    visualize_results(results)
