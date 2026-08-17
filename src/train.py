import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.dataset import SwitchVADataset
from src.model import SwitchVANet

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

def compute_ccc(y_true, y_pred):
    """
    Computes Concordance Correlation Coefficient (CCC).
    """
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    true_var = np.var(y_true)
    pred_var = np.var(y_pred)
    cov = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    ccc = (2 * cov) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + 1e-9)
    return ccc

def compute_cf1(v_true, a_true, v_pred, a_pred):
    """
    Computes continuous F1 (cF1) metric for given aspects.
    """
    v_true = np.clip(v_true, 0.0, 1.0)
    a_true = np.clip(a_true, 0.0, 1.0)
    v_pred = np.clip(v_pred, 0.0, 1.0)
    a_pred = np.clip(a_pred, 0.0, 1.0)
    
    errors = np.sqrt((v_pred - v_true)**2 + (a_pred - a_true)**2)
    d_max = np.sqrt(2.0)
    ctp = 1.0 - (errors / d_max)
    ctp = np.clip(ctp, 0.0, 1.0)
    cf1 = np.mean(ctp)
    return cf1

def evaluate(model, dataloader, alpha=0.0):
    model.eval()
    val_loss = 0.0
    val_loss_reg = 0.0
    val_loss_csl = 0.0
    
    all_v_true, all_v_pred = [], []
    all_a_true, all_a_pred = [], []
    
    csl_criterion = nn.CrossEntropyLoss(ignore_index=2)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_mask = batch['aspect_token_mask'].to(device)
            spe_features = batch['spe_features'].to(device)
            
            token_langs = batch['token_langs'].to(device)
            v_true = batch['valence'].to(device)
            a_true = batch['arousal'].to(device)
            
            out = model(input_ids, attention_mask, aspect_mask, spe_features)
            
            csl_logits = out['csl_logits']
            v_pred = out['valence']
            a_pred = out['arousal']
            
            loss_v = nn.MSELoss()(v_pred, v_true)
            loss_a = nn.MSELoss()(a_pred, a_true)
            loss_csl = csl_criterion(csl_logits.view(-1, 3), token_langs.view(-1))
            
            loss = loss_v + loss_a + alpha * loss_csl
            
            val_loss += loss.item()
            val_loss_reg += (loss_v + loss_a).item()
            val_loss_csl += loss_csl.item()
            
            all_v_true.extend(v_true.cpu().numpy())
            all_v_pred.extend(v_pred.cpu().numpy())
            all_a_true.extend(a_true.cpu().numpy())
            all_a_pred.extend(a_pred.cpu().numpy())
            
    all_v_true = np.array(all_v_true)
    all_v_pred = np.array(all_v_pred)
    all_a_true = np.array(all_a_true)
    all_a_pred = np.array(all_a_pred)
    
    ccc_v = compute_ccc(all_v_true, all_v_pred)
    ccc_a = compute_ccc(all_a_true, all_a_pred)
    cf1 = compute_cf1(all_v_true, all_a_true, all_v_pred, all_a_pred)
    
    mae_v = np.mean(np.abs(all_v_pred - all_v_true))
    mae_a = np.mean(np.abs(all_a_pred - all_a_true))
    
    return {
        'loss': val_loss / len(dataloader),
        'loss_reg': val_loss_reg / len(dataloader),
        'loss_csl': val_loss_csl / len(dataloader),
        'ccc_v': ccc_v,
        'ccc_a': ccc_a,
        'cf1': cf1,
        'mae_v': mae_v,
        'mae_a': mae_a,
        'v_pred': all_v_pred,
        'a_pred': all_a_pred
    }

def train_config(config_name, use_spe, alpha, train_loader, val_loader, test_loader, epochs=8, lr=2e-5, encoder_name='xlm-roberta-base'):
    print(f"\n==========================================")
    print(f"Training Config: {config_name}")
    print(f"use_spe: {use_spe}, alpha (LID loss weight): {alpha}")
    print(f"==========================================")
    
    model = SwitchVANet(encoder_name=encoder_name, use_spe=use_spe).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    csl_criterion = nn.CrossEntropyLoss(ignore_index=2)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_reg = 0.0
        train_loss_csl = 0.0
        
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_mask = batch['aspect_token_mask'].to(device)
            spe_features = batch['spe_features'].to(device)
            
            token_langs = batch['token_langs'].to(device)
            v_true = batch['valence'].to(device)
            a_true = batch['arousal'].to(device)
            
            out = model(input_ids, attention_mask, aspect_mask, spe_features)
            
            csl_logits = out['csl_logits']
            v_pred = out['valence']
            a_pred = out['arousal']
            
            loss_v = nn.MSELoss()(v_pred, v_true)
            loss_a = nn.MSELoss()(a_pred, a_true)
            loss_csl = csl_criterion(csl_logits.view(-1, 3), token_langs.view(-1))
            
            loss = loss_v + loss_a + alpha * loss_csl
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_reg += (loss_v + loss_a).item()
            train_loss_csl += loss_csl.item()
            
        epoch_time = time.time() - t0
        
        # Validation
        val_metrics = evaluate(model, val_loader, alpha=alpha)
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} (Reg: {train_loss_reg/len(train_loader):.4f}, CSL: {train_loss_csl/len(train_loader):.4f})")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} (Reg: {val_metrics['loss_reg']:.4f}, CSL: {val_metrics['loss_csl']:.4f})")
        print(f"  Val CCC V:  {val_metrics['ccc_v']:.4f} | CCC A: {val_metrics['ccc_a']:.4f} | cF1: {val_metrics['cf1']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, alpha=alpha)
    print(f"\n--- Test Results for {config_name} ---")
    print(f"CCC Valence: {test_metrics['ccc_v']:.4f}")
    print(f"CCC Arousal: {test_metrics['ccc_a']:.4f}")
    print(f"cF1 Score:   {test_metrics['cf1']:.4f}")
    print(f"MAE Valence: {test_metrics['mae_v']:.4f} | MAE Arousal: {test_metrics['mae_a']:.4f}")
    
    return test_metrics

def main():
    csv_path = '/Users/kmrinal/SwitchVA/dataset.csv'
    word_languages_path = '/Users/kmrinal/SwitchVA/word_languages.json'
    encoder_name = 'xlm-roberta-base'
    batch_size = 16
    epochs = 8
    lr = 2e-5
    
    # Load dataset splits
    print("Loading datasets...")
    train_dataset = SwitchVADataset(csv_path, word_languages_path, split='train', tokenizer_name=encoder_name)
    val_dataset = SwitchVADataset(csv_path, word_languages_path, split='val', tokenizer_name=encoder_name)
    test_dataset = SwitchVADataset(csv_path, word_languages_path, split='test', tokenizer_name=encoder_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Splits loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Store predictions to construct baseline and SwitchVA-Net test outputs
    results = {}
    
    # Run Configuration 1: Baseline
    res_baseline = train_config("Baseline", use_spe=False, alpha=0.0, 
                                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                epochs=epochs, lr=lr, encoder_name=encoder_name)
    results["Baseline"] = res_baseline
    
    # Run Configuration 2: CSL-Aux Only
    res_csl = train_config("CSL-Aux Only", use_spe=False, alpha=1.0, 
                            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            epochs=epochs, lr=lr, encoder_name=encoder_name)
    results["CSL-Aux Only"] = res_csl
    
    # Run Configuration 3: SPE Only
    res_spe = train_config("SPE Only", use_spe=True, alpha=0.0, 
                            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            epochs=epochs, lr=lr, encoder_name=encoder_name)
    results["SPE Only"] = res_spe
    
    # Run Configuration 4: SwitchVA-Net (Full)
    res_full = train_config("SwitchVA-Net (Full)", use_spe=True, alpha=1.0, 
                             train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                             epochs=epochs, lr=lr, encoder_name=encoder_name)
    results["SwitchVA-Net (Full)"] = res_full
    
    # Save Baseline and SwitchVA-Net predictions for the mixed-effects analysis
    # Extract IDs, text, domain, and SPE details directly from test dataset dataframe
    test_df = test_dataset.df.copy()
    
    # Extract distance and density values for each test row to save along predictions
    distances = []
    densities = []
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        distances.append(item['min_dist'])
        # Reconstruct unscaled density
        densities.append(item['spe_features'][1].item() * 5.0)
        
    test_df['spe_distance'] = distances
    test_df['spe_density'] = densities
    
    # Save Baseline predictions
    baseline_pred_df = test_df.copy()
    baseline_pred_df['predicted_valence'] = res_baseline['v_pred']
    baseline_pred_df['predicted_arousal'] = res_baseline['a_pred']
    baseline_pred_df.to_csv('/Users/kmrinal/SwitchVA/baseline_predictions.csv', index=False)
    print("\nSaved baseline predictions to /Users/kmrinal/SwitchVA/baseline_predictions.csv")
    
    # Save SwitchVA-Net predictions
    full_pred_df = test_df.copy()
    full_pred_df['predicted_valence'] = res_full['v_pred']
    full_pred_df['predicted_arousal'] = res_full['a_pred']
    full_pred_df.to_csv('/Users/kmrinal/SwitchVA/switchva_predictions.csv', index=False)
    print("Saved SwitchVA-Net predictions to /Users/kmrinal/SwitchVA/switchva_predictions.csv")
    
    # Summarize results in a clean table
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            "Configuration": name,
            "Valence CCC": metrics['ccc_v'],
            "Arousal CCC": metrics['ccc_a'],
            "Valence MAE": metrics['mae_v'],
            "Arousal MAE": metrics['mae_a'],
            "cF1 Score": metrics['cf1']
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n================== ABLATION SUMMARY ==================")
    print(summary_df.to_string(index=False))
    print("======================================================")
    
    summary_df.to_csv('/Users/kmrinal/SwitchVA/ablation_summary.csv', index=False)
    print("Saved summary table to /Users/kmrinal/SwitchVA/ablation_summary.csv")

if __name__ == "__main__":
    main()
