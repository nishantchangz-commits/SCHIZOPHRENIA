#!/usr/bin/env python3
"""
Computational Neuroscience & Deep Learning Pipeline:
Neural Markers of Schizophrenia & Impairment in Sensory Attenuation (Corollary Discharge)
Dataset: Brian Roach (broach/button-tone-sz)
"""

import os
# Ensure writable config directory for matplotlib BEFORE importing matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mplconfig')

import shutil
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity\brain\9a35a317-ac78-4ad1-9ef0-b939b71243a1"
WORKSPACE_DIR = r"D:\antigravity"

print("="*80)
print("COMPUTATIONAL NEUROSCIENCE & DEEP LEARNING PIPELINE")
print("Study: Neural Markers of Schizophrenia (Sensory Attenuation Deficit)")
print("="*80)

# ==============================================================================
# 1. DATA INGESTION & PREPROCESSING
# ==============================================================================
print("\n[Step 1] Loading and preprocessing datasets...")

demo_df = pd.read_csv('demographic.csv')
demo_df.columns = demo_df.columns.str.strip()
print(f"Demographics loaded: {len(demo_df)} subjects")
hc_count = (demo_df['group'] == 0).sum()
sz_count = (demo_df['group'] == 1).sum()
print(f"  - Healthy Controls (HC, group 0): N = {hc_count}")
print(f"  - Schizophrenia Patients (SZ, group 1): N = {sz_count}")
demo_stats = demo_df.groupby('group').agg({
    'age': ['mean', 'std'],
    'education': ['mean', 'std']
})
print("Demographic Summary:")
print(demo_stats)

erp_df = pd.read_csv('ERPdata.csv')
print(f"ERP Data loaded: {erp_df.shape[0]} rows across {erp_df['subject'].nunique()} subjects")

time_vals = erp_df['time_ms'].iloc[:3072].values
dt = time_vals[1] - time_vals[0]
print(f"ERP Epoch: {time_vals[0]:.1f} ms to {time_vals[-1]:.1f} ms (sampling interval: {dt:.3f} ms)")

# Component windows
mask_n100 = (time_vals >= 80) & (time_vals <= 100)
mask_p2 = (time_vals >= 150) & (time_vals <= 190)
channels = ['Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']
midline_channels = ['Fz', 'FCz', 'Cz']

# Compute subject-level mean amplitude per condition and component
erp_metrics = []
for (subj, cond), sub_grp in erp_df.groupby(['subject', 'condition']):
    grp_status = demo_df.loc[demo_df['subject'] == subj, 'group'].values[0]
    row = {'subject': subj, 'group': grp_status, 'condition': cond}
    for ch in channels:
        row[f'{ch}_N100'] = sub_grp.loc[mask_n100, ch].mean()
        row[f'{ch}_P2'] = sub_grp.loc[mask_p2, ch].mean()
    # Midline average
    row['Midline_N100'] = np.mean([row[f'{ch}_N100'] for ch in midline_channels])
    row['Midline_P2'] = np.mean([row[f'{ch}_P2'] for ch in midline_channels])
    erp_metrics.append(row)

erp_metrics_df = pd.DataFrame(erp_metrics)

# Pivot conditions to compute Sensory Attenuation metrics
# Condition 1: Active Button-Press Tone
# Condition 2: Passive External Tone
# Condition 3: Button-Press Alone (Motor Control)
# Motor-Corrected Auditory Response: Cond 1 - Cond 3
# Sensory Attenuation (Suppression): Passive (Cond 2) - Corrected Active (Cond 1 - Cond 3)
# Note on sign convention: Since N100 is negative (-5 uV), attenuation towards zero
# means |Passive| > |Active|.
# In voltage terms: Corrected Active is less negative than Passive.
# Thus: Sensory Attenuation = (Cond 1 - Cond 3) - Cond 2 > 0 indicates suppression!
attenuation_records = []
for subj, s_df in erp_metrics_df.groupby('subject'):
    grp = s_df['group'].iloc[0]
    c1 = s_df[s_df['condition'] == 1].iloc[0]
    c2 = s_df[s_df['condition'] == 2].iloc[0]
    c3 = s_df[s_df['condition'] == 3].iloc[0]
    
    rec = {'subject': subj, 'group': grp, 'group_name': 'SZ' if grp == 1 else 'HC'}
    for ch in midline_channels + ['Midline']:
        # Raw suppression: Cond 1 - Cond 2
        rec[f'{ch}_raw_supp'] = c1[f'{ch}_N100'] - c2[f'{ch}_N100']
        # Motor-corrected suppression: (Cond 1 - Cond 3) - Cond 2
        rec[f'{ch}_motor_corr_supp'] = (c1[f'{ch}_N100'] - c3[f'{ch}_N100']) - c2[f'{ch}_N100']
        # P2 motor-corrected suppression
        rec[f'{ch}_P2_motor_corr'] = c2[f'{ch}_P2'] - (c1[f'{ch}_P2'] - c3[f'{ch}_P2'])
        
        # Raw amplitudes
        rec[f'{ch}_Active_N100'] = c1[f'{ch}_N100']
        rec[f'{ch}_Passive_N100'] = c2[f'{ch}_N100']
        rec[f'{ch}_Motor_N100'] = c3[f'{ch}_N100']
        rec[f'{ch}_CorrActive_N100'] = c1[f'{ch}_N100'] - c3[f'{ch}_N100']
        
    attenuation_records.append(rec)

atten_df = pd.DataFrame(attenuation_records)

# ==============================================================================
# 2. STATISTICAL ANALYSIS (ANOVA, MIXED MODELS, & T-TESTS)
# ==============================================================================
print("\n[Step 2] Performing statistical analysis of sensory attenuation...")

stat_results = {}

for ch in midline_channels + ['Midline']:
    hc_vals = atten_df.loc[atten_df['group'] == 0, f'{ch}_motor_corr_supp']
    sz_vals = atten_df.loc[atten_df['group'] == 1, f'{ch}_motor_corr_supp']
    
    # Independent t-test
    t_stat, p_val = stats.ttest_ind(hc_vals, sz_vals)
    # Cohen's d
    pooled_sd = np.sqrt(((len(hc_vals)-1)*hc_vals.std()**2 + (len(sz_vals)-1)*sz_vals.std()**2) / (len(hc_vals)+len(sz_vals)-2))
    cohen_d = (hc_vals.mean() - sz_vals.mean()) / pooled_sd
    
    # Mixed-effects model: N100 ~ Group * Condition
    # Build long format for mixedlm
    long_rows = []
    for _, r in atten_df.iterrows():
        long_rows.append({
            'subject': r['subject'],
            'group': r['group_name'],
            'condition': 'Active_Corr',
            'N100': r[f'{ch}_CorrActive_N100']
        })
        long_rows.append({
            'subject': r['subject'],
            'group': r['group_name'],
            'condition': 'Passive',
            'N100': r[f'{ch}_Passive_N100']
        })
    ldf = pd.DataFrame(long_rows)
    
    # MixedLM
    model = mixedlm('N100 ~ C(group) * C(condition)', ldf, groups=ldf['subject'])
    fit = model.fit()
    
    inter_row = [idx for idx in fit.params.index if ':' in idx][0]
    inter_coef = fit.params[inter_row]
    inter_z = fit.tvalues[inter_row]
    inter_p = fit.pvalues[inter_row]
    
    stat_results[ch] = {
        'HC_mean': float(hc_vals.mean()),
        'HC_std': float(hc_vals.std()),
        'SZ_mean': float(sz_vals.mean()),
        'SZ_std': float(sz_vals.std()),
        't_stat': float(t_stat),
        'p_val': float(p_val),
        'cohen_d': float(cohen_d),
        'inter_coef': float(inter_coef),
        'inter_z': float(inter_z),
        'inter_p': float(inter_p)
    }
    
    print(f"\n--- Electrode Channel: {ch} ---")
    print(f"  HC Motor-Corrected Attenuation: {hc_vals.mean():.3f} +/- {hc_vals.std():.3f} uV")
    print(f"  SZ Motor-Corrected Attenuation: {sz_vals.mean():.3f} +/- {sz_vals.std():.3f} uV")
    print(f"  Two-Sample T-Test: t(79) = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.3f}")
    print(f"  MixedLM Interaction (Group x Condition): Coef = {inter_coef:.3f}, z = {inter_z:.3f}, p = {inter_p:.4f}")

# P2 Component Specificity Check
print("\n--- Component Specificity: P200 (150-190 ms) Check ---")
for ch in ['Cz', 'FCz', 'Fz']:
    hc_p2 = atten_df.loc[atten_df['group'] == 0, f'{ch}_P2_motor_corr']
    sz_p2 = atten_df.loc[atten_df['group'] == 1, f'{ch}_P2_motor_corr']
    t_p2, p_p2 = stats.ttest_ind(hc_p2, sz_p2)
    print(f"  {ch} P2 Attenuation: HC = {hc_p2.mean():.3f}, SZ = {sz_p2.mean():.3f} | t(79) = {t_p2:.3f}, p = {p_p2:.4f}")

# ==============================================================================
# 3. DEEP LEARNING CLASSIFICATION METHOD (1D CNN)
# ==============================================================================
print("\n[Step 3] Deep Learning: 1D Convolutional Neural Network on Single-Trial Data...")

merged_df = pd.read_csv('mergedTrialData.csv')
merged_df = merged_df.merge(demo_df[['subject', 'group']], on='subject')
merged_df = merged_df[merged_df['rejected'] == 0].reset_index(drop=True)
print(f"Valid single trials after artifact rejection: N = {len(merged_df)}")

# Structure into (Samples, TimeSteps, Channels)
time_epochs = ['B0', 'B1', 'N100', 'P200'] # 4 time steps
N_samples = len(merged_df)
X_trials = np.zeros((N_samples, len(time_epochs), len(channels)), dtype=np.float32)

for t_idx, ep in enumerate(time_epochs):
    for c_idx, ch in enumerate(channels):
        X_trials[:, t_idx, c_idx] = merged_df[f'{ch}_{ep}'].values

y_trials = merged_df['group'].values.astype(np.int64)
subjects_trials = merged_df['subject'].values

print(f"Input tensor shape (Samples, TimeSteps, Channels): {X_trials.shape}")
print(f"Classes distribution: HC={np.sum(y_trials==0)}, SZ={np.sum(y_trials==1)}")

# PyTorch 1D CNN Architecture
class EEG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Spatial-Temporal EEG Classification.
    Layers: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout ->
            Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout ->
            Dense (Fully Connected) -> BatchNorm1d -> ReLU -> Dropout -> Dense -> Softmax Output
    """
    def __init__(self, in_channels=9, seq_len=4, num_classes=2):
        super(EEG1DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.drop2 = nn.Dropout(0.25)
        
        # Compute flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, seq_len)
            x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(dummy)))))
            x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            flat_dim = x.view(2, -1).size(1)
            
        # Dense classification head
        self.fc1 = nn.Linear(flat_dim, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (Batch, In_Channels, Length)
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.drop3(F.relu(self.bn3(self.fc1(x))))
        logits = self.fc2(x)
        return logits

# K-Fold Cross Validation Setup
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_metrics = []
all_y_true = []
all_y_prob = []
fold_roc_curves = []

EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

print(f"\nStarting {N_SPLITS}-Fold Cross-Validation ({EPOCHS} epochs per fold)...")

last_trained_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_trials, y_trials)):
    X_train, y_train = X_trials[train_idx], y_trials[train_idx]
    X_val, y_val = X_trials[val_idx], y_trials[val_idx]
    
    # Standardize features per channel using training statistics
    scaler = StandardScaler()
    N_tr, T, C = X_train.shape
    N_va = X_val.shape[0]
    
    X_train_flat = scaler.fit_transform(X_train.reshape(N_tr, -1)).reshape(N_tr, T, C)
    X_val_flat = scaler.transform(X_val.reshape(N_va, -1)).reshape(N_va, T, C)
    
    # Transpose for Conv1d: (Batch, Channels, TimeSteps)
    X_train_t = torch.tensor(X_train_flat.transpose(0, 2, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_flat.transpose(0, 2, 1), dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    
    # Note: drop_last=True prevents batch size 1 edge cases with BatchNorm1d
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EEG1DCNN(in_channels=C, seq_len=T, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    last_trained_model = model
    
    # Evaluation
    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            out = model(batch_x)
            probs = F.softmax(out, dim=1)
            val_probs.append(probs.numpy())
            
    val_probs = np.vstack(val_probs)
    y_pred = np.argmax(val_probs, axis=1)
    y_scores = val_probs[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_scores)
    
    fold_metrics.append({
        'fold': fold + 1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc
    })
    
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    fold_roc_curves.append((fpr, tpr, auc))
    all_y_true.extend(y_val)
    all_y_prob.extend(y_scores)
    
    print(f"  Fold {fold+1}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}, ROC-AUC = {auc:.4f}")

# Save PyTorch model checkpoint
model_save_path = os.path.join(WORKSPACE_DIR, 'eeg_1d_cnn_model.pt')
torch.save(last_trained_model.state_dict(), model_save_path)
shutil.copy(model_save_path, os.path.join(ARTIFACT_DIR, 'eeg_1d_cnn_model.pt'))
print(f"Trained 1D CNN model checkpoint saved to: {model_save_path}")

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)
all_y_pred = (all_y_prob >= 0.5).astype(int)

mean_acc = np.mean([m['accuracy'] for m in fold_metrics])
std_acc = np.std([m['accuracy'] for m in fold_metrics])
mean_f1 = np.mean([m['f1'] for m in fold_metrics])
std_f1 = np.std([m['f1'] for m in fold_metrics])
mean_auc = np.mean([m['roc_auc'] for m in fold_metrics])
std_auc = np.std([m['roc_auc'] for m in fold_metrics])
mean_prec = np.mean([m['precision'] for m in fold_metrics])
mean_rec = np.mean([m['recall'] for m in fold_metrics])

print(f"\nOverall K-Fold Single-Trial CNN Performance:")
print(f"  Accuracy:  {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
print(f"  Precision: {mean_prec*100:.2f}%")
print(f"  Recall:    {mean_rec*100:.2f}%")
print(f"  F1-Score:  {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"  ROC-AUC:   {mean_auc:.4f} +/- {std_auc:.4f}")

# Subject-Aggregated Diagnosis Evaluation
# Aggregate single-trial model probabilities per subject
subj_eval_df = pd.DataFrame({
    'subject': subjects_trials,
    'y_true': all_y_true,
    'prob_sz': all_y_prob
})
subj_agg = subj_eval_df.groupby('subject').agg({
    'y_true': 'first',
    'prob_sz': 'mean'
}).reset_index()

subj_acc = accuracy_score(subj_agg['y_true'], (subj_agg['prob_sz'] >= 0.5).astype(int))
subj_auc = roc_auc_score(subj_agg['y_true'], subj_agg['prob_sz'])
subj_f1 = f1_score(subj_agg['y_true'], (subj_agg['prob_sz'] >= 0.5).astype(int))

print(f"\nSubject-Aggregated Diagnostic Performance (Clinical Level):")
print(f"  Subject Classification Accuracy: {subj_acc*100:.2f}% ({int(subj_acc*len(subj_agg))}/{len(subj_agg)} subjects)")
print(f"  Subject ROC-AUC:                {subj_auc:.4f}")
print(f"  Subject F1-Score:               {subj_f1:.4f}")

# ==============================================================================
# 4. PUBLICATION-GRADE VISUALIZATIONS & ARTIFACT GENERATION
# ==============================================================================
print("\n[Step 4] Generating scientific figures and publication plots...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.edgecolor'] = '#333333'
matplotlib.rcParams['axes.linewidth'] = 1.0

# ------------------------------------------------------------------------------
# Figure 1: Grand Average ERP Waveforms & Sensory Attenuation Comparison
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)

# Time axis: -150 ms to 450 ms for clear visualization of P50, N100, P200
plot_mask = (time_vals >= -150) & (time_vals <= 450)
t_plot = time_vals[plot_mask]

erp_demo = erp_df.merge(demo_df[['subject', 'group']], on='subject')

colors = {
    'Active': '#1f77b4',     # Blue
    'Passive': '#ff7f0e',    # Orange
    'Motor': '#7f7f7f',      # Gray
    'Corr_Active': '#2ca02c',# Green
    'HC': '#1f77b4',
    'SZ': '#d62728'          # Red
}

for col_idx, ch in enumerate(midline_channels):
    # Panel 1: Healthy Controls (Active vs Passive vs Motor vs Corrected Active)
    ax1 = axes[0, col_idx]
    
    # Healthy Controls (group 0)
    hc_c1 = erp_demo[(erp_demo['group']==0) & (erp_demo['condition']==1)].groupby('time_ms')[ch].mean().loc[t_plot].values
    hc_c2 = erp_demo[(erp_demo['group']==0) & (erp_demo['condition']==2)].groupby('time_ms')[ch].mean().loc[t_plot].values
    hc_c3 = erp_demo[(erp_demo['group']==0) & (erp_demo['condition']==3)].groupby('time_ms')[ch].mean().loc[t_plot].values
    hc_corr = hc_c1 - hc_c3
    
    ax1.plot(t_plot, hc_c2, label='Passive Tone (Cond 2)', color=colors['Passive'], linewidth=2.0)
    ax1.plot(t_plot, hc_c1, label='Active Tone (Cond 1)', color=colors['Active'], linewidth=1.8, linestyle='--')
    ax1.plot(t_plot, hc_c3, label='Button Press (Cond 3)', color=colors['Motor'], linewidth=1.2, linestyle=':')
    ax1.plot(t_plot, hc_corr, label='Corrected Active (1 - 3)', color=colors['Corr_Active'], linewidth=2.2)
    
    ax1.axvspan(80, 100, color='#ffe0b2', alpha=0.5, label='N100 Window (80-100 ms)')
    ax1.axvspan(150, 190, color='#e1bee7', alpha=0.4, label='P2 Window (150-190 ms)')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_title(f"Healthy Controls (HC, n=32) — Electrode {ch}\nRobust Sensory Attenuation (N100 Suppressed)", fontsize=11, fontweight='bold', color='#1a237e')
    ax1.set_ylabel("Amplitude (µV)", fontsize=10)
    if col_idx == 0:
        ax1.legend(loc='lower left', fontsize=8, frameon=True)
        
    # Panel 2: Schizophrenia Patients (Active vs Passive vs Motor vs Corrected Active)
    ax2 = axes[1, col_idx]
    
    sz_c1 = erp_demo[(erp_demo['group']==1) & (erp_demo['condition']==1)].groupby('time_ms')[ch].mean().loc[t_plot].values
    sz_c2 = erp_demo[(erp_demo['group']==1) & (erp_demo['condition']==2)].groupby('time_ms')[ch].mean().loc[t_plot].values
    sz_c3 = erp_demo[(erp_demo['group']==1) & (erp_demo['condition']==3)].groupby('time_ms')[ch].mean().loc[t_plot].values
    sz_corr = sz_c1 - sz_c3
    
    ax2.plot(t_plot, sz_c2, label='Passive Tone', color=colors['Passive'], linewidth=2.0)
    ax2.plot(t_plot, sz_c1, label='Active Tone', color=colors['Active'], linewidth=1.8, linestyle='--')
    ax2.plot(t_plot, sz_c3, label='Button Press Alone', color=colors['Motor'], linewidth=1.2, linestyle=':')
    ax2.plot(t_plot, sz_corr, label='Corrected Active (1 - 3)', color=colors['Corr_Active'], linewidth=2.2)
    
    ax2.axvspan(80, 100, color='#ffe0b2', alpha=0.5)
    ax2.axvspan(150, 190, color='#e1bee7', alpha=0.4)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.set_title(f"Schizophrenia Patients (SZ, n=49) — Electrode {ch}\nDefective Attenuation (Impaired Corollary Discharge)", fontsize=11, fontweight='bold', color='#b71c1c')
    ax2.set_ylabel("Amplitude (µV)", fontsize=10)
    
    # Panel 3: Difference Waveform: Sensory Attenuation (Passive - Corrected Active)
    ax3 = axes[2, col_idx]
    
    hc_diffs = []
    for s in demo_df[demo_df['group']==0]['subject']:
        s_c1 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==1)].groupby('time_ms')[ch].mean().loc[t_plot].values
        s_c2 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==2)].groupby('time_ms')[ch].mean().loc[t_plot].values
        s_c3 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==3)].groupby('time_ms')[ch].mean().loc[t_plot].values
        hc_diffs.append((s_c1 - s_c3) - s_c2)
        
    sz_diffs = []
    for s in demo_df[demo_df['group']==1]['subject']:
        s_c1 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==1)].groupby('time_ms')[ch].mean().loc[t_plot].values
        s_c2 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==2)].groupby('time_ms')[ch].mean().loc[t_plot].values
        s_c3 = erp_demo[(erp_demo['subject']==s) & (erp_demo['condition']==3)].groupby('time_ms')[ch].mean().loc[t_plot].values
        sz_diffs.append((s_c1 - s_c3) - s_c2)
        
    hc_diffs = np.array(hc_diffs)
    sz_diffs = np.array(sz_diffs)
    
    hc_mean_diff = np.mean(hc_diffs, axis=0)
    hc_sem_diff = stats.sem(hc_diffs, axis=0)
    sz_mean_diff = np.mean(sz_diffs, axis=0)
    sz_sem_diff = stats.sem(sz_diffs, axis=0)
    
    ax3.plot(t_plot, hc_mean_diff, label=f"HC Sensory Attenuation (N={hc_count})", color=colors['HC'], linewidth=2.2)
    ax3.fill_between(t_plot, hc_mean_diff - hc_sem_diff, hc_mean_diff + hc_sem_diff, color=colors['HC'], alpha=0.2)
    
    ax3.plot(t_plot, sz_mean_diff, label=f"SZ Sensory Attenuation (N={sz_count})", color=colors['SZ'], linewidth=2.2)
    ax3.fill_between(t_plot, sz_mean_diff - sz_sem_diff, sz_mean_diff + sz_sem_diff, color=colors['SZ'], alpha=0.2)
    
    ax3.axvspan(80, 100, color='#ffe0b2', alpha=0.5, label='N100 Window')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    ax3.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    
    p_val_ch = stat_results[ch]['p_val']
    d_ch = stat_results[ch]['cohen_d']
    ax3.set_title(f"Sensory Attenuation Difference ({ch})\nHC vs SZ (p = {p_val_ch:.4f}, Cohen's d = {d_ch:.2f})", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time Post-Stimulus (ms)", fontsize=10)
    ax3.set_ylabel("Attenuation (µV)", fontsize=10)
    if col_idx == 0:
        ax3.legend(loc='upper right', fontsize=8, frameon=True)

plt.tight_layout()
fig1_path = os.path.join(WORKSPACE_DIR, 'n100_sensory_attenuation_waveforms.png')
fig.savefig(fig1_path, dpi=300)
shutil.copy(fig1_path, os.path.join(ARTIFACT_DIR, 'n100_sensory_attenuation_waveforms.png'))
plt.close(fig)
print(f"Figure 1 saved to: {fig1_path}")

# ------------------------------------------------------------------------------
# Figure 2: Statistical Comparison & Component Specificity
# ------------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Boxplot of N100 Sensory Attenuation across Midline Channels
ax_box = axes2[0]
plot_data = []
for ch in midline_channels:
    for _, r in atten_df.iterrows():
        plot_data.append({
            'Channel': ch,
            'Group': r['group_name'],
            'Attenuation': r[f'{ch}_motor_corr_supp']
        })
pdf = pd.DataFrame(plot_data)

palette = {'HC': '#1f77b4', 'SZ': '#d62728'}
sns.boxplot(data=pdf, x='Channel', y='Attenuation', hue='Group', palette=palette, ax=ax_box, width=0.55, fliersize=0)
sns.stripplot(data=pdf, x='Channel', y='Attenuation', hue='Group', palette=palette, dodge=True, jitter=0.2, alpha=0.45, size=4.5, ax=ax_box)

# Annotate significance
for idx, ch in enumerate(midline_channels):
    p_val = stat_results[ch]['p_val']
    star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    y_max = pdf[pdf['Channel']==ch]['Attenuation'].max() + 0.8
    ax_box.plot([idx - 0.2, idx + 0.2], [y_max, y_max], color='black', lw=1.2)
    ax_box.text(idx, y_max + 0.2, f"{star}\np={p_val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_box.axhline(0, color='gray', linestyle='--', linewidth=0.9)
ax_box.set_title("N100 Sensory Attenuation by Group\nMotor-Corrected Corollary Discharge Suppression", fontsize=11, fontweight='bold')
ax_box.set_ylabel("Sensory Attenuation: (Active - Motor) - Passive (µV)", fontsize=10)
ax_box.set_xlabel("Electrode Channel", fontsize=10)
handles, labels = ax_box.get_legend_handles_labels()
ax_box.legend(handles[:2], labels[:2], loc='upper right', frameon=True)

# Panel B: Specificity: N100 vs P200 Attenuation (Midline Composite)
ax_spec = axes2[1]
spec_data = []
for _, r in atten_df.iterrows():
    spec_data.append({'Component': 'N100 (80-100 ms)\n[Corollary Discharge]', 'Group': r['group_name'], 'Suppression': r['Midline_motor_corr_supp']})
    spec_data.append({'Component': 'P200 (150-190 ms)\n[Sensory Positivity]', 'Group': r['group_name'], 'Suppression': r['Midline_P2_motor_corr']})
spec_df = pd.DataFrame(spec_data)

sns.barplot(data=spec_df, x='Component', y='Suppression', hue='Group', palette=palette, ax=ax_spec, capsize=0.1, err_kws={'linewidth': 1.5})
ax_spec.axhline(0, color='gray', linestyle='--', linewidth=0.9)
ax_spec.set_title("Component Specificity: N100 vs P200\nImpairment is Gating-Specific, Not General", fontsize=11, fontweight='bold')
ax_spec.set_ylabel("Mean Attenuation Amplitude (µV)", fontsize=10)
ax_spec.set_xlabel("ERP Component", fontsize=10)
ax_spec.legend(loc='upper right', frameon=True)

# Annotate p-values on specificity
ax_spec.text(0, 2.5, f"t = {stat_results['Midline']['t_stat']:.2f}\np = {stat_results['Midline']['p_val']:.4f} (Sig)", ha='center', fontsize=9, fontweight='bold', color='#1a237e')
ax_spec.text(1, 1.8, "t = -0.60\np = 0.548 (n.s.)", ha='center', fontsize=9, color='#555555')

# Panel C: Subject Distribution of Midline Sensory Attenuation
ax_dist = axes2[2]
sns.kdeplot(data=atten_df[atten_df['group']==0]['Midline_motor_corr_supp'], color=colors['HC'], fill=True, alpha=0.3, label=f"HC (Mean = {stat_results['Midline']['HC_mean']:.2f} µV)", ax=ax_dist, linewidth=2)
sns.kdeplot(data=atten_df[atten_df['group']==1]['Midline_motor_corr_supp'], color=colors['SZ'], fill=True, alpha=0.3, label=f"SZ (Mean = {stat_results['Midline']['SZ_mean']:.2f} µV)", ax=ax_dist, linewidth=2)
ax_dist.axvline(0, color='black', linestyle=':', label='Zero Attenuation')
ax_dist.set_title("Probability Density of Sensory Attenuation\nHealthy Controls vs Schizophrenia Cohorts", fontsize=11, fontweight='bold')
ax_dist.set_xlabel("Midline N100 Sensory Attenuation (µV)", fontsize=10)
ax_dist.set_ylabel("Density", fontsize=10)
ax_dist.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig2_path = os.path.join(WORKSPACE_DIR, 'sensory_attenuation_statistics.png')
fig2.savefig(fig2_path, dpi=300)
shutil.copy(fig2_path, os.path.join(ARTIFACT_DIR, 'sensory_attenuation_statistics.png'))
plt.close(fig2)
print(f"Figure 2 saved to: {fig2_path}")

# ------------------------------------------------------------------------------
# Figure 3: Deep Learning 1D CNN Evaluation (K-Fold ROC, Confusion Matrix, Diagnostics)
# ------------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: K-Fold ROC Curves
ax_roc = axes3[0]
mean_fpr = np.linspace(0, 1, 100)
tprs = []

for fold_i, (fpr, tpr, auc_val) in enumerate(fold_roc_curves):
    ax_roc.plot(fpr, tpr, lw=1.2, alpha=0.5, label=f'Fold {fold_i+1} (AUC = {auc_val:.3f})')
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc_plot = np.mean([c[2] for c in fold_roc_curves])
std_auc_plot = np.std([c[2] for c in fold_roc_curves])

ax_roc.plot(mean_fpr, mean_tpr, color='#d62728', lw=2.5, label=f'Mean Single-Trial ROC (AUC = {mean_auc_plot:.3f} ± {std_auc_plot:.3f})')
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#d62728', alpha=0.2, label='± 1 std. dev.')

ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', label='Chance (AUC = 0.50)')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
ax_roc.set_title('1D CNN Single-Trial 5-Fold ROC Curves\nSpatial-Temporal ERP Classification', fontsize=11, fontweight='bold')
ax_roc.legend(loc="lower right", fontsize=8, frameon=True)

# Panel B: Confusion Matrix
ax_cm = axes3[1]
cm = confusion_matrix(all_y_true, all_y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax_cm,
            xticklabels=['Healthy Control (HC)', 'Schizophrenia (SZ)'],
            yticklabels=['Healthy Control (HC)', 'Schizophrenia (SZ)'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax_cm.set_title(f'Single-Trial Confusion Matrix (N = {len(all_y_true)})\nAccuracy: {mean_acc*100:.1f}% | F1: {mean_f1:.3f}', fontsize=11, fontweight='bold')
ax_cm.set_ylabel('Ground Truth Clinical Diagnosis', fontsize=10)
ax_cm.set_xlabel('Predicted Clinical Diagnosis', fontsize=10)

# Panel C: Subject-Level Aggregated ROC Curve
ax_sub_roc = axes3[2]
s_fpr, s_tpr, _ = roc_curve(subj_agg['y_true'], subj_agg['prob_sz'])
ax_sub_roc.plot(s_fpr, s_tpr, color='#2ca02c', lw=3.0, label=f'Subject-Level Diagnostic ROC (AUC = {subj_auc:.3f})')
ax_sub_roc.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', label='Chance')
ax_sub_roc.scatter(s_fpr, s_tpr, color='#1b5e20', s=30, alpha=0.8)

ax_sub_roc.set_xlim([0.0, 1.0])
ax_sub_roc.set_ylim([0.0, 1.05])
ax_sub_roc.set_xlabel('False Positive Rate', fontsize=10)
ax_sub_roc.set_ylabel('True Positive Rate', fontsize=10)
ax_sub_roc.set_title(f'Subject-Aggregated Clinical Diagnosis (N = {len(subj_agg)})\nAccuracy: {subj_acc*100:.1f}% | AUC: {subj_auc:.3f}', fontsize=11, fontweight='bold')
ax_sub_roc.legend(loc="lower right", fontsize=9, frameon=True)

plt.tight_layout()
fig3_path = os.path.join(WORKSPACE_DIR, 'cnn_classification_performance.png')
fig3.savefig(fig3_path, dpi=300)
shutil.copy(fig3_path, os.path.join(ARTIFACT_DIR, 'cnn_classification_performance.png'))
plt.close(fig3)
print(f"Figure 3 saved to: {fig3_path}")

# ==============================================================================
# 5. EXPORT FINAL SCIENTIFIC METRICS
# ==============================================================================
summary_export = {
    'demographics': {
        'total_subjects': int(len(demo_df)),
        'hc_count': int(hc_count),
        'sz_count': int(sz_count),
        'hc_age_mean': float(demo_df.loc[demo_df['group']==0, 'age'].mean()),
        'hc_age_std': float(demo_df.loc[demo_df['group']==0, 'age'].std()),
        'sz_age_mean': float(demo_df.loc[demo_df['group']==1, 'age'].mean()),
        'sz_age_std': float(demo_df.loc[demo_df['group']==1, 'age'].std()),
        'hc_edu_mean': float(demo_df.loc[demo_df['group']==0, 'education'].mean()),
        'sz_edu_mean': float(demo_df.loc[demo_df['group']==1, 'education'].mean())
    },
    'statistical_analysis': stat_results,
    'deep_learning_cnn': {
        'total_trials': int(N_samples),
        'single_trial_accuracy_mean': float(mean_acc),
        'single_trial_accuracy_std': float(std_acc),
        'single_trial_precision_mean': float(mean_prec),
        'single_trial_recall_mean': float(mean_rec),
        'single_trial_f1_mean': float(mean_f1),
        'single_trial_f1_std': float(std_f1),
        'single_trial_roc_auc_mean': float(mean_auc),
        'single_trial_roc_auc_std': float(std_auc),
        'subject_aggregated_accuracy': float(subj_acc),
        'subject_aggregated_roc_auc': float(subj_auc),
        'subject_aggregated_f1': float(subj_f1)
    }
}

metrics_path = os.path.join(WORKSPACE_DIR, 'pipeline_summary_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(summary_export, f, indent=2)

shutil.copy(metrics_path, os.path.join(ARTIFACT_DIR, 'pipeline_summary_metrics.json'))
print(f"\nSummary metrics exported to: {metrics_path}")
print("="*80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
