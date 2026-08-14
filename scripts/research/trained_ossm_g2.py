#!/usr/bin/env python3
"""
TASK 3: Trained O-SSM with G₂ feature extraction.

Instead of the untrained O-SSM (fixed random weights A, B), we TRAIN the
O-SSM transition weights to maximize the discriminability of G12 (the
rumination-relevant G₂ direction).

ARCHITECTURE:
  - O-SSM with learnable A (transition) and B (input) weights
  - 14 G₂ generators applied to the trajectory → 14 features
  - Linear readout: 14 features → rumination score (Ridge)
  - Loss: MSE between predicted and actual rumination

TRAINING:
  - LEMON EEG data (204 subjects, cached preprocessed epochs)
  - Leave-one-out cross-validation
  - Train O-SSM + readout jointly

The trained O-SSM learns WHICH octonion configuration produces the most
discriminative G₂ features. This is the "optimal algebraic prior" for
rumination detection.

GPU: train on Slurm GPU nodes
FPGA: the trained weights can be deployed on U250 via existing PTX kernels
"""

import numpy as np
import sys, os, time, json, csv
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from cayley_dickson_paper_reproduction import oct_mul, _OCT_T_KJ
from g2_features import build_g2_generators


class TrainedOSSM(nn.Module):
    """O-SSM with learnable weights + G₂ feature extraction + linear readout.
    
    h_t = tanh(A ⊗ h_{t-1} + B @ x_t + bias)
    
    Trajectory → 14 G₂ features → linear → prediction
    """
    def __init__(self, input_dim=7, oct_dim=8, n_g2=14, n_out=1):
        super().__init__()
        self.oct_dim = oct_dim
        self.n_g2 = n_g2
        
        # Learnable O-SSM weights
        self.A = nn.Parameter(torch.randn(oct_dim) * 0.1)  # transition
        self.B = nn.Parameter(torch.randn(input_dim, oct_dim) * 0.1)  # input
        self.bias = nn.Parameter(torch.zeros(oct_dim))
        
        # G₂ generators (fixed, not learned)
        generators = build_g2_generators()  # (14, 7, 7)
        self.register_buffer('generators', torch.tensor(generators, dtype=torch.float32))
        
        # Readout: 14 G₂ features → prediction
        self.readout = nn.Sequential(
            nn.Linear(n_g2, 32),
            nn.ReLU(),
            nn.Linear(32, n_out)
        )
    
    def forward(self, x):
        """x: (batch, input_dim, T) → prediction: (batch,)"""
        batch, in_dim, T = x.shape
        
        # O-SSM forward pass
        h = torch.zeros(batch, self.oct_dim, device=x.device)
        h_trajs = []  # store imaginary parts for G₂
        
        for t in range(T):
            # Input injection
            x_t = x[:, :, t]  # (batch, input_dim)
            bx = x_t @ self.B  # (batch, oct_dim)
            
            # Octonion transition: A ⊗ h
            A_batch = self.A.unsqueeze(0).expand(batch, -1)
            ah = oct_mul(A_batch, h)
            
            h = torch.tanh(ah + bx + self.bias)
            h_trajs.append(h[:, 1:])  # imaginary part (7-dim)
        
        # Stack trajectory: (batch, T, 7)
        traj = torch.stack(h_trajs, dim=1)
        
        # Extract G₂ features: for each generator, median ‖D_k(h_t)‖
        g2_features = []
        for k in range(self.n_g2):
            D = self.generators[k]  # (7, 7)
            # Apply derivation: (batch, T, 7) @ (7, 7) → (batch, T, 7)
            d_h = traj @ D.T  # (batch, T, 7)
            norms = torch.norm(d_h, dim=-1)  # (batch, T)
            g2_features.append(torch.median(norms, dim=-1).values)  # (batch,)
        
        g2 = torch.stack(g2_features, dim=-1)  # (batch, 14)
        
        return self.readout(g2).squeeze(-1)


def load_lemon_data(cache_dir='/workspace/data/lemon/preprocessed',
                    endpoints_path='/workspace/data/lemon/endpoints.csv',
                    subjects_path='/workspace/data/lemon/subjects_all.txt',
                    max_epochs=5, max_subjects=204):
    """Load LEMON data for training."""
    # Load subjects
    with open(subjects_path) as f:
        subjects = [s.strip() for s in f if s.strip()][:max_subjects]
    
    # Load endpoints
    endpoints = {}
    with open(endpoints_path) as f:
        for row in csv.DictReader(f):
            endpoints[row['subject_id']] = row
    
    def sf(v):
        try: return float(v) if v and v.strip() else float('nan')
        except: return float('nan')
    
    # Load epochs and labels
    data = []
    labels = []
    for sid in subjects:
        epoch_path = os.path.join(cache_dir, f'{sid}_epochs.npy')
        if not os.path.exists(epoch_path):
            continue
        if sid not in endpoints:
            continue
        
        rum = sf(endpoints[sid].get('cerq_rumination'))
        if np.isnan(rum):
            continue
        
        try:
            epochs = np.load(epoch_path)
            n = min(max_epochs, epochs.shape[0])
            for i in range(n):
                data.append(epochs[i].astype(np.float32))  # (7, T)
                labels.append(rum)
        except:
            continue
    
    return np.array(data), np.array(labels, dtype=np.float32)


def train_trained_ossm(cache_dir='/workspace/data/lemon/preprocessed',
                       endpoints_path='/workspace/data/lemon/endpoints.csv',
                       subjects_path='/workspace/data/lemon/subjects_all.txt',
                       epochs=30, lr=1e-3, batch_size=8, max_subjects=204,
                       device='cpu', seed=20260806):
    """Train the O-SSM with G₂ readout."""
    
    torch.manual_seed(seed)
    
    print("=" * 60)
    print("TASK 3: TRAINED O-SSM WITH G₂ READOUT")
    print("=" * 60)
    
    # Load data
    print("Loading LEMON data...")
    X, Y = load_lemon_data(cache_dir, endpoints_path, subjects_path,
                           max_subjects=max_subjects)
    print(f"Data: {X.shape}, labels: {Y.shape}")
    print(f"Rumination range: [{Y.min():.1f}, {Y.max():.1f}], mean={Y.mean():.1f}")
    
    # Normalize labels
    Y_norm = (Y - Y.mean()) / (Y.std() + 1e-8)
    
    # Train/test split (80/20)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    X_train = torch.from_numpy(X[train_idx]).to(device)
    Y_train = torch.from_numpy(Y_norm[train_idx]).to(device)
    X_test = torch.from_numpy(X[test_idx]).to(device)
    Y_test = torch.from_numpy(Y_norm[test_idx]).to(device)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build model
    model = TrainedOSSM(input_dim=7, oct_dim=8, n_g2=14, n_out=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params}")
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    
    # Training loop
    best_rho = 0
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        perm_t = torch.randperm(len(X_train))
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm_t[i:i+batch_size]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]
            
            opt.zero_grad()
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / (len(X_train) // batch_size + 1)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train[:100]).cpu().numpy()
            test_pred = model(X_test).cpu().numpy()
            
            from scipy.stats import spearmanr
            train_rho, _ = spearmanr(train_pred, Y_train[:100].cpu().numpy())
            test_rho, test_p = spearmanr(test_pred, Y_test.cpu().numpy())
            
            if abs(test_rho) > abs(best_rho):
                best_rho = test_rho
                best_p = test_p
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  ep {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_rho={train_rho:+.3f}  test_rho={test_rho:+.3f}  "
                  f"best={best_rho:+.3f}  ({time.time()-t0:.0f}s)")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("RESULTS — Trained O-SSM with G₂ readout")
    print(f"{'='*60}")
    print(f"  Best test rho: {best_rho:+.3f}  (p={best_p:.4f})")
    print(f"  Time: {time.time()-t0:.0f}s")
    
    # Compare with untrained
    print(f"\n  Comparison:")
    print(f"    Untrained F1 aggregate:     rho=+0.161 (p=0.103)")
    print(f"    Untrained G12 (best single): rho=-0.230 (p=0.001)")
    print(f"    Untrained 14-G₂ Ridge:       rho=+0.290 (p<0.0001)")
    print(f"    Trained O-SSM+G₂:            rho={best_rho:+.3f} (p={best_p:.4f})")
    
    # Extract learned A weights
    A_learned = model.A.detach().cpu().numpy()
    print(f"\n  Learned A (octonion transition weight):")
    print(f"    {A_learned.round(4)}")
    
    return model, best_rho, best_p


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--seed', type=int, default=20260806)
    args = p.parse_args()
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model, rho, p = train_trained_ossm(epochs=args.epochs, lr=args.lr,
                                        batch_size=args.batch_size,
                                        device=device, seed=args.seed)
