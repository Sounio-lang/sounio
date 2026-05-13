#!/bin/bash
#SBATCH --job-name=lemon_preprocess
#SBATCH --output=/workspace/data/lemon/logs/preprocess_%A_%a.out
#SBATCH --error=/workspace/data/lemon/logs/preprocess_%A_%a.err
#SBATCH --array=1-220%28
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

# LEMON Preprocessing Job Array
# Each task processes one subject

SUBJECTS_FILE="/workspace/data/lemon/subjects_all.txt"
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECTS_FILE)

if [ -z "$SUBJECT" ]; then
    echo "No subject found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing subject: $SUBJECT (Task $SLURM_ARRAY_TASK_ID)"

cd /workspace/sounio

python3 -c "
import sys
sys.path.insert(0, '/workspace/sounio')
from scripts.research.ossm_168_dryrun import lemon_preprocess as lp
from pathlib import Path
import json
import numpy as np
import time

sid = '$SUBJECT'
raw_root = '/workspace/data/lemon/raw_eeg'
cache_dir = Path('/workspace/data/lemon/preprocessed')
cache_dir.mkdir(parents=True, exist_ok=True)

# Check if already cached
npy = cache_dir / f'{sid}_epochs.npy'
meta = cache_dir / f'{sid}_meta.json'

if npy.exists() and meta.exists():
    print(f'[{sid}] Already cached, skipping')
    sys.exit(0)

t0 = time.time()
try:
    res = lp.preprocess_subject(raw_root, sid)
    epochs = np.asarray(res.epochs, dtype=np.float64)
    
    prov = {
        'subject': sid,
        'sfreq': res.sfreq,
        'n_total_epochs': res.n_total_epochs,
        'n_retained_epochs': res.n_retained_epochs,
        'retention_rate': res.retention_rate(),
        'n_ica_rejected': res.n_ica_rejected,
        'n_ica_total': res.n_ica_total,
        'wall_time_s': time.time() - t0,
    }
    
    np.save(npy, epochs)
    meta.write_text(json.dumps(prov, indent=2))
    
    print(f'[{sid}] Success: {res.n_retained_epochs} epochs in {time.time()-t0:.1f}s')
except Exception as e:
    print(f'[{sid}] Error: {type(e).__name__}: {e}')
    sys.exit(1)
"
