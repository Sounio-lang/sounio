#!/bin/bash
#SBATCH --job-name=lemon_features
#SBATCH --output=/workspace/data/lemon/logs/features_%A_%a.out
#SBATCH --error=/workspace/data/lemon/logs/features_%A_%a.err
#SBATCH --array=1-220%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

# LEMON Feature Extraction Job Array (GPU-accelerated)
# Each task processes one subject's cached epochs

SUBJECTS_FILE="/workspace/data/lemon/subjects_all.txt"
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECTS_FILE)

if [ -z "$SUBJECT" ]; then
    echo "No subject found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Extracting features for: $SUBJECT (Task $SLURM_ARRAY_TASK_ID)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd /workspace/sounio

python3 -c "
import sys
sys.path.insert(0, '/workspace/sounio')
import json
import time
from pathlib import Path
import numpy as np

sid = '$SUBJECT'
cache_dir = Path('/workspace/data/lemon/preprocessed')
results_dir = Path('/workspace/data/lemon/results_gpu')
results_dir.mkdir(parents=True, exist_ok=True)

# Check cache
npy = cache_dir / f'{sid}_epochs.npy'
meta = cache_dir / f'{sid}_meta.json'

if not npy.exists():
    print(f'[{sid}] Cache not found, skipping')
    sys.exit(1)

# Check if already processed
feat_file = results_dir / f'{sid}_features.json'
if feat_file.exists():
    print(f'[{sid}] Features already extracted, skipping')
    sys.exit(0)

t0 = time.time()
try:
    epochs = np.load(npy)
    
    # Load subject index from endpoints
    import csv
    endpoints_file = '/workspace/data/lemon/endpoints.csv'
    subject_index = None
    with open(endpoints_file) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['subject_id'] == sid:
                subject_index = i
                break
    
    if subject_index is None:
        print(f'[{sid}] Not found in endpoints')
        sys.exit(1)
    
    # GPU feature extraction
    from scripts.research.ossm_168_dryrun.features_gpu import process_subject_gpu
    
    F1, F2, F3 = process_subject_gpu(epochs, subject_index, max_epochs=50)
    
    result = {
        'subject_id': sid,
        'subject_index': subject_index,
        'n_epochs': min(epochs.shape[0], 50),
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'wall_time_s': time.time() - t0,
    }
    
    feat_file.write_text(json.dumps(result, indent=2))
    print(f'[{sid}] Success: F1={F1:.4f} F2={F2:.4f} F3={F3:.4f} in {time.time()-t0:.1f}s')
    
except Exception as e:
    import traceback
    print(f'[{sid}] Error: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(1)
"
