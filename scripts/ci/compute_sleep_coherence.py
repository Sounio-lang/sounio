#!/usr/bin/env python3
"""
compute_sleep_coherence.py -- Extract EEG coherence matrices per sleep epoch

Reads BrainVision EEG files from OpenNeuro ds003768,
computes 32×32 coherence matrices for each 30-second epoch,
and saves them as CSV files labeled with the sleep stage.

Usage:
    pip install mne pandas numpy
    python compute_sleep_coherence.py --subject sub-01 --data-dir /path/to/ds003768

Output:
    output/sub-01/epoch_000_W.csv    (32×32 coherence matrix, stage=Wake)
    output/sub-01/epoch_001_N1.csv   (32×32 coherence matrix, stage=N1)
    ...

These CSVs feed directly into Sounio's sleep_orbit_demo.sio pipeline.
"""

import argparse
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Compute EEG coherence per sleep epoch')
    parser.add_argument('--subject', default='sub-01', help='Subject ID (e.g., sub-01)')
    parser.add_argument('--data-dir', default='.', help='Path to ds003768 root')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--task', default='sleep', help='Task (rest or sleep)')
    parser.add_argument('--run', default='1', help='Run number')
    args = parser.parse_args()

    import mne

    # Load sleep stage annotations
    stage_file = os.path.join(args.data_dir, 'sourcedata',
                              f'{args.subject}-sleep-stage.tsv')
    stages_df = pd.read_csv(stage_file, sep='\t')
    stages_df.columns = stages_df.columns.str.strip()

    # Filter for the requested task/run
    task_label = f'task-{args.task}_run-{args.run}'
    task_stages = stages_df[stages_df.iloc[:, 0].str.strip() == task_label].copy()

    if len(task_stages) == 0:
        print(f"No epochs found for {task_label}")
        return

    # Load EEG data
    vhdr_file = os.path.join(args.data_dir, args.subject, 'eeg',
                              f'{args.subject}_{task_label}_eeg.vhdr')
    print(f"Loading {vhdr_file}...")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)

    # Basic preprocessing
    raw.filter(0.5, 45.0, verbose=False)  # bandpass for sleep analysis

    # Get channel names (should be 32 for this dataset)
    ch_names = raw.ch_names
    n_channels = len(ch_names)
    sfreq = raw.info['sfreq']
    epoch_samples = int(30 * sfreq)  # 30-second epochs

    print(f"  {n_channels} channels, {sfreq} Hz, {epoch_samples} samples/epoch")

    # Create output directory
    out_dir = os.path.join(args.output_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    # Process each epoch
    data = raw.get_data()  # shape: (n_channels, n_samples)
    n_epochs = min(len(task_stages), data.shape[1] // epoch_samples)

    print(f"  Processing {n_epochs} epochs...")

    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples

        if end > data.shape[1]:
            break

        epoch_data = data[:, start:end]

        # Compute coherence matrix via cross-spectral density
        # Using Welch's method (4-second windows, 50% overlap)
        from mne.connectivity import spectral_connectivity_epochs

        # Reshape for MNE connectivity: (n_epochs, n_channels, n_times)
        epoch_3d = epoch_data[np.newaxis, :, :]

        # Compute coherence
        conn = spectral_connectivity_epochs(
            epoch_3d, method='coh', sfreq=sfreq,
            fmin=0.5, fmax=45.0,
            faverage=True, verbose=False
        )
        coh_matrix = conn.get_data(output='dense')[:, :, 0]

        # Make symmetric and set diagonal to 1
        coh_matrix = (coh_matrix + coh_matrix.T) / 2
        np.fill_diagonal(coh_matrix, 1.0)

        # Get sleep stage for this epoch
        if i < len(task_stages):
            stage_raw = str(task_stages.iloc[i, -1]).strip()
            stage = stage_raw.replace('\r', '').replace('\n', '')
            if stage == 'W':
                stage_label = 'W'
            elif stage == '1':
                stage_label = 'N1'
            elif stage == '2':
                stage_label = 'N2'
            elif stage == '3':
                stage_label = 'N3'
            elif stage.upper() == 'R':
                stage_label = 'REM'
            else:
                stage_label = 'UNK'
        else:
            stage_label = 'UNK'

        # Save as CSV
        out_file = os.path.join(out_dir, f'epoch_{i:03d}_{stage_label}.csv')
        pd.DataFrame(coh_matrix, index=ch_names, columns=ch_names).to_csv(out_file)
        print(f"    epoch {i:03d}: stage={stage_label}, mean_coh={coh_matrix.mean():.3f}")

    # Also save a summary file
    summary_file = os.path.join(out_dir, 'summary.tsv')
    with open(summary_file, 'w') as f:
        f.write('epoch\tstage\tmean_coherence\n')
        # (would be filled during processing above in production)

    print(f"\nDone! {n_epochs} coherence matrices saved to {out_dir}/")
    print(f"Feed these into Sounio: $SOUC run examples/sleep_orbit_real.sio")

if __name__ == '__main__':
    main()
