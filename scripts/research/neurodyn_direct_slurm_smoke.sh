#!/usr/bin/env bash
set -euo pipefail

# Run one Brain O-SSM synthetic smoke directly through Slurm.
#
# This intentionally executes from worker-local /tmp and streams the result
# bundle back over stdout in a second srun call. OrangeFS is not used as the
# canonical result path because this lane observed zero-prefixed tgz writes
# during a prior smoke.

RUN_ID="${RUN_ID:?set RUN_ID}"
MANIFEST_PATH="${MANIFEST_PATH:?set MANIFEST_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
NODE="${NODE:-gpuorangefs-r770-proxmox}"
PARTITION="${PARTITION:-all}"
CPUS="${CPUS:-4}"
MEM="${MEM:-8G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
PAIRS_EXPECTED="${PAIRS_EXPECTED:-}"

TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DROP_CHANNEL_FRAC="${DROP_CHANNEL_FRAC:-0.0}"
NOISE_STD="${NOISE_STD:-0.0}"
GLOBAL_TRAIN_EPOCHS="${GLOBAL_TRAIN_EPOCHS:-24}"
GLOBAL_TRAIN_LR="${GLOBAL_TRAIN_LR:-0.010}"
GLOBAL_CORE_LR_SCALE="${GLOBAL_CORE_LR_SCALE:-0.50}"
OCT_PROFILE_ID="${OCT_PROFILE_ID:-4}"
OCT_TRAIN_MODE="${OCT_TRAIN_MODE:-4}"
OCT_TRAIN_EPOCHS="${OCT_TRAIN_EPOCHS:-36}"
OCT_TRAIN_LR="${OCT_TRAIN_LR:-0.012}"
OCT_CORE_LR_SCALE="${OCT_CORE_LR_SCALE:-0.65}"
OCT_WARMUP_EPOCHS="${OCT_WARMUP_EPOCHS:-6}"
OCT_INIT_PRESET="${OCT_INIT_PRESET:-1}"
OCT_ASSOC_REG="${OCT_ASSOC_REG:-0.04}"
OCT_ASSOC_TARGET="${OCT_ASSOC_TARGET:-0.86}"
OCT_ASSOC_SCHEDULE_MODE="${OCT_ASSOC_SCHEDULE_MODE:-1}"
OCT_ASSOC_SCHEDULE_START="${OCT_ASSOC_SCHEDULE_START:-1.10}"
OCT_ASSOC_SCHEDULE_MID="${OCT_ASSOC_SCHEDULE_MID:-0.86}"
OCT_ASSOC_SCHEDULE_END="${OCT_ASSOC_SCHEDULE_END:-0.74}"
OCT_ASSOC_DRIFT_REG="${OCT_ASSOC_DRIFT_REG:-0.03}"
OCT_HEAD_TRUST_REGION="${OCT_HEAD_TRUST_REGION:-0.05}"
OCT_HEAD_RENORM="${OCT_HEAD_RENORM:-1}"
OCT_ASSOCIATIVE_PROJECTION="${OCT_ASSOCIATIVE_PROJECTION:-0}"
OCT_STATE_ORDER_AUX="${OCT_STATE_ORDER_AUX:-0.0}"
OCT_ASSOCIATOR_SIGN_AUX="${OCT_ASSOCIATOR_SIGN_AUX:-0.0}"
OCT_ASSOCIATOR_FIXED_DIM_AUX="${OCT_ASSOCIATOR_FIXED_DIM_AUX:-0.0}"
OCT_ASSOCIATOR_FIXED_DIM_READOUT_AUX="${OCT_ASSOCIATOR_FIXED_DIM_READOUT_AUX:-0.0}"
OCT_ASSOCIATOR_FIXED_ORIENTATION_AUX="${OCT_ASSOCIATOR_FIXED_ORIENTATION_AUX:-0.0}"
OCT_ASSOCIATOR_FIXED_ORIENTATION_READOUT_AUX="${OCT_ASSOCIATOR_FIXED_ORIENTATION_READOUT_AUX:-0.0}"
OCT_ASSOCIATOR_FIXED_ORIENTATION_ALIGN_READOUT_AUX="${OCT_ASSOCIATOR_FIXED_ORIENTATION_ALIGN_READOUT_AUX:-0.0}"
OCT_ASSOCIATOR_ORIENTATION_HEAD_AUX="${OCT_ASSOCIATOR_ORIENTATION_HEAD_AUX:-0.0}"
OCT_ASSOCIATOR_ORIENTATION_HEAD_EPOCHS="${OCT_ASSOCIATOR_ORIENTATION_HEAD_EPOCHS:-0}"
OCT_ASSOCIATOR_FIXED_ORIENTATION_RAMP_EPOCHS="${OCT_ASSOCIATOR_FIXED_ORIENTATION_RAMP_EPOCHS:-0}"
OCT_ASSOCIATOR_FIXED_DIM="${OCT_ASSOCIATOR_FIXED_DIM:-4}"
OCT_ASSOCIATOR_VECTOR_PAIR_AUX="${OCT_ASSOCIATOR_VECTOR_PAIR_AUX:-0.0}"
OCT_PAIR_CONTRAST_AUX="${OCT_PAIR_CONTRAST_AUX:-0.08}"
OCT_TRAIN_NOISE_STD="${OCT_TRAIN_NOISE_STD:-0.0}"
OCT_RELATION_PRESERVE_AUX="${OCT_RELATION_PRESERVE_AUX:-0.0}"
OCT_RELATION_TARGET_AUX="${OCT_RELATION_TARGET_AUX:-0.0}"
OCT_RELATION_MARGIN_AUX="${OCT_RELATION_MARGIN_AUX:-0.0}"
OCT_RELATION_IDENTITY_AUX="${OCT_RELATION_IDENTITY_AUX:-0.0}"
OCT_RELATION_IDENTITY_SRC_AUX="${OCT_RELATION_IDENTITY_SRC_AUX:--1.0}"
OCT_RELATION_IDENTITY_DST_AUX="${OCT_RELATION_IDENTITY_DST_AUX:--1.0}"
OCT_RELATION_IDENTITY_START_EPOCH="${OCT_RELATION_IDENTITY_START_EPOCH:-0}"
OCT_RELATION_IDENTITY_RAMP_EPOCHS="${OCT_RELATION_IDENTITY_RAMP_EPOCHS:-0}"
OCT_RELATION_IDENTITY_TIE_MARGIN="${OCT_RELATION_IDENTITY_TIE_MARGIN:-0.0}"
OCT_RELATION_IDENTITY_GATE_MARGIN="${OCT_RELATION_IDENTITY_GATE_MARGIN:-0.0}"
OCT_RELATION_IDENTITY_GATE_FLOOR="${OCT_RELATION_IDENTITY_GATE_FLOOR:-1.0}"
OCT_RELATION_IDENTITY_TASK_GUARD="${OCT_RELATION_IDENTITY_TASK_GUARD:-0}"
OCT_RELATION_IDENTITY_TASK_GUARD_TOL="${OCT_RELATION_IDENTITY_TASK_GUARD_TOL:-0.0}"
OCT_RELATION_READOUT_CORRECT_STEPS="${OCT_RELATION_READOUT_CORRECT_STEPS:-0}"
OCT_RELATION_READOUT_CORRECT_LR_SCALE="${OCT_RELATION_READOUT_CORRECT_LR_SCALE:-1.0}"
OCT_ASSOCIATOR_READOUT_CORRECT_STEPS="${OCT_ASSOCIATOR_READOUT_CORRECT_STEPS:-0}"
OCT_ASSOCIATOR_READOUT_CORRECT_LR_SCALE="${OCT_ASSOCIATOR_READOUT_CORRECT_LR_SCALE:-1.0}"
OCT_ASSOCIATOR_READOUT_ALIGN_EPOCHS="${OCT_ASSOCIATOR_READOUT_ALIGN_EPOCHS:-0}"
OCT_ASSOCIATOR_READOUT_ALIGN_LR_SCALE="${OCT_ASSOCIATOR_READOUT_ALIGN_LR_SCALE:-1.0}"
OCT_RELATION_MARGIN_GATE_CAP="${OCT_RELATION_MARGIN_GATE_CAP:-2.0}"
OCT_RELATION_MARGIN_START_EPOCH="${OCT_RELATION_MARGIN_START_EPOCH:-0}"
OCT_RELATION_MARGIN_RAMP_EPOCHS="${OCT_RELATION_MARGIN_RAMP_EPOCHS:-0}"
OCT_RELATION_FREEZE_AFTER_EPOCH="${OCT_RELATION_FREEZE_AFTER_EPOCH:--1}"
OCT_RELATION_POST_FREEZE_SCALE="${OCT_RELATION_POST_FREEZE_SCALE:-0.0}"
OCT_BINARY_LR_SCALE_AFTER_EPOCH="${OCT_BINARY_LR_SCALE_AFTER_EPOCH:--1}"
OCT_BINARY_LR_POST_SCALE="${OCT_BINARY_LR_POST_SCALE:-1.0}"
OCT_RELATION_TASK_GUARD="${OCT_RELATION_TASK_GUARD:-0}"
OCT_RELATION_TASK_GUARD_TOL="${OCT_RELATION_TASK_GUARD_TOL:-0.0}"
OCT_RELATION_TARGET_SRC_POS="${OCT_RELATION_TARGET_SRC_POS:-2}"
OCT_RELATION_TARGET_DST_POS="${OCT_RELATION_TARGET_DST_POS:-0}"
OCT_INPUT_PROJ_MODE="${OCT_INPUT_PROJ_MODE:-4}"
OCT_PROJ_LR_SCALE="${OCT_PROJ_LR_SCALE:-0.6}"
OCT_PROJ_STRUCTURED_SCALE="${OCT_PROJ_STRUCTURED_SCALE:-0.35}"
OCT_PROJ_DELTA_SCALE="${OCT_PROJ_DELTA_SCALE:-0.18}"
OCT_PROJ_HYBRID_SCALE="${OCT_PROJ_HYBRID_SCALE:-0.12}"
H_TRAIN_EPOCHS="${H_TRAIN_EPOCHS:-24}"
H_TRAIN_LR="${H_TRAIN_LR:-0.010}"
H_CORE_LR_SCALE="${H_CORE_LR_SCALE:-0.50}"
H_WARMUP_EPOCHS="${H_WARMUP_EPOCHS:-4}"
H_INIT_PRESET="${H_INIT_PRESET:-1}"
H_INPUT_PROJ_MODE="${H_INPUT_PROJ_MODE:-0}"
H_PROJ_LR_SCALE="${H_PROJ_LR_SCALE:-0.6}"
READOUT_ASSOC_SCALE="${READOUT_ASSOC_SCALE:-0.0}"
READOUT_MEAN_SCALE="${READOUT_MEAN_SCALE:-0.0}"
READOUT_DELTA_SCALE="${READOUT_DELTA_SCALE:-0.0}"
READOUT_FLAT_SCALE="${READOUT_FLAT_SCALE:-0.0}"
TRACE_HIDDEN_STATE="${TRACE_HIDDEN_STATE:-1}"
TRACE_READOUT_ALL_FOLDS="${TRACE_READOUT_ALL_FOLDS:-1}"
MIN_SUBJECTS="${MIN_SUBJECTS:-112}"

SNAPSHOT_DIR="${SNAPSHOT_DIR:-/tmp/${RUN_ID}-snapshot}"
PAYLOAD_TGZ="${PAYLOAD_TGZ:-/tmp/${RUN_ID}.tgz}"
RESULT_TGZ="${RESULT_TGZ:-/tmp/${RUN_ID}-results.tgz}"

rm -rf "${SNAPSHOT_DIR}" "${PAYLOAD_TGZ}" "${RESULT_TGZ}" "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
OUT_ROOT="${SNAPSHOT_DIR}" SOUNIO_DIR="${SOUNIO_DIR}" ABIDE_MANIFEST_PATH="${MANIFEST_PATH}" \
  bash "${SOUNIO_DIR}/scripts/gpu/prepare_abide_campaign_snapshot.sh" >/dev/null
tar -C "${SNAPSHOT_DIR}" -czf "${PAYLOAD_TGZ}" .
tar -tzf "${PAYLOAD_TGZ}" >/dev/null

base64 -w0 "${PAYLOAD_TGZ}" | srun \
  --chdir=/tmp \
  -p "${PARTITION}" \
  -w "${NODE}" \
  -N1 -n1 -c "${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  bash -lc "
set -euo pipefail
RUN_ID='${RUN_ID}'
ROOT=\"/tmp/\${RUN_ID}\"
rm -rf \"\${ROOT}\"
mkdir -p \"\${ROOT}\"
base64 -d > \"\${ROOT}/payload.tgz\"
tar -xzf \"\${ROOT}/payload.tgz\" -C \"\${ROOT}\"
cd \"\${ROOT}\"
python3 -c 'from pathlib import Path; p=Path(\"scripts/research/abide_campaign_lib.py\"); data=p.read_bytes(); assert data.startswith(b\"#!/usr/bin/env python3\") and b\"\\0\" not in data[:200], data[:40]'
cat > abide_run_config.tsv <<'CFG'
train_fraction=${TRAIN_FRACTION}
drop_channel_frac=${DROP_CHANNEL_FRAC}
noise_std=${NOISE_STD}
global_train_epochs=${GLOBAL_TRAIN_EPOCHS}
global_train_lr=${GLOBAL_TRAIN_LR}
global_core_lr_scale=${GLOBAL_CORE_LR_SCALE}
oct_profile_id=${OCT_PROFILE_ID}
oct_train_mode=${OCT_TRAIN_MODE}
oct_train_epochs=${OCT_TRAIN_EPOCHS}
oct_train_lr=${OCT_TRAIN_LR}
oct_core_lr_scale=${OCT_CORE_LR_SCALE}
oct_warmup_epochs=${OCT_WARMUP_EPOCHS}
oct_init_preset=${OCT_INIT_PRESET}
oct_assoc_reg=${OCT_ASSOC_REG}
oct_assoc_target=${OCT_ASSOC_TARGET}
oct_assoc_schedule_mode=${OCT_ASSOC_SCHEDULE_MODE}
oct_assoc_schedule_start=${OCT_ASSOC_SCHEDULE_START}
oct_assoc_schedule_mid=${OCT_ASSOC_SCHEDULE_MID}
oct_assoc_schedule_end=${OCT_ASSOC_SCHEDULE_END}
oct_assoc_drift_reg=${OCT_ASSOC_DRIFT_REG}
oct_head_trust_region=${OCT_HEAD_TRUST_REGION}
oct_head_renorm=${OCT_HEAD_RENORM}
oct_associative_projection=${OCT_ASSOCIATIVE_PROJECTION}
oct_state_order_aux=${OCT_STATE_ORDER_AUX}
oct_associator_sign_aux=${OCT_ASSOCIATOR_SIGN_AUX}
oct_associator_fixed_dim_aux=${OCT_ASSOCIATOR_FIXED_DIM_AUX}
oct_associator_fixed_dim_readout_aux=${OCT_ASSOCIATOR_FIXED_DIM_READOUT_AUX}
oct_associator_fixed_orientation_aux=${OCT_ASSOCIATOR_FIXED_ORIENTATION_AUX}
oct_associator_fixed_orientation_readout_aux=${OCT_ASSOCIATOR_FIXED_ORIENTATION_READOUT_AUX}
oct_associator_fixed_orientation_align_readout_aux=${OCT_ASSOCIATOR_FIXED_ORIENTATION_ALIGN_READOUT_AUX}
oct_associator_orientation_head_aux=${OCT_ASSOCIATOR_ORIENTATION_HEAD_AUX}
oct_associator_orientation_head_epochs=${OCT_ASSOCIATOR_ORIENTATION_HEAD_EPOCHS}
oct_associator_fixed_orientation_ramp_epochs=${OCT_ASSOCIATOR_FIXED_ORIENTATION_RAMP_EPOCHS}
oct_associator_fixed_dim=${OCT_ASSOCIATOR_FIXED_DIM}
oct_associator_vector_pair_aux=${OCT_ASSOCIATOR_VECTOR_PAIR_AUX}
oct_pair_contrast_aux=${OCT_PAIR_CONTRAST_AUX}
oct_train_noise_std=${OCT_TRAIN_NOISE_STD}
oct_relation_preserve_aux=${OCT_RELATION_PRESERVE_AUX}
oct_relation_target_aux=${OCT_RELATION_TARGET_AUX}
oct_relation_margin_aux=${OCT_RELATION_MARGIN_AUX}
oct_relation_identity_aux=${OCT_RELATION_IDENTITY_AUX}
oct_relation_identity_src_aux=${OCT_RELATION_IDENTITY_SRC_AUX}
oct_relation_identity_dst_aux=${OCT_RELATION_IDENTITY_DST_AUX}
oct_relation_identity_start_epoch=${OCT_RELATION_IDENTITY_START_EPOCH}
oct_relation_identity_ramp_epochs=${OCT_RELATION_IDENTITY_RAMP_EPOCHS}
oct_relation_identity_tie_margin=${OCT_RELATION_IDENTITY_TIE_MARGIN}
oct_relation_identity_gate_margin=${OCT_RELATION_IDENTITY_GATE_MARGIN}
oct_relation_identity_gate_floor=${OCT_RELATION_IDENTITY_GATE_FLOOR}
oct_relation_identity_task_guard=${OCT_RELATION_IDENTITY_TASK_GUARD}
oct_relation_identity_task_guard_tol=${OCT_RELATION_IDENTITY_TASK_GUARD_TOL}
oct_relation_readout_correct_steps=${OCT_RELATION_READOUT_CORRECT_STEPS}
oct_relation_readout_correct_lr_scale=${OCT_RELATION_READOUT_CORRECT_LR_SCALE}
oct_associator_readout_correct_steps=${OCT_ASSOCIATOR_READOUT_CORRECT_STEPS}
oct_associator_readout_correct_lr_scale=${OCT_ASSOCIATOR_READOUT_CORRECT_LR_SCALE}
oct_associator_readout_align_epochs=${OCT_ASSOCIATOR_READOUT_ALIGN_EPOCHS}
oct_associator_readout_align_lr_scale=${OCT_ASSOCIATOR_READOUT_ALIGN_LR_SCALE}
oct_relation_margin_gate_cap=${OCT_RELATION_MARGIN_GATE_CAP}
oct_relation_margin_start_epoch=${OCT_RELATION_MARGIN_START_EPOCH}
oct_relation_margin_ramp_epochs=${OCT_RELATION_MARGIN_RAMP_EPOCHS}
oct_relation_freeze_after_epoch=${OCT_RELATION_FREEZE_AFTER_EPOCH}
oct_relation_post_freeze_scale=${OCT_RELATION_POST_FREEZE_SCALE}
oct_binary_lr_scale_after_epoch=${OCT_BINARY_LR_SCALE_AFTER_EPOCH}
oct_binary_lr_post_scale=${OCT_BINARY_LR_POST_SCALE}
oct_relation_task_guard=${OCT_RELATION_TASK_GUARD}
oct_relation_task_guard_tol=${OCT_RELATION_TASK_GUARD_TOL}
oct_relation_target_src_pos=${OCT_RELATION_TARGET_SRC_POS}
oct_relation_target_dst_pos=${OCT_RELATION_TARGET_DST_POS}
oct_input_proj_mode=${OCT_INPUT_PROJ_MODE}
oct_proj_lr_scale=${OCT_PROJ_LR_SCALE}
oct_proj_structured_scale=${OCT_PROJ_STRUCTURED_SCALE}
oct_proj_delta_scale=${OCT_PROJ_DELTA_SCALE}
oct_proj_hybrid_scale=${OCT_PROJ_HYBRID_SCALE}
h_train_epochs=${H_TRAIN_EPOCHS}
h_train_lr=${H_TRAIN_LR}
h_core_lr_scale=${H_CORE_LR_SCALE}
h_warmup_epochs=${H_WARMUP_EPOCHS}
h_init_preset=${H_INIT_PRESET}
h_input_proj_mode=${H_INPUT_PROJ_MODE}
h_proj_lr_scale=${H_PROJ_LR_SCALE}
readout_assoc_scale=${READOUT_ASSOC_SCALE}
readout_mean_scale=${READOUT_MEAN_SCALE}
readout_delta_scale=${READOUT_DELTA_SCALE}
readout_flat_scale=${READOUT_FLAT_SCALE}
trace_hidden_state=${TRACE_HIDDEN_STATE}
trace_readout_all_folds=${TRACE_READOUT_ALL_FOLDS}
min_subjects=${MIN_SUBJECTS}
CFG
python3 scripts/research/normalize_abide_manifest.py --input abide_source_manifest.tsv --output abide_roi_manifest.tsv --layout flat
if [[ -n '${PAIRS_EXPECTED}' ]]; then
  export PAIRS_EXPECTED_VALUE='${PAIRS_EXPECTED}'
  python3 - <<'PY'
import csv, os
expected = int(os.environ.get('PAIRS_EXPECTED_VALUE', '0'))
with open('abide_roi_manifest.tsv', newline='') as f:
    rows = list(csv.DictReader((line for line in f if not line.startswith('#')), delimiter='\t'))
pairs = {row['subject_id'].split('__', 1)[0] for row in rows}
if len(pairs) != expected:
    raise SystemExit(f'expected {expected} pairs, got {len(pairs)}')
PY
fi
python3 scripts/research/abide_manifest_quality_gate.py --manifest abide_roi_manifest.tsv --min-nonzero-frac 0.01 --min-variance 1e-10 > manifest_quality_gate.log
export SOUNIO_SOUC_ENGINE=lean_single
./bin/souc compile examples/brain_ossm_abide.sio -o brain_ossm_abide.elf > compile.log 2>&1
chmod +x brain_ossm_abide.elf
set +e
./brain_ossm_abide.elf > brain_ossm_abide.raw.txt 2>&1
rc=\$?
set -e
printf '%s\n' \"\${rc}\" > run.rc
if [[ \"\${rc}\" -ne 0 ]]; then
  tail -80 brain_ossm_abide.raw.txt >&2
  exit \"\${rc}\"
fi
python3 scripts/research/parse_brain_ossm_abide_output.py --input brain_ossm_abide.raw.txt --output-dir results > parse.log
sha256sum abide_source_manifest.tsv abide_roi_manifest.tsv abide_run_config.tsv brain_ossm_abide.raw.txt run.rc results/overall_metrics.tsv > SHA256SUMS.output
cat results/overall_metrics.tsv
"

srun \
  --chdir=/tmp \
  -p "${PARTITION}" \
  -w "${NODE}" \
  -N1 -n1 -c1 \
  --mem=1G \
  --time=00:05:00 \
  tar -C "/tmp/${RUN_ID}" -czf - \
    abide_source_manifest.tsv abide_roi_manifest.tsv abide_run_config.tsv \
    manifest_quality_gate.log compile.log brain_ossm_abide.raw.txt run.rc \
    parse.log results SHA256SUMS.output > "${RESULT_TGZ}"

tar -tzf "${RESULT_TGZ}" >/dev/null
tar -xzf "${RESULT_TGZ}" -C "${OUTPUT_DIR}"
printf 'output_dir=%s\n' "${OUTPUT_DIR}"
