#!/usr/bin/env bash
# Follow-up training run after the baseline (logs/ppo_multi3_realdrc_gpu.log).
# Same env + cells + BC checkpoint, but with strict-row-alignment ON and
# the soft row_delta reward exercised by training (the baseline trained
# before row_delta was committed).
#
# Usage:
#   ./scripts/train_followup.sh        # default 10k timesteps
#   STEPS=20000 ./scripts/train_followup.sh
set -euo pipefail

STEPS="${STEPS:-10000}"
RUN_NAME="${RUN_NAME:-ppo_multi3_realdrc_rowstrict}"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG=logs/${RUN_NAME}_${TAG}.log
PID_FILE=logs/${RUN_NAME}_${TAG}.pid

mkdir -p logs checkpoints
echo "[followup] launching ${RUN_NAME} for ${STEPS} timesteps"
echo "[followup] log: ${LOG}"

nohup python3 -m layout_gen.rl.scripts.train_ppo \
    --topologies inverter,nand2,nor2 \
    --enable-place --enable-route \
    --bc-init      checkpoints/bc_full16.pt \
    --ibrl-bc-init checkpoints/bc_full16.pt \
    --ibrl-beta-start 0.5 --ibrl-beta-end 0.0 \
    --total-timesteps "${STEPS}" --n-envs 3 --n-steps 128 \
    --max-place-steps 4 --max-route-steps 6 --max-steps 14 \
    --device-cap 8 --net-cap 8 --position-bins 16 --route-size-bins 4 \
    --mag-bins 8 --ent-coef 0.005 \
    --routing-mode std_cell \
    --strict-row-alignment \
    --device cuda \
    --out checkpoints/${RUN_NAME}.zip \
    > "${LOG}" 2>&1 &

echo "PID=$!" > "${PID_FILE}"
echo "[followup] PID=$(cat ${PID_FILE} | cut -d= -f2)  log=${LOG}"
