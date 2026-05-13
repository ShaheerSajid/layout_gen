#!/usr/bin/env bash
# Post-training sanity check: generate inverter/nand2/nor2 from a PPO
# checkpoint, run inspect_gds + --lvs-check on each. Prints a one-line
# verdict per cell.
#
# Usage:
#   ./scripts/eval_checkpoint.sh <checkpoint.zip>
#   ./scripts/eval_checkpoint.sh checkpoints/ppo_multi3_realdrc_gpu.zip
set -euo pipefail

CKPT="${1:-checkpoints/ppo_multi3_realdrc_gpu.zip}"
[ -f "$CKPT" ] || { echo "error: $CKPT not found" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-out/$(basename "$CKPT" .zip)}"
mkdir -p "$OUT_DIR"

CELLS="${CELLS:-inverter nand2 nor2}"
COMMON_FLAGS=(
    --device-cap 8 --net-cap 8
    --position-bins 16 --route-size-bins 4
    --mag-bins 8 --routing-mode std_cell
    --max-place-steps 4 --max-route-steps 6 --max-steps 14
    --poly-cap 128 --viol-cap 32 --target-cap 128
    --quiet
)
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

echo "checkpoint: $CKPT"
echo "out:        $OUT_DIR"
echo "cells:      $CELLS"
echo

printf "%-12s | %-8s | %-15s | %s\n" "cell" "polys" "inspect"  "lvs"
printf "%-12s-+-%-8s-+-%-15s-+-%s\n" "------------" "--------" "---------------" "------"

for cell in $CELLS; do
    gds="$OUT_DIR/${cell}.gds"
    python3 -m layout_gen.rl.scripts.generate \
        --topology "$cell" --checkpoint "$CKPT" \
        --cell-name "$cell" --out "$gds" \
        "${COMMON_FLAGS[@]}" $EXTRA_FLAGS \
        --lvs-check > "$OUT_DIR/${cell}.gen.log" 2>&1 || true

    lvs_line=$(grep '^\[lvs\]' "$OUT_DIR/${cell}.gen.log" | tail -1 || true)
    lvs_short="${lvs_line#\[lvs\] }"
    [ -z "$lvs_short" ] && lvs_short="(skipped)"

    inspect_out=$(python3 -m layout_gen.rl.scripts.inspect_gds "$gds" --strict 2>&1 \
                  | grep "totals:" | head -1 || true)

    polys=$(python3 -c "import gdspy; lib=gdspy.GdsLibrary(); lib.read_gds('$gds'); c=lib.top_level()[0]; print(len(c.polygons))" 2>/dev/null || echo "?")

    printf "%-12s | %-8s | %-15s | %s\n" "$cell" "$polys" "$inspect_out" "$lvs_short"
done
