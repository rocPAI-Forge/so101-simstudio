#!/bin/bash
# Optional: poll halo + DORobot until sweeps finish (or 10h), then write RESULTS.md.
# Logs stay under outputs/eval/hz_bench/ on each host.
set -euo pipefail
LAB_HZ="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$LAB_HZ/../../.." && pwd)"
BENCH="${LAB01_HZ_BENCH_LOG:-$ROOT/outputs/eval/hz_bench}"
DEADLINE=$((SECONDS + 36000))
log() { echo "[$(date -Is)] $*"; }

while (( SECONDS < DEADLINE )); do
    halo_done=0
    drob_done=0
    grep -q 'DONE halo' "$BENCH/halo_progress.txt" 2>/dev/null && halo_done=1
    if ssh -o BatchMode=yes -o ConnectTimeout=20 DORobot \
        'grep -q "DONE dorobot" /root/so101-simstudio/outputs/eval/hz_bench/dorobot_progress.txt'; then
        drob_done=1
    fi
    rsync -a --ignore-missing-args \
      DORobot:/root/so101-simstudio/outputs/eval/hz_bench/dorobot_*.log \
      DORobot:/root/so101-simstudio/outputs/eval/hz_bench/dorobot_progress.txt \
      DORobot:/root/so101-simstudio/outputs/eval/hz_bench/dorobot_skip.txt \
      DORobot:/root/so101-simstudio/outputs/eval/hz_bench/dorobot_osmesa.tsv \
      "$BENCH/" 2>/dev/null || true
    python3 "$LAB_HZ/summarize.py" > "$BENCH/summarize.out" 2>&1 || true
    log "halo_done=$halo_done drob_done=$drob_done"
    if [[ "$halo_done" == 1 && "$drob_done" == 1 ]]; then
        log "both done"
        python3 "$LAB_HZ/summarize.py" | tee "$BENCH/summarize.out"
        exit 0
    fi
    if [[ "$halo_done" != 1 ]] && ! pgrep -f 'labs/lab01_pnp/hz_bench.cmd' >/dev/null; then
        log "WARN halo hz_bench.cmd not running"
    fi
    sleep 120
done
log "deadline reached"
python3 "$LAB_HZ/summarize.py" | tee "$BENCH/summarize.out"
exit 1
