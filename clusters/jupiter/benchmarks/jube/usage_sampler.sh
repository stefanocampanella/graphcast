#!/usr/bin/env bash
# Sample node memory and GPU usage every INTERVAL seconds into CSV files in the current directory, until killed.
# Run once per node, alongside training, by train_bench.sbatch.in. Usage: usage_sampler.sh [INTERVAL]
set -u
INTERVAL=${1:-5}
HOST=$(hostname -s)
MEM_CSV="node_mem-$HOST.csv"
GPU_CSV="gpu-$HOST.csv"

echo "time,total_gb,used_gb,shared_gb,buff_cache_gb,available_gb" > "$MEM_CSV"
HAVE_GPU=false
if command -v nvidia-smi >/dev/null 2>&1; then
    HAVE_GPU=true
    echo "time,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w" > "$GPU_CSV"
fi

while true; do
    NOW=$(date +%s)
    free -b | awk -v t="$NOW" '/^Mem:/ { g = 2^30
        printf "%d,%.1f,%.1f,%.1f,%.1f,%.1f\n", t, $2 / g, $3 / g, $5 / g, $6 / g, $7 / g }' >> "$MEM_CSV"
    if $HAVE_GPU; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | sed "s/ //g; s/^/$NOW,/" >> "$GPU_CSV"
    fi
    sleep "$INTERVAL"
done
