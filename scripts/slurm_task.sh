#!/usr/bin/env bash
# One task of a scripts/submit_set.sh array: runs the configuration at index SLURM_ARRAY_TASK_ID.
# Arguments: <set name> <config> [<config> ...]
set -euo pipefail
set_name="$1"
shift
configs=("$@")
config="${configs[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
output="$KV_LAB_WORKDIR/runs/$set_name-$SLURM_ARRAY_JOB_ID/$(basename "$config" .json)"
mkdir -p "$(dirname "$output")"
echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): $config -> $output"
bash scripts/run_baseline.sh --config "$config" --output "$output"
