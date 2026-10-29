#!/usr/bin/env bash
# Submit every configuration in a folder as one Slurm array job, one A100 per configuration.
# Usage: bash scripts/submit_set.sh configs/pilot [extra sbatch options, e.g. --time=01:00:00]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
set_dir="${1:?Usage: bash scripts/submit_set.sh <config folder> [sbatch options]}"
shift
: "${KV_LAB_SLURM_ACCOUNT:?Export KV_LAB_SLURM_ACCOUNT first (see docs/arc.md).}"
: "${KV_LAB_WORKDIR:?Export KV_LAB_WORKDIR first (see docs/arc.md).}"

configs=("$set_dir"/*.json)
if [ ! -f "${configs[0]}" ]; then
  echo "No configurations in $set_dir." >&2
  exit 1
fi
for config in "${configs[@]}"; do
  uv run --locked --extra inference python -m kv_cache_lab validate --config "$config"
done
# The final split runs only after the pilot gate; docs/experiment-spec.md says how to freeze it.
if grep -q '"split": "final"' "${configs[@]}" && [ ! -f "$set_dir/FROZEN" ]; then
  echo "$set_dir holds final-split configurations but has no FROZEN file. Pass the pilot gate first." >&2
  exit 1
fi

set_name="$(basename "$set_dir")"
mkdir -p "$KV_LAB_WORKDIR/logs" "$KV_LAB_WORKDIR/runs"
job_id="$(sbatch --parsable \
  --account="$KV_LAB_SLURM_ACCOUNT" --partition=a100_normal_q \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:1 --time=02:00:00 \
  --array="0-$((${#configs[@]} - 1))" --job-name="kv-$set_name" \
  --output="$KV_LAB_WORKDIR/logs/$set_name-%A_%a.log" --export=ALL \
  "$@" scripts/slurm_task.sh "$set_name" "${configs[@]}")"
job_id="${job_id%%;*}"

echo "Submitted array job $job_id: ${#configs[@]} configurations from $set_dir."
for index in "${!configs[@]}"; do
  echo "  task $index: ${configs[$index]}"
done
echo "Runs:   $KV_LAB_WORKDIR/runs/$set_name-$job_id/"
echo "Logs:   $KV_LAB_WORKDIR/logs/$set_name-${job_id}_<task>.log"
echo "Status: squeue -u \"\$USER\"   (when finished: sacct -j $job_id)"
echo "Table:  uv run --locked --extra inference python -m kv_cache_lab aggregate --runs \"$KV_LAB_WORKDIR/runs/$set_name-$job_id\" --output \"$KV_LAB_WORKDIR/results/$set_name-$job_id.csv\""
