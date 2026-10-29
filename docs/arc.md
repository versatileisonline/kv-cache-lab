# Run the FullKV pilot on Virginia Tech ARC

Use your own VT PID and an approved Slurm allocation. Replace `YOURPID` and `YOUR_SLURM_ACCOUNT` below with your values. Do not put passwords, tokens, account details, or allocation IDs into committed files.

## Connect

Connect to the VT VPN if off campus, then open a terminal:

```bash
ssh YOURPID@tinkercliffs1.arc.vt.edu
```

Complete VT/Duo authentication. The [Open OnDemand portal](https://ood.arc.vt.edu/) is a browser alternative. ColdFront manages allocations; it is not the compute terminal. See [VT's connection FAQ](https://www.docs.arc.vt.edu/usage/00faq.html) if you cannot sign in.

## Get the repository

On ARC:

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/versatileisonline/kv-cache-lab.git
cd kv-cache-lab
git rev-parse HEAD
```

For an existing clone, use `git pull --ff-only`. Team members should use the same Git commit, `uv.lock`, and experiment configuration. Each machine needs its own environment; do not copy a Mac `.venv` to ARC.

## Request one GPU

Set the allocation account in the current terminal session:

```bash
export KV_LAB_SLURM_ACCOUNT="YOUR_SLURM_ACCOUNT"
sacctmgr show assoc where user="$USER" account="$KV_LAB_SLURM_ACCOUNT" format=Cluster,Account,User,Partition,QOS
sinfo -o '%P %a %l %G'
```

Verify your allocation supports the intended cluster/partition. This example requests one A100 GPU on TinkerCliffs, eight CPU cores, 32 GB of host RAM, and one hour:

```bash
srun --account="$KV_LAB_SLURM_ACCOUNT" --partition=a100_normal_q \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
  --gres=gpu:1 --time=01:00:00 --pty bash -l
```

Wait for a compute-node shell, then check the assignment:

```bash
hostname
echo "$SLURM_JOB_ID"
nvidia-smi
```

The request is an example, not a guaranteed assignment or a measured memory requirement. `--mem` refers to host RAM, not GPU VRAM. Keep GPU type consistent for paired comparisons. Prepare the environment and run the model inside the compute allocation. [VT GPU queues](https://www.docs.arc.vt.edu/resources/compute/00tinkercliffs.html), [login-node guidance](https://www.docs.arc.vt.edu/usage/abuse.html).

## Install the shared environment

Inside the compute-node shell:

```bash
cd "$HOME/projects/kv-cache-lab"
export KV_LAB_WORKDIR="/scratch/$USER/kv-cache-lab"
export UV_CACHE_DIR="$KV_LAB_WORKDIR/uv-cache"
export HF_HUB_CACHE="$KV_LAB_WORKDIR/model-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$KV_LAB_WORKDIR/runs"
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. Then use the committed Python selection and lockfile:

```bash
bash scripts/setup_env.sh
uv run --locked --extra inference python -m unittest discover -s tests -v
```

The environment selects Python 3.11, PyTorch 2.4.1 with CUDA 12.1 wheels on Linux, and Transformers 4.44.2. `uv.lock` pins the remaining packages. Verify the assigned GPU/driver supports this stack. Keep `CUDA_VISIBLE_DEVICES` as set by Slurm.

## Authenticate and run

Obtain access to [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) through your Hugging Face account. Authenticate interactively; do not paste a token into source files or command arguments:

```bash
uv run --locked --extra inference huggingface-cli login
run_dir="$KV_LAB_WORKDIR/runs/fullkv-$SLURM_JOB_ID"
bash scripts/run_baseline.sh --output "$run_dir"
cat "$run_dir/summary.json"
```

The runner loads the model/tokenizer revision in `configs/fullkv-pilot.json`. It runs ten 2,048-token retrieval prompts with FP16 weights/cache, SDPA, batch size one, greedy decoding, seed 42, two warmups, and up to 32 output tokens. Downloads and model loading are outside inference timing. The runner refuses GPUs other than an A100, because official comparisons use one GPU type.

## Run a whole set of methods

Each folder under `configs/` is a set: `configs/pilot/` holds FullKV, H2O at 25% and 12.5% budgets, and KIVI at 4 and 2 bits, all on the same 2K/4K/8K prompts. One command submits a set as a Slurm array, one A100 per method. Run it on the login node after exporting `KV_LAB_SLURM_ACCOUNT`, `KV_LAB_WORKDIR`, `UV_CACHE_DIR`, and `HF_HUB_CACHE`; the jobs inherit them:

```bash
bash scripts/submit_set.sh configs/pilot --time=01:00:00
```

Extra arguments go to `sbatch` (the default limit is two hours). The script prints the array job ID, where runs and logs go, and the command that combines the finished runs into one CSV:

```bash
uv run --locked --extra inference python -m kv_cache_lab aggregate \
  --runs "$KV_LAB_WORKDIR/runs/pilot-<jobid>" --output "$KV_LAB_WORKDIR/results/pilot-<jobid>.csv"
```

The CSV has one row per method and input length: accuracy, cache bytes and their ratio to FullKV, peak memory, and median timings (diagnostic only). `squeue -u "$USER"` lists queued (`PD`) and running (`R`) tasks, and `sacct -j <jobid>` shows their final states. `configs/final/` is refused until the pilot gate passes ([experiment spec](experiment-spec.md#pilot-gate)).

Review predictions, accuracy, cache reuse, memory, and the recorded environment in the run directory. Compare only runs with matching source/configuration and compatible hardware. Short-answer timing is diagnostic.

Scratch is temporary and differs between clusters. Copy results you need into private permanent storage before cleanup. Keep run metadata and usage records local unless intentionally reviewed for publication. [VT storage guidance](https://www.docs.arc.vt.edu/resources/storage.html)

Release the interactive allocation when finished:

```bash
exit
```
