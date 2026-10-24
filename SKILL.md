---
name: fidip-explorer
description: Operate the FiDIP (infant pose estimation) project locally and on Northeastern's Explorer SLURM cluster -- env setup, data/model paths, submit/monitor/cancel training and eval jobs. Read before running anything for this project.
---

# FiDIP Project Ops

Adapted from the course template (`../EECE-7398-Machine-Learning-with-Small-Data/SKILL.md`).
Placeholders marked `TODO` get filled in once cluster access is set up.

## Local machine (development / quick eval)

- Windows 11, RTX 4060 Laptop (8 GB VRAM), NVIDIA driver 592.
- Env: conda env `fidip` (Python 3.12, torch 2.3.1 + cu121 wheels).
  Run with `conda activate fidip` or directly via
  `C:\Users\nshef\anaconda3\envs\fidip\python.exe`.
- Project root: `C:\Users\nshef\personal_projects\fidip\Infant-Pose-Estimation` (moved out of
  OneDrive 2026-10-08). SyRIP data in `data/syrip/` (lowercase, matches `DATASET.ROOT`).
- `lib/nms` C/CUDA extensions are **not built** on Windows (no MSVC/nvcc). `lib/nms/nms.py`
  imports them optionally; only the pure-numpy `oks_nms` is used by the SyRIP pipeline.
- 8 GB VRAM: HRNet-W48 training at batch 20 will likely OOM locally -- do full training on
  the cluster, use the laptop for eval/demo/debugging (lower `TRAIN.BATCH_SIZE_PER_GPU`).

```bash
python gpu_test.py
python tools/test_adaptive_model.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
python tools/train_adaptive_model_hrnet.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

## Cluster connection facts

- **Host:** `login.explorer.northeastern.edu`, alias `explorer` in `~/.ssh/config`
- **Username:** `sheft.n` (groups: `users ipl_lab ML_Small_Data`; SLURM accounts `closas`, `ostadabbas`)
- **Auth:** passwordless SSH with `~/.ssh/id_ed25519` (set up 2026-10-08). Always use
  `ssh -o BatchMode=yes explorer '...'` so a broken key fails fast instead of prompting.

```
Host explorer
    HostName login.explorer.northeastern.edu
    User sheft.n
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

## Cluster environment

- Python env: **venv `~/venvs/fidip`** (Python 3.12 via `uv`, torch 2.3.1+cu121, same pins as
  the laptop; exact versions in `requirements-explorer.txt`). Scripts call
  `~/venvs/fidip/bin/python` directly -- no `module load`/activate needed.
- `uv` lives at `~/.local/bin/uv`. Rebuild the env with `sbatch ~/fidip_env_setup.sh` style job.
- **Login-node quirks:** outbound internet goes through an HTTP proxy that stalls large
  downloads, and conda's `defaults` channel needs a ToS acceptance (not accepted). Do package
  installs inside a job on the `short` partition, never on the login node.
- GPUs (partition `gpu`, 8 h max; `gpu-short` 2 h): V100 (16/32 GB), T4, A100, H200.
  torch 2.3.1+cu121 supports all of them. Request a type with `--gres=gpu:a100:1` etc.

## Filesystem layout (cluster)

Quotas verified 2026-10-23 with RC's `check-quota <path>` (only works from a `short`-partition
node, e.g. inside an sbatch job; `quota` itself fails on VAST):

| Storage | Path | Quota (soft / hard) | Use for |
|---|---|---|---|
| Home | `~/Infant-Pose-Estimation` | **0.07 / 0.07 TiB (~77 GB)** | code, venv; was 61 GB used (output 29 GB + eval500 25 GB) |
| Project | `/projects/ML_Small_Data` | **1.0 / 1.5 TiB** | storage granted to the user (RITM8406068). Members: `sheft.n`, `ostadabbas` only (+ RC admins), not class-wide. Right place for checkpoints/eval outputs |
| Scratch | `/scratch/sheft.n/fidip/logs/` | 49 / 50 TiB | SLURM stdout `<jobname>_<jobid>.out` (scratch is purged -- nothing precious here) |

**Code sync via git** (since 2026-10-23): repo `https://github.com/Sheftboyardee/fidip-reproduction`
(private). `origin` = that repo, `upstream` = `ostadabbas/Infant-Pose-Estimation`, branch `main`.
- Laptop: commit, then `git push`. Explorer: `ssh explorer 'cd ~/Infant-Pose-Estimation && git pull --ff-only'`.
- Explorer pulls over SSH with a **read-only deploy key** `~/.ssh/fidip_deploy` (alias `github-fidip`
  in Explorer's `~/.ssh/config`); it cannot push. Never edit tracked files on the cluster.
- Gitignored (stay on disk only): `data/`, `models/`, `output/`, `log/`, `eval500/`, `.env`. Data and
  weights are synced by hand once (tar over ssh, ~0.5 MB/s -- avoid re-sending).
- `.gitattributes` forces LF under `scripts/` so sbatch/shell files survive a Windows checkout.
- **Disk:** ~1.2 GB per HRNet run in `output/` (`checkpoint.pth`, `model_best.pth`,
  `final_state.pth` + debug images); each `eval500/<tag>/` holds a ~600 MB `feature.npy` dump
  written by `tools/test_adaptive_model.py` that nothing uses. Home is only ~77 GB -- check
  `du -sh ~/Infant-Pose-Estimation/{output,eval500}` before big sweeps; ask the user before
  deleting or moving anything.

## Job scripts (`scripts/`)

- `scripts/eval500.sbatch <list.txt>` -- scores many checkpoints on corrected Test500/Test400/Test100;
  list lines `<tag> <ckpt>`; writes `eval500/results_<jobid>.csv`. **Report Test400** (unbiased).
- `scripts/eval.sbatch <cfg> <model.pth> [overrides]` -- Test100 eval, GT boxes, 30 min.
- `scripts/train.sbatch <cfg> [overrides]` -- trains (one run is ~25 min on a V100: submit with `--time=01:00:00`, short requests start much sooner) with `tools/train_adaptive_model_{hrnet,mobile}.py`
  (picked from the cfg name), 4 h. HRNet initialises from `models/coco/w48_384×288.pth` via
  `TRAIN.CHECKPOINT` -- the shipped config leaves both `MODEL.PRETRAINED` and
  `TRAIN.CHECKPOINT` empty, which would mean random init. Override with `INIT=path sbatch ...`.
  New switches: `TRAIN.ADV_LOSS {neg_bce,confusion}`, `TRAIN.PRE_BN_FREEZE {False,True}`.
  Give every experiment its own `OUTPUT_DIR output/<exp>` so runs don't auto-resume each other.

## Common operations (cluster)

```bash
ssh explorer 'cd ~/Infant-Pose-Estimation && sbatch scripts/train.sbatch experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml OUTPUT_DIR output/repro_hrnet'
ssh explorer 'cd ~/Infant-Pose-Estimation && sbatch scripts/eval.sbatch experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml models/hrnet_fidip.pth'
ssh explorer 'squeue -u $USER --format="%.10i %.12P %.20j %.2t %.12M %R"'
ssh explorer 'sacct -u $USER --starttime=now-2days --format=JobID,JobName,State,ExitCode,Elapsed'
ssh explorer 'tail -n 50 /scratch/$USER/fidip/logs/fidip-train_<JOBID>.out'
ssh explorer 'scancel <JOBID>'        # confirm with user first
```

**Resume caveat:** `AUTO_RESUME: true` reloads `checkpoint.pth`, but the training script only
restores the pose net + its optimizer -- not the domain classifier (`model_d`/`optimizer_d`) or
the true best AP. A 20-epoch run fits in one 4 h job, so avoid splitting training across chained
jobs until resume is fixed.

## Active runs

| Run | Purpose | Job ID | Status | Output path |
|---|---|---|---|---|
| cluster eval check | released HRNet FiDIP on Test100 | 10929307 | COMPLETED, AP 93.6 | `/scratch/sheft.n/fidip/logs/fidip-eval_10929307.out` |
| repro A (FiDIP, LAMBDA 5e-4, PRE_EPOCH 10) | train reproduction, target 93.6 | 11329332 | COMPLETED 24 min, best AP 92.0 (ep 12) | `~/Infant-Pose-Estimation/output/repro_fidip/`, log `fidip-repro-A_11329332.out` |
| repro B (shipped cfg = finetune) | train reproduction, target 92.7 | 11329348 | COMPLETED 22 min, best AP 93.0 (ep 20) | `~/Infant-Pose-Estimation/output/repro_shipped/`, log `fidip-repro-B_11329348.out` |
| sweep rep2 / rep3 | repeats of A (noise) | 11330470 / 11330481 | COMPLETED: 91.3 / 91.9 | `output/rep2`, `output/rep3` |
| sweep lrstep | A + `LR_STEP [10,15]` | 11330486 | COMPLETED: 91.2 | `output/lrstep` |
| sweep lam5e-3 / lam5e-2 | A with LAMBDA x10 / x100 | 11330496 / 11330524 | diverged: 85.5 -> 73.9 / 86.4 -> 0.8 | `output/lam5e-3`, `output/lam5e-2` |
| round 2 | B repeats, pre-only, adv-only, bnfreeze, conf5e-4/3/2 | 11331011 11331062 11331060 11331069 11331067 11331081 11331086 11331331 | all COMPLETED (see AGENT.md round 2 table) | `output/<tag>` |
| round 3 | adv-only/bnfreeze repeats + Test500 eval of 31 ckpts | 11332165 11332174 11332183 11332187; eval 11332206 11332236 11332224 11332831 | all COMPLETED | `results/eval500*.csv` |
| round 4 | 40-epoch ft40-s1/s2, adv40-s1/s2 (+ dependent eval500-04) | 11333089 11333092 11333095 11333110 (+ replacement adv40-s3 11334290, evals 11333148 11334329) | all COMPLETED (adv40-s1 TIMEOUT, excluded) | `output/<tag>` |

## Guardrails

- Never print, log or commit `.env` contents or any token. Read secrets from `.env` only.
- Submit with `source ~/fidip_submit_lib.sh; submit <tag> [overrides]` (from the repo dir): it
  skips tags already in `sacct` and refuses to submit if `sacct` fails/returns nothing. Never
  re-run a whole sweep script to add one job. `TIME=01:30:00 submit ...` overrides the 1 h limit.
- **Avoid c2204-c2207** (V100-PCIe, ~3x slower than the SXM2 nodes for this model); the submit
  helper excludes them by default. To exclude on an already-pending job: `scontrol update JobId=<id> ExcNodeList="c[2204-2207]"`.
- Max **8 jobs queued+running per user** (QOSMaxSubmitJobPerUserLimit) -- batch sweeps accordingly.
- `sbatch` can fail transiently with "Unexpected message received" -- run `squeue -u $USER` before
  retrying so a job isn't submitted twice.
- Confirm with the user before `scancel`, deleting checkpoints, or overwriting results.
- `git pull --ff-only` on the cluster.
- No compute on the login node -- use `sbatch`/`srun`.
- Record every run in `AGENT.md` and every artifact in `MANIFEST.md`.
