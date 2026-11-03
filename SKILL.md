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
- **QOS limit: 4 running jobs and 8 queued per user** (QOSMaxJobsPerUserLimit). The sbatch error
  "Unexpected message received" often means the job *was* queued: check `squeue` before resubmitting.
- **Second venv `~/venvs/sapiens2`** (torch 2.7.1+cu126, transformers 5.x) is used only for
  Sapiens2 (`scripts/sapiens2_setup.sbatch`). Do not mix it with `~/venvs/fidip` (transformers 4.53.3).

## Filesystem layout (cluster)

Quotas verified 2026-10-23 with RC's `check-quota <path>` (only works from a `short`-partition
node, e.g. inside an sbatch job; `quota` itself fails on VAST):

| Storage | Path | Quota (soft / hard) | Use for |
|---|---|---|---|
| Home | `~/Infant-Pose-Estimation` | **0.07 / 0.07 TiB (~77 GB)** | code, `data/`, `models/`, `log/`, venv (~7 GB used after the 2026-10-23 move) |
| Project | `/projects/ML_Small_Data/sheft.n/fidip/{output,eval500}` | **1.0 / 1.5 TiB** | storage granted to the user (RITM8406068); members `sheft.n`, `ostadabbas` only (+ RC admins), not class-wide. **Checkpoints and eval outputs live here**; `~/Infant-Pose-Estimation/output` and `.../eval500` are symlinks to it, so all scripts/paths are unchanged |
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
  `final_state.pth` + debug images) -- now on the 1 TiB project space via symlink.
  `eval500.sbatch` deletes the unused ~600 MB `feature.npy` per checkpoint after scoring.
  Never write bulky outputs to a real directory in home (~77 GB). Ask the user before deleting
  or moving anything.

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

- Stage A/D additions (2026-10-26):
  - `scripts/stageA_soups.sbatch [spec]` builds weight soups with `scripts/weight_average.py`
    into `output/soups/`. It also writes `output/soups/eval_{raw,bn}.txt` for `eval500.sbatch`.
  - `scripts/vitpose_setup.sbatch` (short partition) installs transformers 4.53.3 into the venv
    and downloads `scripts/vitpose_models.txt` into the HF cache
    `/projects/ML_Small_Data/sheft.n/fidip/hf_cache`. All jobs set `HF_HUB_OFFLINE=1`.
  - `scripts/vitpose_zeroshot.sbatch` scores zero-shot ViTPose using `tools/test_vitpose.py`.
  - **ViTPose fine-tuning:** `train.sbatch` with the vitpose cfg runs `tools/train_vitpose.py`:
    ```
    C=experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml
    submit <tag> MODEL.HF_NAME usyd-community/vitpose-plus-{large,huge} TRAIN.LR ...
    ```
    Use `TIME=01:30:00`.
  - **384x288 ViTPose** (`vitpose_plus_384x288_infant.yaml`, AUTO_RESUME true): position embeddings
    are interpolated on load. Add `TRAIN.GRAD_CKPT True` for ++-H on 32 GB GPUs. A run takes about 4 h,
    so submit part 1 with `submit`, then 1-2 more `sbatch -J fidip-<tag>-pN --dependency=afterany:<prev>`
    with the same overrides and OUTPUT_DIR. Each resumes from `checkpoint.pth`; a spare one exits.
  - **Evaluating ViTPose checkpoints:** pass the same HF_NAME, otherwise the wrong architecture
    is built and `strict=False` loading silently drops the weights:
    ```
    CFG=<vitpose cfg> OPTS="MODEL.HF_NAME <same as training>" sbatch scripts/eval500.sbatch list.txt
    ```
  - The HRNet training script freezes the first 109 parameters (stem + stage 1). This is hard-coded
    upstream; `train_vitpose.py` does not do it.

- Round 12 additions (2026-10-27/28):
  - **Grouped CV for method selection.**
    - `scripts/make_cv_folds.py` writes `data/syrip_cv/f<k>/` (5 folds of the 200 real images, grouped by source).
    - Train a fold with `submit cv-<tag>-f<k> <overrides> DATASET.ROOT data/syrip_cv/f<k>/ DATASET.TEST_SET heldout_infant`.
    - Copy results locally with `scratchpad`-style `ssh -n ... cat`. `tar` into the symlinked `output/` fails across devices.
    - Score with `python scripts/cv_score.py --tags cv-h20 cv-<tag> --ref cv-h20`. It reports pooled OOF AP and the advance rule.
  - **Overlap audit and bootstrap.**
    - `scripts/audit_overlap.py` writes `output/audit12/overlap/` (test_subsets.json, eligible_unlabeled.json).
    - `scripts/bootstrap_ap.py --a ref.json --b cand.json` gives a paired source-group bootstrap CI on Test400 / Test400_indep / Test100.
  - **New training switches** (`tools/train_vitpose.py`):
    - `LOSS.TYPE rspose` (+ `LOSS.RS_INST`).
    - Mean teacher: `TRAIN.MT_ITEMS <boxes.json> TRAIN.MT_WITH_LABELED True TRAIN.GRAD_CKPT True AUTO_RESUME True`.
      This needs 2 chained jobs per fold (~10 min/epoch on V100) and about 45 GB host RAM.
  - **Labels and boxes.**
    - `tools/label_audit.py` builds a LOCAL review page. It contains SyRIP images, so never publish it.
    - `tools/apply_label_corrections.py` applies the exported decisions.json.
    - `tools/estimate_boxes.py` gives zero-shot boxes for unannotated images.
  - **Sapiens2.**
    - `scripts/sapiens2_zeroshot.sbatch` runs `tools/test_sapiens2.py` (GT boxes, Goliath-308 -> COCO-17) for 0.4b/1b.
    - Outputs go to `output/audit12/sapiens2/`.
    - `scripts/ensemble_kpts.py --a x.json --b y.json --out ens.json` averages two prediction files.
    - Fine-tune: `sbatch -J fidip-<tag>-f<k> scripts/sapiens2_ft.sbatch <tag> <k> --epochs 10` (`tools/train_sapiens2.py`; 1b needs ~38 GB). Default asks for H200 (~4 min/epoch); when the H200 queue is long, override with `--gres=gpu:a100:1 --constraint=a100@80g --time=02:30:00` (~8 min/epoch).

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
| round 5 | aug-cj-s2/s3, aug-geo-s1/s2, aug-light-s1/s2, mob-ft, mob-fidip (+ evals 11388501 11389154 11389155) | 11386778 11387264 11386746 11386785 11386757 11386795 11386820 11386845 | all COMPLETED (aug-cj-s1 11386742 cancelled on T4) | `output/<tag>` |
| round 6 | mob-ft120-s1/s2, mob-ft200-s1/s2 (+ eval 11389519) | 11389498 11389506 11389505 11389508 | all COMPLETED | `output/<tag>` |
| round 7 (stage A + D1) | vitpose-setup, soups, vitpose-zs, eval500-A-raw/bn | 11394626 11394628 11394634 11394636 11394637 | all COMPLETED | `output/soups/`, `eval500/vitpose/` |
| round 8 (stage D2, BN bug) | vit-{l,h}-lr{5e-5,1e-5}-s1 (+ eval500-D2-l/h 11395213 11395216) | 11395200 11395203 11395205 11395206 | CANCELLED (user-approved): decoder BN in train mode collapsed AP | `output/<tag>` (discard) |
| round 8b (stage D2, BN fixed) | vit2-{l,h}-lr{5e-5,1e-5}-s1 (+ eval500-D2b-l/h 11395527 11395547) | 11395455 11395479 11395499 11395504 | COMPLETED (H runs TIMEOUT at ep 24, best ckpt scored) | `output/<tag>` |
| round 8c/8d (stage D2 seeds) | vit2-l-lr1e-5-s2/s3, vit2-h20-lr1e-5-s1/s2 (+ eval 11398156 11398679) | 11398128 11398145 11398580 11398625 | all COMPLETED. **Best: H20, Test400 92.8 ± 0.1** | `output/<tag>` |
| round 9 (stage A on D) | vit-soups (CPU), eval500-Dsoup-h/l | 11400326 11400360 11400382 | all COMPLETED. **D-soup-h20: Test400 93.1** | `output/soups/D-soup-*.pth` |
| round 10 (C2 + B on ViTPose++-H) | c2-ld{95,90}-lr1e-5-s1, c2-ld90-lr3e-5-s1 (+ eval500-C2 11400473); b-rf{17,50,75,100}-s1 (+ eval500-B 11401502) | 11400434 11400442 11400455 11400506 11400510 11400521 11400537 | all COMPLETED (B eval 11401502). Null: Test400 92.6-93.0 | `output/<tag>` |
| round 11 (B2 confirm + 384x288) | b-rf100-s2 (+ eval500-B2 11410741); d3-h384-lr1e-5-s1 chained p1/p2/p3 (+ eval500-D3 11410743) | 11410487 11410716 11410725 11410732 | all COMPLETED (p1, p2 TIMEOUT by design, resumed). Null: real-only 92.9; 384x288 92.7 | `output/<tag>` |
| round 12 (CV + E2/E3/E4) | cv-h20-f0..4, cv-rs-f0..4, cv-mt-f0..4 (+p2/p3), sapiens2-setup/zs, r12-prep | 11422061-11422084, 11422319-11424695, 11424797-11434194, 11422150 11424809 11422557 | all COMPLETED. CV: h20 94.39, rs 94.03, mt 93.61; S2-1b zs 94.76; **H20+S2-1b ensemble 95.94, Test400 93.8** | `output/cv-*-f<k>`, `output/audit12/` |
| round 13 (S2-1b fine-tune CV) | cv-s2ft-f0..4 (H200 11476656-79 cancelled unstarted) | 11483756 11483758 11483798 11483802 11483803 | all COMPLETED (A100 80GB). S2-1b-ft 96.10; H20+S2-ft 96.05 vs 95.94 -> no advance | `output/cv-s2ft-f<k>` |
| round 13b (S2-1b fine-tune, full train) | full-s2ft-s1, full-s2ft-s2 | 11504452 11504476 | COMPLETED (A100 80GB). Test400 94.2 / 94.3; non-inferior to D-soup + S2-zs (94.0) -> recommended system | `output/full-s2ft-s<seed>` (weights ~3 GB each) |

## Guardrails

- Before launching a new fine-tuning wrapper, compare train-mode vs eval-mode loss on one batch.
  In round 8 the ViTPose++ decoder BatchNorm collapsed AP; the wrapper now freezes BN.
- After an sbatch "Unexpected message", wait a few minutes (not one) before concluding it did not queue.

- Never print, log or commit `.env` contents or any token. Read secrets from `.env` only.
- Submit with `source ~/fidip_submit_lib.sh; submit <tag> [overrides]` (from the repo dir): it
  skips tags already in `sacct` and refuses to submit if `sacct` fails/returns nothing. Never
  re-run a whole sweep script to add one job. `TIME=01:30:00 submit ...` overrides the 1 h limit.
- **Avoid c2204-c2207** (V100-PCIe, ~3x slower than the SXM2 nodes for this model) **and d1025**
  (Tesla T4, ~2.3 min/epoch for HRNet-W48 -> a 40-epoch run needs ~92 min); the submit helper
  excludes them by default. To exclude on an already-pending job:
  `scontrol update JobId=<id> ExcNodeList="c[2204-2207],d1025"`.
- Max **8 jobs queued+running per user** (QOSMaxSubmitJobPerUserLimit) -- batch sweeps accordingly.
- `sbatch` can fail transiently with "Unexpected message received" -- run `squeue -u $USER` before
  retrying so a job isn't submitted twice.
- Confirm with the user before `scancel`, deleting checkpoints, or overwriting results.
- `git pull --ff-only` on the cluster.
- No compute on the login node -- use `sbatch`/`srun`.
- Record every run in `AGENT.md` and every artifact in `MANIFEST.md`.
