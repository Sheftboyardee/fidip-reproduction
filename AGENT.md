# AGENT.md -- Research Log

Running log of what was asked of the agent, what it did, and why direction changed.
Newest entries at the bottom.

## 2026-10-08 -- Project setup

**Asked:** Read `STUDENT_GUIDE.md` and set up the project locally; create a fresh `fidip` env
instead of reusing `torch_env` (CUDA 11.8).

**Done:**
- Cloned upstream `ostadabbas/Infant-Pose-Estimation` (guide's clone URL is a placeholder).
- Upstream `fidip_env.yml` is a Linux export (libgcc, ld_impl_linux-64, ...) and can't be solved
  on Windows. Created conda env `fidip` (Python 3.12) and pip-installed the same pinned versions:
  torch 2.3.1 / torchvision 0.18.1 (cu121 wheels; driver 592 supports CUDA 12.x), numpy 1.26.4,
  opencv 4.10, pycocotools 2.0.8, yacs, json-tricks, easydict, etc., plus wandb, streamlit,
  python-dotenv.
- `lib/nms` extensions need MSVC + nvcc (not installed). Made the `cpu_nms`/`gpu_nms` imports in
  `lib/nms/nms.py` optional -- the SyRIP pipeline only calls pure-numpy `oks_nms`/`soft_oks_nms`.
- Fixed `lib/dataset/syrip.py` parsing image ids with `split('/')`, which breaks on
  Windows backslash paths; now `os.path.basename`. (Correction, found when committing: the same
  sed was run on `coco.py` but matched nothing -- `coco.py` has no such parsing, so only
  `syrip.py` changed.)
- Added `gpu_test.py` (referenced by the guide but missing upstream), `.env.example`, `.gitignore`
  entries (`.env`, data, models, outputs), and `SKILL.md` / `AGENT.md` / `MANIFEST.md`.

**Not in upstream repo (guide refers to them):** `fidip_simple.yml`, `fidip_cuda12.2.yml`,
`tools/demo.py`, `tools/streamlit_demo.py`, `demo_test_data/`, `requirements.txt`.

**Blocked on user:** SyRIP dataset password (email authors), pretrained model downloads
(Google Drive), cluster SSH setup.

**Install note:** pinned versions without cp312 Windows wheels (e.g. matplotlib 3.9.1) tried to
compile from source and failed (no C compiler). Reinstalled with `--only-binary :all:` and
slightly relaxed pins (matplotlib 3.9.4, etc.). Exact versions frozen to `requirements-win.txt`.

**Verified:** `gpu_test.py` -> CUDA 12.1, cuDNN 8.9, RTX 4060 Laptop 8 GB OK. All imports OK.
`adaptive_pose_hrnet` builds from the infant config and runs a forward pass on GPU
(outputs [1,48,96,72] features + [1,17,96,72] heatmaps).

## 2026-10-08 -- Dataset extraction check

**Context:** Project moved out of OneDrive to `C:\Users\nshef\personal_projects\fidip\`. User
extracted both zips into `data/`; Windows reported 1600 conflicting files and the user chose
"skip", and no separate `syrip` folder appeared.

**Findings:**
- `SyRIP.zip` (top folder `SyRIP/`) and `syrip_for_train.zip` (top folder `syrip/`) merged into
  one folder because Windows paths are case-insensitive. Renamed it to lowercase `data/syrip/`
  to match `DATASET.ROOT` (matters on the Linux cluster).
- Compared CRCs of all 1600 overlapping zip members: 1599 identical. The single difference is
  `validate_infant/test11.jpg`: SyRIP.zip copy is 1536x2048, syrip_for_train copy is 2048x2048.
  All annotation files (top-level, validate100, validate500) list test11 as 1536x2048, so the
  kept (SyRIP.zip) copy is the correct one. Skipping duplicates lost nothing.
- Every image referenced by the three config annotation files exists on disk.
- Top-level validate json == Test100 (same image ids, identical keypoints).
- Repo `dataset.syrip` loads all splits: validate_infant 100, train_infant 1200,
  train_pre_infant 3904 (of 4057 images).

## 2026-10-08 -- Baseline reproduction (released FiDIP HRNet-W48)

**Asked:** Run the baseline test with the downloaded pretrained models.

**Ran (laptop, RTX 4060):**
`python tools/test_adaptive_model.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True`

**Result on SyRIP Test100 (GT boxes, flip test):** AP 93.6 / AP50 98.5 / AP75 98.5 /
AR 94.6 / AR50 99.0 / AR75 99.0 -- **exact match** to the paper's DarkPose + FiDIP row.
Took about a minute. Stdout saved to
`log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-10-08-17-55/baseline_test_stdout.log`,
predictions in `output/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/results/`.

**Notes:** confirms the Windows env, nms patch and path fix give correct eval numbers. Test100
has only 100 images, so one AP point is about one image's worth of keypoint error; differences
under ~1 AP in later experiments need care (consider also reporting Test500).
The COCO checkpoint filename uses a Unicode `×` (`w48_384×288.pth`), which matches the config.

## 2026-10-08 -- Baseline reproduction (released FiDIP Pose-MobileNet)

**Ran:** `python tools/test_adaptive_model.py --cfg experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml TEST.MODEL_FILE models/mobile_fidip.pth TEST.USE_GT_BBOX True`

**Result on Test100:** AP 79.2 / AP50 99.0 / AP75 89.4 / AR 84.1 / AR50 99.0 / AR75 92.0 --
exact match to the paper's Pose-MobileNet + FiDIP row. Both released models now reproduce.
MobileNet (3.9M params, 224x224) is a practical model to train locally on the 8 GB laptop GPU.

**Cluster:** Explorer username `sheft.n` recorded in SKILL.md; passwordless SSH setup pending.

## 2026-10-08 -- Explorer cluster setup

**Asked:** Set up the cluster side (user `sheft.n`; user installed the SSH key themselves).

**Done:**
- Generated `~/.ssh/id_ed25519` + `Host explorer` alias on the laptop; passwordless SSH verified.
- Copied code, `data/syrip`, `models/` to `~/Infant-Pose-Estimation` on Explorer (tar over ssh,
  ~43 min at ~0.5 MB/s upload). Image counts and model md5 checksums match the laptop.
  Kept the password-protected dataset in private home, not the class-shared `/projects/ML_Small_Data`.
- Python env: conda `defaults` needs ToS acceptance (not accepted on the user's behalf) and
  conda-forge metadata fetch / `uv` Python download both stalled on the login node (outbound
  traffic goes via a proxy). Built a `uv` venv `~/venvs/fidip` inside a `short`-partition job
  instead (job 10929044): Python 3.12.15, torch 2.3.1+cu121, numpy 1.26.4 -- same as laptop.
- Wrote `scripts/eval.sbatch` and `scripts/train.sbatch`; SLURM logs go to
  `/scratch/sheft.n/fidip/logs/`.

**Found while reading the training code (affects reproduction):**
- The shipped infant configs leave `MODEL.PRETRAINED` and `TRAIN.CHECKPOINT` empty, so
  `train_adaptive_model_hrnet.py` would start from random weights while still freezing the first
  109 parameter tensors. Verified that `models/coco/w48_384×288.pth` loads into the FiDIP pose
  net with all keys matching; `train.sbatch` passes it as `TRAIN.CHECKPOINT` by default.
- `AUTO_RESUME` restores only the pose net + its optimizer, not the domain classifier or its
  optimizer, and stores the current (not best) AP as `perf`. Splitting training across chained
  jobs would silently change the adversarial training -- keep each run in one job for now.
- `TRAIN.PRE_EPOCH: 0` in the config, i.e. the domain-classifier pre-training stage is skipped.
  Worth checking against the paper before the reproduction-from-scratch run.

**Cluster check:** `scripts/eval.sbatch` with `models/hrnet_fidip.pth` (job 10929307, V100-SXM2-32GB,
node d1010) -> AP 93.6 / AR 94.6, identical to the laptop and paper. Cluster pipeline verified.
Job took 9 min wall time vs ~1 min on the laptop, probably first-run startup plus `DEBUG` image
dumps to NFS home; check this before starting long training runs.

## 2026-10-23 -- Training reproduction on Explorer

**Asked:** Run the training reproduction of the paper on an Explorer GPU.

**Finding before launch:** the shipped `w48_384x288_adam_lr1e-3_infant.yaml` sets
`TRAIN.LAMBDA: 0.000` (adversarial domain-confusion term in `lib/core/function.py:140` is
switched off) and `TRAIN.PRE_EPOCH: 0` (domain-classifier pre-training on `train_pre_infant`
skipped). With these values the pose net never sees the domain classifier, so this is plain
fine-tuning (paper: DarkPose + Finetune = 92.7 AP), not FiDIP (93.6). `lib/config/default.py`
has `LAMBDA 0.0005`, `PRE_EPOCH 10`, which look like the intended FiDIP settings.

**Hypothesis:** A (code-default FiDIP settings) reaches ~93.6; B (shipped yaml) reaches ~92.7.

**Launched** (both init from COCO DarkPose `models/coco/w48_384×288.pth`, 20 epochs, V100):
| Run | Job | Overrides | Output |
|---|---|---|---|
| A: FiDIP | 11329332 | `TRAIN.LAMBDA 0.0005 TRAIN.PRE_EPOCH 10` | `output/repro_fidip/` |
| B: shipped cfg (= finetune) | 11329348 | none | `output/repro_shipped/` |

**Code change:** the three `tools/train_adaptive_model*.py` saved loss/accuracy curves to
`./*.npy` (cwd), so concurrent runs would overwrite each other; now saved into each run's
`final_output_dir`.
**Note:** no random seed is set anywhere, so expect run-to-run noise; a single run within
~0.5 AP of the paper should count as reproduced.
**Ops:** `sbatch` intermittently failed with "Unexpected message received" (controller load);
check `squeue` before retrying so jobs aren't submitted twice.
**Duplicate-job incident:** two `sbatch` calls that returned "Unexpected message received" were
still queued by the controller a minute later, which left duplicate pending jobs 11329352 (B)
and 11329381 (A) writing to the same `OUTPUT_DIR` as the real runs. Both were cancelled while
still PENDING (never ran). Fix: `scripts/train.sbatch` now takes an exclusive `flock` on
`<OUTPUT_DIR>/.train.lock` and exits if another job holds it (tested on the home filesystem).
Rule: after any sbatch error, wait and check `sacct`, never blind-retry.

### Run A result (job 11329332, FiDIP settings) -- COMPLETED in 24 min on V100-SXM2-32GB

Per-epoch Test100 AP (validation inside training, same settings as eval: GT boxes, flip test):

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP | 89.1 | 90.0 | 90.6 | 90.7 | 91.2 | 90.7 | 90.7 | 89.6 | 91.1 | 91.1 | 91.1 | **92.0** | 91.6 | 90.6 | 89.6 | 90.0 | 91.4 | 91.6 | 91.8 | 90.4 |

- Best: **92.0 AP** (AP50 98.4, AP75 98.4, AR 93.6) at epoch 12 -> `model_best.pth`.
  Final epoch: 90.4. Paper FiDIP: 93.6 -> **not reproduced (-1.6 AP)**; still above the
  pretrained DarkPose zero-shot number (88.5) and 0.7 below the paper's Finetune row (92.7).
- AP oscillates by +-1 between epochs with no clear trend after epoch 5: LR stays at 1e-4 the
  whole run (`LR_STEP: [40, 200]` never triggers within 20 epochs).
- Caveat: `model_best.pth` is picked on Test100 itself (that's how the upstream code works, and
  presumably how the paper's number was obtained), so "best epoch" is optimistically biased.
- Domain classifier: ~90% real/synthetic accuracy after pre-training, ~92% during adversarial
  training -- with LAMBDA 5e-4 the features never become domain-invariant.
- Ops note: the monitor only surfaced the latest table line per poll, so the epoch numbers I
  reported live were off; read the whole log for per-epoch values.

**Open questions for the gap:** paper's actual LAMBDA / PRE_EPOCH / frozen-layer count / epochs
are unknown (config ships with the adversarial term off); no seed, so a second seed is needed to
know whether 1.6 AP is noise. Wait for run B before changing anything.

### Run B result (job 11329348, shipped cfg: LAMBDA 0, PRE_EPOCH 0) -- COMPLETED in 22 min

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP | 90.0 | 91.1 | 91.2 | 91.7 | 91.4 | 91.9 | 92.5 | 91.9 | 92.2 | 92.7 | 91.8 | 91.6 | 92.0 | 92.8 | 92.9 | 92.2 | 92.0 | 92.4 | 92.3 | **93.0** |

- Best **93.0** at the final epoch (paper Finetune row: 92.7 -> reproduced, +0.3).
- **Plain fine-tuning beat the FiDIP-settings run A (92.0)** and was steadier, still rising at
  epoch 20. Opposite of the paper's ordering (FiDIP 93.6 > Finetune 92.7). With n=1 per arm and
  ~+-1 AP epoch noise this is not conclusive -> repeats are needed (rep2/rep3 below).
- Possible reasons if it holds: the 10 pre-training epochs run `model_p.train()` on
  train_pre_infant (COCO+SURREAL adults), so BatchNorm running stats drift toward the adult
  domain before infant training even starts; and/or LAMBDA 5e-4 adds noise without making
  features domain-invariant (domain-classifier accuracy stayed ~92%).

## 2026-10-23 -- Follow-up sweep (1 h time limit per job, per user tip on queue speed)

Submitted via `~/submit_sweep.sh` on Explorer (one at a time, skips tags already submitted).
All use FiDIP settings (LAMBDA 5e-4, PRE_EPOCH 10, COCO init, 20 epochs) unless noted.
3 of 5 started immediately with `--time=01:00:00` (vs B waiting ~1 h with 4 h).

| Tag | Job | Change | Question |
|---|---|---|---|
| rep2 | 11330470 | none | run-to-run noise of A |
| rep3 | 11330481 | none | run-to-run noise of A |
| lrstep | 11330486 | `TRAIN.LR_STEP [10,15]` (verified parsed) | does LR decay stabilise/improve? |
| lam5e-3 | 11330496 | `TRAIN.LAMBDA 0.005` | stronger adversarial term |
| lam5e-2 | 11330524 | `TRAIN.LAMBDA 0.05` | much stronger adversarial term |

### Sweep results (2026-10-23)

| Run | LAMBDA | PRE_EPOCH | LR_STEP | Best AP (ep) | Final AP | Late-epoch domain acc (per batch) |
|---|---|---|---|---|---|---|
| B (shipped cfg) | 0 | 0 | none | **93.0** (20) | 93.0 | ~1.00 |
| A | 5e-4 | 10 | none | 92.0 (12) | 90.4 | ~0.9 |
| rep2 | 5e-4 | 10 | none | 91.3 (9) | 90.9 | ~0.9 |
| rep3 | 5e-4 | 10 | none | 91.9 (8) | 91.3 | -- |
| lrstep | 5e-4 | 10 | [10,15] | 91.2 (11) | 91.2 | -- |
| lam5e-3 | 5e-3 | 10 | none | 85.5 (1) | 73.9 | ~0.6 |
| lam5e-2 | 5e-2 | 10 | none | pending | | |

**Findings**
- FiDIP settings over 3 seeds: 91.7 +- 0.4 AP (best-of-epoch). Paper FiDIP: 93.6. Our plain
  fine-tune (B, n=1): 93.0 (paper 92.7). **Fine-tuning reproduces; FiDIP does not, and is worse
  than fine-tuning in our hands** (gap ~1.3 AP, about 3x the seed spread -- B still needs repeats).
- LR decay (lrstep) removes the epoch-to-epoch oscillation (best == final) but doesn't raise
  the ceiling -- the gap is not a schedule artefact.
- **LAMBDA scale problem:** the pose loss is a heatmap MSE of ~7e-4, and the adversarial term is
  `-LAMBDA * BCE` with BCE ~0.3-0.7. At LAMBDA 5e-3 the adversarial term is ~3x the pose loss,
  `Loss_p` goes negative, and pose accuracy collapses (AP 85.5 -> 73.9). Note the `-BCE` form is
  unbounded below (the pose net is rewarded for making the classifier confidently *wrong*), unlike
  a gradient-reversal / confusion (0.5-target) loss, so larger LAMBDA diverges instead of
  saturating.
- Monotone trend: more domain confusion (acc 1.0 -> 0.9 -> 0.6) == lower pose AP. In this
  implementation the adversary trades pose accuracy for invariance rather than helping.
- **Log-reading correction:** the parenthesised `Accuracy_d (...)` in training logs is a running
  average that includes the 10 pre-training epochs (counter `acc_num_total` is never reset), so
  it barely moves. Use the per-batch value. (Run A's ~0.9 claim still holds per batch.)
- Confound in A vs B: B differs in two things (no PRE_EPOCH, LAMBDA 0). Next isolate them.

**Next candidates**
1. Repeat B (2 seeds) -- is 93.0 a lucky draw?
2. `PRE_EPOCH 10, LAMBDA 0` and `PRE_EPOCH 0, LAMBDA 5e-4` -- which half of FiDIP hurts?
3. Check pre-stage BatchNorm drift hypothesis: pre-stage runs `model_p.train()` on adult data
   (BN running stats move) even though only `model_d` is optimised. Try `model_p.eval()` there.
4. Bounded adversarial loss (confusion loss toward 0.5, or gradient reversal) with LAMBDA sweep.

## 2026-10-23 -- Round 2: isolate the FiDIP ingredients

**Asked:** proceed with the proposed next round.

**Code changes** (both default off, so earlier runs/configs behave as before):
- `TRAIN.PRE_BN_FREEZE` (default False): pose net in `eval()` mode during domain-classifier
  pre-training, so BatchNorm running stats aren't updated on adult `train_pre_infant` images.
  In all three `tools/train_adaptive_model*.py`.
- `TRAIN.ADV_LOSS` (default `neg_bce` = original `loss_p - LAMBDA*BCE(d, domain)`):
  `confusion` = `loss_p + LAMBDA*BCE(d, 0.5)`, minimised (0.693) when the classifier is at chance,
  so it can't run away like `-BCE`. In `lib/core/function.py`.

**Submitted** via `~/submit_round2.sh` (1 h limit each, FiDIP init from COCO, 20 epochs):

| Tag | Job | Overrides | Question |
|---|---|---|---|
| B-rep2 / B-rep3 | 11331011 / 11331062 | none (shipped cfg) | is B's 93.0 a lucky draw? |
| pre-only | 11331060 | PRE_EPOCH 10, LAMBDA 0 | does classifier pre-training alone hurt? |
| adv-only | 11331069 | PRE_EPOCH 0, LAMBDA 5e-4 | does the adversarial term alone hurt? |
| bnfreeze | 11331067 | A + PRE_BN_FREEZE True | is pre-stage BN drift the cause? |
| conf5e-4 / conf5e-3 | 11331081 / 11331086 | A with ADV_LOSS confusion, LAMBDA 5e-4 / 5e-3 | does a bounded adversary help? |
| conf5e-2 | not yet | ... LAMBDA 5e-2 | (blocked by per-user job limit) |

**Ops:** 4 of 8 `sbatch` calls printed "Unexpected message received" but were queued anyway
(checked via squeue, no duplicates). Explorer enforces **QOSMaxSubmitJobPerUserLimit = 8 jobs
queued+running per user**; conf5e-2 was rejected by it and will be submitted when a slot frees.

**lam5e-2 (job 11330524, neg_bce, LAMBDA 0.05):** best 86.4 at epoch 1, final **0.8 AP** --
total collapse; confirms the unbounded `-BCE` adversary diverges as LAMBDA grows
(5e-4: ~91.7, 5e-3: 85.5 -> 73.9, 5e-2: 86.4 -> 0.8).

**Ops mistake (mine):** to submit only conf5e-2 I re-ran the whole `~/submit_round2.sh`. Its
"already submitted?" check read an empty/slow `sacct` result as "never submitted", so it
resubmitted every tag; several of those "Unexpected message" calls were queued by the controller
minutes later (duplicates 11331186 B-rep3, 11331276 conf5e-4), cancelled while PENDING. No run
was affected (the OUTPUT_DIR lock would have stopped a duplicate anyway).
Fixes: `~/fidip_submit_lib.sh` `submit()` refuses to submit when `sacct` fails or returns
nothing; submit single runs with it, never re-run a whole sweep script. A watcher cancels any
PENDING job whose name duplicates an earlier active one.

### Round 2 results (2026-10-23)

| Run | PRE_EPOCH | LAMBDA | ADV_LOSS | PRE_BN_FREEZE | Best AP (ep) | Final AP |
|---|---|---|---|---|---|---|
| B / B-rep2 / B-rep3 (fine-tune) | 0 | 0 | -- | -- | 93.0 (20) / 92.6 (19) / 92.7 (10) | 93.0 / 91.5 / 92.5 |
| **adv-only** | 0 | 5e-4 | neg_bce | -- | **93.0** (18) | 92.7 |
| pre-only | 10 | 0 | -- | False | 92.1 (20) | 92.1 |
| A / rep2 / rep3 (FiDIP settings) | 10 | 5e-4 | neg_bce | False | 92.0 / 91.3 / 91.9 | 90.4 / 90.9 / 91.3 |
| **bnfreeze** | 10 | 5e-4 | neg_bce | True | 92.6 (19) | 91.6 |
| conf5e-4 | 10 | 5e-4 | confusion | False | 91.2 (3) | 86.4 |
| conf5e-3 | 10 | 5e-3 | confusion | False | 86.3 (1) | 26.9 |
| conf5e-2 | 10 | 5e-2 | confusion | False | 85.9 (1) | 0.0 |

**Conclusions**
- Fine-tune is solid: 92.8 +- 0.2 over 3 seeds (paper 92.7). FiDIP settings: 91.7 +- 0.4.
- **The domain-classifier pre-training stage causes the FiDIP deficit**, not the adversarial term:
  adv-only (93.0) == fine-tune; pre-only (92.1) is ~0.7 below fine-tune; both together ~91.7.
- **Pre-stage BatchNorm drift is the likely mechanism:** the same FiDIP run with the pose net in
  `eval()` during pre-training (bnfreeze) gets 92.6, recovering most of the gap (n=1).
- The neg_bce adversary at LAMBDA 5e-4 is neutral (domain acc ~0.9 vs 1.0 without it), so in
  this setup the "invariant representation" part never helps.
- **Bounded confusion loss does not fix large LAMBDA -- it collapses too**, by a different route:
  at LAMBDA 5e-2 the term (>= 0.05 * 0.693 = 0.035) is ~50x the heatmap MSE (~7e-4); the pose net
  satisfies it by making features domain-uninformative (per-batch domain acc -> 0.50 exactly)
  and training pose accuracy falls 0.52 -> 0.02. Bounded below != safe; the weight has to be
  scaled to the ~1e-3 pose-loss magnitude. Even LAMBDA 5e-4 confusion erodes test AP
  (91.2 -> 86.4) while training pose acc stays ~0.58, i.e. the damage is on real test images.
- Best result so far: **93.0 AP** (B and adv-only), still 0.6 below the paper's FiDIP 93.6 --
  within best-of-20-epochs-on-Test100 noise. No setting has beaten fine-tuning.

**Log-reading slip:** my first awk pass mis-indexed the tab-separated training-log columns
(printed data time as loss_p); fields are 2 acc_d, 3 Loss_d_2, 4 Time, 5 Data, 6 Loss_p,
7 Loss_d, 8 Accuracy_p.

## 2026-10-23 -- Round 3: repeats of the two promising setups + Test500 re-scoring

**Asked:** (1) repeat adv-only and bnfreeze twice each; (2) re-score best checkpoints on Test500.
(Longer 40-epoch training noted as the natural next step.)

**Test500 annotation problem (important):** `annotations/validate500/person_keypoints_validate_infant.json`
gives **every** annotation the same placeholder `area` = 2,443,298 px^2, whereas the Test100 file
uses `area = bbox w*h` exactly (checked: ratio 1.000000 for all 100). COCO keypoint OKS scales its
tolerance by area, so scoring against the shipped Test500 file is far too lenient: the released
model's own Test100 predictions scored 98.1 AP against it vs 93.6 against the Test100 file.
The 100 shared images also have slightly different bboxes (and keypoints for 2 images) in the two files.
**Fix:** `data/syrip500/annotations/person_keypoints_validate_infant.json` = Test500 with the
100 Test100 annotations copied verbatim from the Test100 file and `area = bbox w*h` for the other
400; `data/syrip500/images` symlinks to `../syrip/images`. Check: the released model's Test100
predictions score exactly 93.6 on the Test100 subset of the corrected file.
**Do not report numbers against the shipped validate500 json.**

**Eval design:** `scripts/eval500.sbatch <list>` runs Test500 inference once per checkpoint
(GT boxes, flip test, `DEBUG.DEBUG False`), then `scripts/subset_ap.py` scores the same
predictions on Test500, **Test400** (= Test500 minus Test100) and Test100. `model_best.pth`
was chosen on Test100, so **Test400 is the unbiased held-out number**; `final_state.pth`
(epoch 20, no selection) is scored too. Results -> `eval500/results.csv` on Explorer.
23 checkpoints (released + best/final of 11 non-collapsed runs) in `eval500/list_0{0,1,2}.txt`.

**Submitted** (1 h limit each):
| Tag | Job | Overrides |
|---|---|---|
| adv-only-rep2 / rep3 | 11332174 / 11332165 | PRE_EPOCH 0, LAMBDA 5e-4 |
| bnfreeze-rep2 / rep3 | 11332183 / 11332187 | PRE_EPOCH 10, LAMBDA 5e-4, PRE_BN_FREEZE True |
| eval500-00 | 11332206 | list_00 (released, A, B, rep2, rep3 best/final) |
| eval500-01 / 02 | 11332236 / 11332224 | lists 01, 02 (sbatch said "Unexpected message"; both appeared ~2 min later, once each) |

### Round 3 results (2026-10-23)

Repeats (Test100 best AP, from training logs): adv-only-rep2 92.2, adv-only-rep3 92.2;
bnfreeze-rep2 91.6, bnfreeze-rep3 91.6.

**Corrected Test500 / Test400 / Test100 scores** (mean ± sd over seeds; `results/eval500_summary.md`,
per-checkpoint numbers in `results/eval500.csv`, raw in `results/eval500_raw.csv`):

| Setup | n | **Test400** (best ckpt) | Test400 (final ckpt) | Test100 (best) | Test500 (best) |
|---|---|---|---|---|---|
| Released FiDIP (`hrnet_fidip.pth`) | 1 | 90.7 | -- | 93.6 | 91.2 |
| Fine-tune (shipped cfg) | 3 | 89.7 ± 0.3 | 89.7 ± 0.4 | 92.7 ± 0.2 | 90.2 ± 0.3 |
| Adversarial only (LAMBDA 5e-4, no pre-training) | 3 | 90.0 ± 0.3 | 90.0 ± 0.3 | 92.5 ± 0.4 | 90.3 ± 0.3 |
| Pre-training only (PRE 10, LAMBDA 0) | 1 | 89.5 | 89.5 | 92.0 | 90.0 |
| FiDIP + PRE_BN_FREEZE | 3 | 89.3 ± 0.4 | 88.2 ± 1.1 | 91.9 ± 0.6 | 89.7 ± 0.5 |
| FiDIP settings (LAMBDA 5e-4, PRE 10) | 3 | 88.9 ± 0.3 | 88.8 ± 0.5 | 91.7 ± 0.4 | 89.4 ± 0.3 |
| FiDIP + LR_STEP [10,15] | 1 | 88.5 | 88.5 | 91.2 | 89.1 |
| FiDIP + confusion loss 5e-4 | 1 | 86.7 | 83.9 | 91.1 | 87.4 |

**Conclusions**
- On the unbiased Test400, **adversarial-only (90.0) ≈ fine-tune (89.7)**: the FiDIP adversarial
  term at LAMBDA 5e-4 gives at most +0.3 AP, inside one seed-sd. Not a demonstrated gain.
- **The domain-classifier pre-training stage costs ~1 AP** (FiDIP settings 88.9 vs adversarial-only
  90.0). PRE_BN_FREEZE recovers only ~0.4 of that and is noisier at the final epoch (88.2 ± 1.1),
  so **BN drift is at most a partial explanation -- hypothesis mostly rejected.** With BN frozen
  the pose net is untouched by pre-training, so the remaining cost must come from starting the
  adversarial game against an already-trained (strong) discriminator.
- **The released checkpoint (90.7 Test400) beats every setting we trained, by ~0.7 over our best
  mean (> 2 seed-sd).** The paper's recipe differs from what this repo ships in some way we
  can't see (epochs, LR, LAMBDA, init, or training data -- the README mentions a newer FiDIP
  model trained on the *expanded* SyRIP; unclear which one `hrnet_fidip.pth` is).
- Test100 is ~3 AP higher than Test400 for **every** model including the released one, so the
  gap is mostly Test100 being easier for AP, not best-epoch selection bias. Best vs final
  checkpoint differ by <= 0.3 on Test400 for fine-tune / adversarial-only.
- Our reproduction verdict: **fine-tune reproduces (paper 92.7, ours 92.7 ± 0.2 Test100);
  FiDIP's reported gain over fine-tuning does not reproduce with the shipped code** (best
  FiDIP variant ≈ fine-tune).

**Ops incident:** concurrent eval jobs `tee -a`-ing into one shared `eval500/results.csv` lost
rows (B-final missing, a duplicate shown). Rebuilt all 93 rows from each job's own stdout log
(`eval500/results_rebuilt_from_logs.csv`); `scripts/eval500.sbatch` now writes
`eval500/results_<jobid>.csv` per job. Old file kept as `eval500/old_shared_results.csv.bak`.

## 2026-10-23 -- Round 4: 40-epoch training (fine-tune and adversarial-only)

**Why:** fine-tune runs were still improving at epoch 20; LR never decays in the shipped
schedule. Test400 says these two are the best setups we have.
**Setup:** `TRAIN.END_EPOCH 40 TRAIN.LR_STEP [30]` (x0.1 at epoch 30), COCO init, 2 seeds each,
`--time=01:30:00` (`TIME=` override added to `~/fidip_submit_lib.sh`).

| Tag | Job | Extra overrides |
|---|---|---|
| ft40-s1 / ft40-s2 | 11333089 / 11333092 | (shipped cfg: PRE_EPOCH 0, LAMBDA 0) |
| adv40-s1 / adv40-s2 | 11333095 / 11333110 | PRE_EPOCH 0, LAMBDA 5e-4 |
| eval500-04 | 11333148 (queued late after "Unexpected message"; state PENDING (Dependency)) | `--dependency=afterany` on all four; scores best+final on Test500/400/100 |

**Compare against:** Test400 fine-tune 89.7 ± 0.3, adversarial-only 90.0 ± 0.3, released 90.7.

**Slow-node issue:** adv40-s1 (11333095) landed on c2207, a **V100-PCIe** node running at
1.14 s/iter vs 0.40 s/iter on the V100-SXM2 nodes -> only 11 epochs in 40 min; it will hit the
1.5 h limit around epoch ~25 (before the LR step at 30), so it can't give a 40-epoch result.
Submitted replacement **adv40-s3 (11334290)** with `--exclude=c[2204-2207]` (applied via
`scontrol update` -- `SBATCH_EXCLUDE` env var is not a thing) plus dependent eval
**eval500-05 (11334329)**. `~/fidip_submit_lib.sh` now excludes c[2204-2207] by default
(`EXCLUDE=` to override). Left 11333095 running pending the user's OK to cancel (guardrail);
eval500-04 waits on it via afterany and will score whatever checkpoint it reached.
Progress at this point: ft40-s1 and ft40-s2 finished all 40 epochs; adv40-s2 at epoch 29.

### Round 4 results (2026-10-23) -- 40 epochs, LR x0.1 at epoch 30

Test100 per run (from training logs): ft40-s1 best 93.1 (ep 30) / final 92.4; ft40-s2 best 92.4
(ep 9) / final 91.7; adv40-s2 best 92.3 (ep 6) / final 91.6; adv40-s3 best 91.9 (ep 11) / final 91.8.
adv40-s1 TIMEOUT at epoch 26 (slow node; excluded).

**Updated summary across all rounds** (`results/eval500_summary.md`; now ranked by Test400 at the
*final* checkpoint, which involves no selection on any test data):

| Setup | n | **Test400** (final ckpt) | Test400 (best ckpt) | Test100 (best) | Test500 (best) |
|---|---|---|---|---|---|
| Released FiDIP (`hrnet_fidip.pth`) | 1 | -- | 90.7 | 93.6 | 91.2 |
| Fine-tune, 40 ep + LR step 30 | 2 | 90.2 ± 0.2 | 90.0 ± 0.6 | 92.8 ± 0.5 | 90.4 ± 0.6 |
| Adversarial only, 20 ep | 3 | 90.0 ± 0.3 | 90.0 ± 0.3 | 92.5 ± 0.4 | 90.3 ± 0.3 |
| Fine-tune (shipped cfg), 20 ep | 3 | 89.7 ± 0.4 | 89.7 ± 0.3 | 92.7 ± 0.2 | 90.2 ± 0.3 |
| Adversarial only, 40 ep + LR step 30 | 2 | 89.8 ± 0.2 | 89.1 ± 0.4 | 92.1 ± 0.2 | 89.6 ± 0.4 |
| Pre-training only, 20 ep | 1 | 89.5 | 89.5 | 92.0 | 90.0 |
| FiDIP + PRE_BN_FREEZE, 20 ep | 3 | 88.2 ± 1.1 | 89.3 ± 0.4 | 91.9 ± 0.6 | 89.7 ± 0.5 |
| FiDIP settings (LAMBDA 5e-4, PRE 10), 20 ep | 3 | 88.8 ± 0.5 | 88.9 ± 0.3 | 91.7 ± 0.4 | 89.4 ± 0.3 |
| FiDIP + LR_STEP [10,15], 20 ep | 1 | 88.5 | 88.5 | 91.2 | 89.1 |
| FiDIP + confusion loss 5e-4, 20 ep | 1 | 83.9 | 86.7 | 91.1 | 87.4 |

Released model has a single checkpoint (Test400 90.7). adv40-s1 (job 11333095) timed out at epoch 26 on a slow V100-PCIe node and is excluded (best ckpt Test400 89.2).

**Conclusions**
- **Best training recipe found: plain fine-tuning for 40 epochs with an LR drop at epoch 30 --
  Test400 90.2 ± 0.2 (final ckpt), 0.5 below the released FiDIP checkpoint (90.7)** and +0.5 over
  20-epoch fine-tuning. n=2, so treat +0.5 as suggestive.
- Longer training does **not** help the adversarial variant (40 ep 89.8 vs 20 ep 90.0).
- Across 4 rounds and ~30 runs, **no FiDIP component beats plain fine-tuning on the unbiased
  Test400**; the domain-classifier pre-training stage reliably costs ~1 AP.
- On Test100 (the paper's metric), longer training mostly reshuffles which epoch is best; best-
  epoch values cluster at 92-93 for every non-broken setup, so Test100 alone cannot separate them.
- Final verdict for the report: fine-tune baseline reproduced (92.7 vs paper 92.7 on Test100);
  FiDIP's +0.9 over fine-tune not reproduced with the shipped code/config; released checkpoint
  remains the best model we have (Test100 93.6 / Test400 90.7).

## 2026-10-23 -- GitHub repo

**Asked:** set up a GitHub repo and commit everything relevant.
- Committed on top of the upstream history (keeps the authors' commits/attribution); upstream
  remote renamed `origin` -> `upstream`.
- Committed: code changes, `scripts/` (incl. `scripts/cluster/` copies of the Explorer-home helpers
  `fidip_submit_lib.sh`, `fidip_env_setup.sh`, `submit_sweep.sh`, `submit_round2.sh`),
  `requirements-{win,explorer}.txt`, `results/`, the three research docs, README note.
- **Not committed (gitignored):** `data/` (password-protected SyRIP, incl. the derived corrected
  Test500 annotations), `models/`, `output/`, `log/`, `.env`. Scanned for tokens/secrets: none.
- `gh` is not installed and there is no `GITHUB_TOKEN`, so the user creates the empty repo on
  github.com; pushing uses their existing git credential manager login.
