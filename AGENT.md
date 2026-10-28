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
- Pushed to **https://github.com/Sheftboyardee/fidip-reproduction** (private; confirmed 404 from
  the unauthenticated API), branch `main`.
- Commit hygiene slip (caught before pushing): `git add --renormalize .` with a broad
  `*.py eol=lf` rule restaged 36 untouched upstream `syn_generation/` files as line-ending
  rewrites. Reset that local commit and scoped `.gitattributes` to `scripts/**` only.
- Explorer now syncs by git: read-only deploy key `~/.ssh/fidip_deploy` (user added it in the repo
  settings, write access off), ssh alias `github-fidip`. Before switching the cluster checkout I
  compared all 129 tracked files to `origin/main` ignoring CR: the only differences were files
  edited on the laptop after the last tar sync, so `git checkout -f -B main origin/main` lost
  nothing. Data/models/outputs untouched. Added `eval500/` to `.gitignore`.
- Noticed `output/` on Explorer is 29 GB (75 GB quota) -- noted in SKILL.md; nothing deleted.

## 2026-10-23 -- Storage check

**Asked:** user believed they had more than 75 GB (RC ticket RITM8406068 granted
`/projects/ML_Small_Data`, 1 TB). Verified with RC's `check-quota` from a short-partition job
(11338149):
- Home: 0.06 TiB used of 0.07 TiB hard limit (~77 GB) -- the "75 GB" figure was right for home,
  but I had only taken it from the course template, not measured it.
- `/projects/ML_Small_Data`: 1.0 TiB soft / 1.5 TiB hard, empty. Group members are only
  `sheft.n` and `ostadabbas`. **Correction:** I earlier called it "shared with the whole class"
  from its name alone -- wrong; it is the user's own project space.
- Home usage: `output/` 29 GB, `eval500/` 25 GB (mostly unused `feature.npy` dumps), `.cache` 5 GB.
- Proposed (pending user OK): move `output/` and `eval500/` to `/projects/ML_Small_Data` and
  symlink them back; drop the unused `feature.npy` files.
- **Done (user chose "move + drop features"):** deleted the 40 unused `eval500/*/feature.npy`
  dumps (26.5 GB; the 40 prediction JSONs kept). Copied `output/` (840 files, 30.6 GB) and
  `eval500/` (133 files) to `/projects/ML_Small_Data/sheft.n/fidip/` in a short-partition job
  (11339328) that verified file counts + byte totals; also md5-checked the final-result
  checkpoint. My first attempt bundled `rm -rf "$SRC/$d"` into the job script and was blocked by
  a safety check (variable rm target); redid it as copy-and-verify only, then removed the two
  home copies by literal path after reading the verification, and symlinked them back.
  Home 61 GB -> 7.4 GB. `flock` verified on the project filesystem through the symlink.
  `eval500.sbatch` now deletes `feature.npy` after scoring; `.gitignore` also matches the
  symlinks (`/output`, `/eval500`).

## 2026-10-26 -- Round 5: MobileNet backbone + data augmentation

**Asked:** try the MobileNet backbone, then data augmentation.

**MobileNet starting point problem:** the paper's MobileNet numbers start from a COCO-pretrained
Pose-MobileNet (46.5 AP zero-shot). The mobile config has `MODEL.PRETRAINED: ''` (random init),
the repo ships no such checkpoint, the public TGA_models Drive folder only has HRNet / ResNet /
Hourglass COCO models (checked in the browser), and Explorer has no shared COCO copy.
`mobile_fidip.pth` is already SyRIP-trained, so can't be a start. **User chose: ImageNet init.**
- PoseMobileNet's 312 `features.*` tensors match torchvision MobileNetV2 one-to-one in order and
  shape (not names) -> `scripts/make_mobilenet_init.py` maps them by position and verifies via
  `init_weights` (0 mismatches in a local test with random weights). Head stays random.
- Built on Explorer in job 11386711 (downloads `mobilenet_v2-b0353104.pth`, ~14 MB) ->
  `models/imagenet/mobilenet_v2_pose_init.pth`. sbatch said "Unexpected message"; queued late.
- Expect results well below the paper's 78.9 / 79.2: not comparable (no COCO pose pretraining).

**Augmentation (HRNet, best recipe = ft40: shipped cfg + END_EPOCH 40, LR_STEP [30]; baseline
Test400 final 90.2 ± 0.2).** Existing: flip, scale (sf 0.35), rotation (rf 45, 60% of samples),
half-body (p 0.3). Added `DATASET.COLOR_JITTER` (brightness/contrast/saturation ~U[1-j,1+j],
training crops only; default 0). Hypothesis: colour jitter helps most, since 1000 of the 1200
training images are synthetic renders with different colour statistics from real photos.

| Tag | Jobs (s1 / s2) | Change |
|---|---|---|
| aug-cj | 11386742 / 11386778 | COLOR_JITTER 0.4 |
| aug-geo | 11386746 / 11386785 | ROT_FACTOR 60, SCALE_FACTOR 0.5 |
| aug-light | 11386757 / 11386795 | ROT_FACTOR 15, SCALE_FACTOR 0.15, PROB_HALF_BODY 0 |

MobileNet runs (mob-ft, mob-fidip) to be submitted once the init job finishes (8-job cap).
- Init job 11386711 COMPLETED: real ImageNet weights (`mobilenet_v2-b0353104.pth`, 14.2 MB) ->
  `models/imagenet/mobilenet_v2_pose_init.pth`, 312 tensors, 0 mismatches after `init_weights`.
- MobileNet runs (mobile cfg, `MODEL.PRETRAINED models/imagenet/mobilenet_v2_pose_init.pth`,
  END_EPOCH 40, LR_STEP [30], LR 1e-3 as shipped): **mob-ft 11386820** (LAMBDA 0, PRE 0) and
  **mob-fidip 11386845** (LAMBDA 5e-4, PRE 10). Both sbatch calls said "Unexpected message";
  both appeared once in sacct. Reference: released `mobile_fidip.pth` Test100 79.2.
- **Slow-node issue again:** aug-cj-s1 (11386742) landed on d1025, a **Tesla T4**: 14 epochs in
  32 min (~2.3 min/epoch) -> 40 epochs needs ~92 min vs the 1.5 h limit; likely to time out
  around epoch 38 (after the LR step, but without `final_state.pth`). Added d1025 to the
  exclusion on all 7 pending jobs (`scontrol update`) and to the submit helper default.
- User chose cancel + resubmit: scancel 11386742 after 37 min (partial output left in
  `output/aug-cj-s1`, not used); replacement **aug-cj-s3 = 11387264** (new OUTPUT_DIR so it
  doesn't auto-resume the cancelled run), excluded from c[2204-2207],d1025.

### Round 5 results (2026-10-26)

Test400 = held-out 400 images (unbiased); full tables in `results/eval500_summary.md`.

| Run | Test100 best / final | **Test400 best / final** |
|---|---|---|
| aug-cj-s2 / aug-cj-s3 (colour jitter 0.4) | 92.5 / 92.1, 93.2 / 92.5 | 90.3 / 90.3, 89.8 / 90.3 |
| aug-geo-s1 / aug-geo-s2 (rot 60, scale 0.5) | 93.3 / 92.8, 93.9 / 93.7 | 89.4 / 90.0, 90.4 / 90.3 |
| aug-light-s1 / aug-light-s2 (rot 15, scale 0.15, no half-body) | 91.6 / 90.7, 91.1 / 90.5 | 88.4 / 88.3, 88.2 / 88.5 |
| mob-ft (MobileNetV2 ImageNet init, fine-tune) | 57.3 / 57.3 | 43.5 / 43.5 |
| mob-fidip (ImageNet init, FiDIP) | 17.2 / 9.0 | 12.1 / 8.7 |
| released `mobile_fidip.pth` (reference) | 79.2 | 68.3 |

**Augmentation conclusions (vs ft40 baseline Test400 final 90.2 ± 0.2)**
- **Less augmentation clearly hurts: aug-light 88.4 ± 0.1 (-1.8).** Consistent across seeds and
  on both test sets -> with 1200 training images the default geometric augmentation matters.
- **More geometric augmentation: no held-out gain (90.1 ± 0.3)**, despite the best Test100 numbers
  of the whole project (aug-geo-s2 final 93.7, above the paper's 93.6). The Test100 jump did not
  transfer to Test400 -- a direct example of Test100 being too small to rank methods, and the
  reason all conclusions here use Test400.
- **Colour jitter: 90.3 ± 0.0, +0.1 over baseline** -- within noise; hypothesis (colour jitter
  bridges synthetic->real) **not supported** at this strength. Still the top retrained setting,
  0.4 below the released checkpoint (90.7).

**MobileNet conclusions**
- ImageNet init (no COCO pose pretraining) costs ~25 AP on Test400 (43.5 vs released 68.3):
  most of Pose-MobileNet's infant performance comes from COCO pose pretraining. mob-ft was still
  improving at epoch 40 (best = final), so longer training would help somewhat.
- **FiDIP from ImageNet init collapses (8.7):** per-batch domain-classifier accuracy stays at
  chance (0.43-0.69) all run while pose accuracy lags (0.29 vs 0.67 for mob-ft at ep 32). With
  a randomly initialised head the adversary makes features domain-uninformative before they
  become pose-informative; the pose loss stays positive, so this is not the -BCE runaway seen
  with HRNet at large LAMBDA.
- MobileNet loses much more on the harder Test400 than HRNet (released: 79.2 -> 68.3 vs
  93.6 -> 90.7).

## 2026-10-26 -- Round 6: longer MobileNet fine-tuning

**Asked:** run MobileNet fine-tuning longer.
**Why:** mob-ft (40 ep, 9 min on a V100-SXM2) was still rising: ~49 AP around epoch 29, then
+5 from the LR drop at 30 to 57.3 (Test100) at the final epoch.
**Setup:** mobile cfg, ImageNet init (`models/imagenet/mobilenet_v2_pose_init.pth`), LR 1e-3,
fine-tune only (LAMBDA 0, PRE 0), 2 seeds per length, `--time=01:30:00`, slow nodes excluded.

| Tag | Jobs (s1 / s2) | Epochs | LR_STEP |
|---|---|---|---|
| mob-ft120 | 11389498 / 11389506 | 120 | [90, 110] |
| mob-ft200 | 11389505 / 11389508 | 200 | [150, 180] |
| eval500-09 | 11389519 | -- | Test400/Test100 of best+final, `--dependency=afterany` on all four |

**Compare against:** mob-ft (40 ep) Test400 43.5; released COCO-pretrained `mobile_fidip.pth` 68.3.

### Round 6 results (2026-10-26)

| Run | Test100 best (ep) / final | **Test400 final** |
|---|---|---|
| mob-ft (40 ep, earlier) | 57.3 (40) / 57.3 | 43.5 |
| mob-ft120-s1 / s2 | 58.0 (109) / 56.7, 57.8 (96) / 55.9 | 43.9, 43.8 |
| mob-ft200-s1 / s2 | 60.4 (69) / 58.1, 60.7 (194) / 59.0 | 44.6, 46.1 |
| released `mobile_fidip.pth` | 79.2 | 68.3 (single ckpt) |

**Conclusions**
- Longer training helps a little: **200 ep -> Test400 45.4 ± 1.0 (+1.9 over 40 ep)**; 120 ep
  gives ~nothing (43.8 ± 0.1). Most of the gain arrives before the first LR step (mob-ft200-s1
  was 59.2 Test100 at epoch 90 vs 50.8 at epoch 40) -- the model keeps learning slowly at LR 1e-3.
- **Training length is not what separates us from the released model: still ~23 AP short on
  Test400 (45.4 vs 68.3).** The missing ingredient is COCO pose pretraining of the whole network
  (head included), which 1200 infant images cannot replace. Further MobileNet gains need that
  checkpoint (ask the authors) or our own COCO pre-training.
- Runtime: 120 ep ~18-26 min, 200 ep ~45 min on V100-SXM2/H200 -- cheap to explore if needed.

## 2026-10-26 -- Literature review and improvement plan (MobileNet parked)

User asked for a literature search on recent directions that could beat the FiDIP baseline, plus
a multi-step plan of hypotheses. Full write-up with sources is in **PLAN.md**.

Main findings from the literature:
- Generic large models (ViTPose) beat infant-specialised ones on infants (Jahn et al. 2025, two
  papers; SMaL 2022).
- Mean-teacher self-training (SHIFT, CVPRW 2025) beats adversarial alignment, including FiDIP.
  This matches our own finding that FiDIP's adversarial term does not help.
- Weight averaging (WiSE-FT / model soups) is untested on pose estimation.

Stages, ordered by cost:
- A: eval-only weight soups and WiSE-FT of checkpoints we already have
- B: value of the synthetic data (real oversampling vs real-only)
- C: regularised fine-tuning (freeze, layer-wise LR decay, L2-SP)
- D: ViTPose zero-shot, then fine-tuned. Largest expected gain.
- E: self-training on unlabelled infant images. Gated on obtaining non-test unlabelled data.
- F: label-noise audit

Decision rule: 2-seed Test400 mean must beat the comparison by ≥ 0.5 AP (the seed noise is
±0.2-0.3). The bar is the released hrnet_fidip at 90.7 Test400.

## 2026-10-26 -- Round 7: PLAN.md stages A and D1 in parallel

User: "Start A and D in parallel."

**Stage A: weight soups / WiSE-FT. Evaluation only; no training.**
- `scripts/weight_average.py` averages state dicts.
  - Keys of COCO w48, `hrnet_fidip.pth` and our `final_state.pth` were checked identical (1754 tensors).
  - `--recal-bn` re-estimates BN stats on train_infant with a cumulative average.
  - Smoke-tested on the laptop.
- The spec list is `scripts/stageA_soups.txt`:
  - A1: seed soups. ft40 ×2, aug-cj ×2, those four together, and those plus aug-geo ×2.
  - A2: WiSE-FT. COCO ↔ ft40-s1/s2 at α 0.5/0.7/0.8/0.9.
  - A3: released `hrnet_fidip` ↔ ft40-s1/s2 at 50/50 and 70/30.
- Every soup is built twice: `-raw` (averaged BN stats) and `-bn` (recalibrated BN).
- COCO w48 zero-shot is also scored, as the D1 reference.
- Pre-registered rule: the reported soup is the uniform one (A1-four or A1-six), not whichever
  scores best on Test400.

**Stage D1: ViTPose zero-shot.**
- `tools/test_vitpose.py` runs HF transformers ViTPose with GT boxes, flip test and the repo's
  rescoring. It writes a COCO results json scored by `subset_ap.py`.
- Models are listed in `scripts/vitpose_models.txt`:
  - base-simple
  - base
  - plus-base, plus-large and plus-huge, using expert 0, the COCO head
- Laptop smoke test, ViTPose-B-simple zero-shot on Test100: **86.1 AP with flip, 85.4 without.**
  So the pipeline is sane, and the model is already strong with no infant training.
- transformers 4.53.3 installed in the laptop `fidip` env. Explorer gets the same pin through
  `scripts/vitpose_setup.sbatch`, with the HF cache at
  `/projects/ML_Small_Data/sheft.n/fidip/hf_cache`.

| Job | ID | Notes |
|---|---|---|
| vitpose-setup | 11394626 | short partition: install transformers, pre-download 5 checkpoints |
| vitpose-zs | 11394634 | afterok setup |
| soups | 11394628 | builds 32 soups into `output/soups/` |
| eval500-A-raw | 11394636 | afterok soups, `output/soups/eval_raw.txt` |
| eval500-A-bn | 11394637 | afterok soups, `output/soups/eval_bn.txt` (+ coco-w48) |

### Round 7 results (2026-10-26)

All 5 jobs completed with no failures.
- Full tables: `results/eval500_summary.md`.
- Raw rows: `results/eval500_raw.csv`.

**Stage A: weight soups. Seed members' Test400 mean is 90.2; released hrnet_fidip is 90.7.**

| Soup | Test400 raw / bn | Test100 raw / bn |
|---|---|---|
| A1 ft40 s1+s2 | 90.7 / 90.8 | 92.6 / 93.2 |
| A1 aug-cj s2+s3 | 90.5 / 91.1 | 92.3 / 92.9 |
| **A1 four (pre-registered)** | 90.7 / **91.1** | 92.5 / 92.9 |
| A1 six (+ aug-geo) | 91.0 / 91.0 | 93.0 / 93.3 |
| A2 WiSE-FT α 0.5 / 0.7 / 0.8 / 0.9 (mean of s1, s2; raw and bn pooled) | 89.4 / 90.5 / 90.7 / 90.7 | ~92.6 / 93.1 / 93.2 / 92.9 |
| A3 released ⊕ ft40 (4 soups × raw/bn) | 90.3-91.1 | 93.0-94.0 |

- **Seed soups work.**
  - The pre-registered uniform soup of the four seeds reaches **Test400 91.1**. That is +0.9
    over the mean of its members, which passes the ≥ 0.5 rule against our own recipe.
  - Every seed soup lands at 90.7-91.1, so this is a consistent effect, not one lucky draw.
  - Against the released model (90.7) the soup is +0.4. That is a real but not decisive
    improvement under our rule.
  - On Test100 it is 92.9 vs 93.6. The paper's own split still favours the released model.
- **BN recalibration** helps the seed soups by +0.1 to +0.6. It is mixed for WiSE-FT and A3.
- **WiSE-FT gives no gain.**
  - α 0.8-0.9 matches the fine-tuned endpoint (90.2-90.3). α 0.5 loses about 1 AP.
  - Pulling back toward COCO does not help. Forgetting adult-pose knowledge is not the
    bottleneck, so stage C3 (L2-SP) is deprioritised.
- **Souping with the released model** gives about 91.0, the same as souping our own seeds. Its
  best Test100 (94.0, A3-rel70-s1-raw) beats 93.6, but that is a pick among 8 soups, so we
  don't claim it.

**Stage D1: zero-shot, no infant training.**
- **COCO HRNet-W48: Test400 86.7.** Fine-tuning on SyRIP adds only about +3.5 on top of that.
- **ViTPose++-H: Test400 90.1, Test100 92.7.**
  - That is +3.4 over HRNet zero-shot, and it ties our fine-tuned HRNet with zero infant images.
  - Scaling is monotonic: ++-B 87.5 → ++-L 89.4 → ++-H 90.1.
- If fine-tuning adds even part of HRNet's +3.5, ViTPose++-L/H should clear 91-93.
- **D2 (fine-tuning ViTPose++ on SyRIP) is a clear go.**

**Storage:** soups 7.7 GB under `output/soups/`; HF cache 6.1 GB. Both are on /projects.

## 2026-10-26 -- Round 8: stage D2, fine-tuning ViTPose++ on SyRIP

D1 cleared the gate (ViTPose++-H zero-shot Test400 90.1), so I built the fine-tuning path.
- `lib/models/adaptive_pose_vitpose.py` wraps HF `VitPoseForPoseEstimation` in the
  `adaptive_pose_*` interface. ViTPose++ uses the COCO expert (`dataset_index` 0). This means
  `test_adaptive_model.py`, `validate_adaptive` and `eval500.sbatch` all work unchanged.
- `tools/train_vitpose.py`:
  - **Same as HRNet ft40:** data pipeline, augmentation, balanced real/synthetic sampler,
    JointsMSELoss, Test100 validation each epoch, and 40 epochs with LR ×0.1 at epoch 30.
  - **Dropped:** the domain classifier, and the HRNet script's hard-coded freezing of the first
    109 parameters.
    - Side finding: that freeze means all our HRNet runs had the stem and stage 1 frozen.
      PLAN.md C1 (freezing) is therefore partly the existing baseline already.
  - **Added:** AdamW with WD 0.1 and no decay on norms or biases, fp16/bf16 autocast, and
    optional `TRAIN.LAYER_DECAY` (off for now).
- Config `experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml`: 256×192 input,
  64×48 heatmap, sigma 2, batch 32.
- **Laptop smoke test** with ViTPose++-B, 1 epoch:
  - Zero-shot Test100 through the *repo* pipeline is 89.5. The HF pipeline gives 89.8, so the two
    protocols agree to 0.3.
  - After 1 epoch: 89.5.
  - `test_adaptive_model.py` reloads the checkpoint and reproduces 89.5.
- `.gitignore`: the `models/` rule was also ignoring `lib/models/` for new files. Anchored it to
  `/models/`.

| Run | Job | Model | LR |
|---|---|---|---|
| vit-l-lr5e-5-s1 | 11395200 | ViTPose++-L | 5e-5 |
| vit-l-lr1e-5-s1 | 11395203 | ViTPose++-L | 1e-5 |
| vit-h-lr5e-5-s1 | 11395205 | ViTPose++-H | 5e-5 |
| vit-h-lr1e-5-s1 | 11395206 | ViTPose++-H | 1e-5 |
| eval500-D2-l / -h | 11395213 / 11395216 | final and best checkpoints | `--dependency=afterany` |

All four runs use `TIME=01:30:00`.

**Compare against:**
- zero-shot ViTPose++-L 89.4 and ++-H 90.1
- the soup A1-four-bn at 91.1
- the released hrnet_fidip at 90.7 (Test400)

The second seed of the best LR comes after these results.

### Round 8, first attempt: BatchNorm bug. Cancelled with user approval.

**What went wrong.**
- In the first runs (11395200/03/05/06), Test100 *collapsed* after the first epochs:
  - ViTPose++-L at LR 5e-5 went 92.1 at the start → 2.9 (epoch 1) → 24.5 (epoch 2).
  - At LR 1e-5 it fell 92.1 → 80.9 after epoch 0.
- **Diagnosis (laptop):** the ViTPose++ "classic" decoder has two BatchNorm layers
  (`head.batchnorm1/2`). In `train()` mode they use per-batch statistics. On our small,
  augmented infant batches that roughly doubled the heatmap amplitude (peak 0.84 → 1.97) and
  multiplied the loss by about 100 on the *same* input. Every step then pushed the weights the
  wrong way.
  - There is no dropout or drop-path in the HF config, so BN was the only train/eval difference.
- **Fix:** `ViTPoseNet.train()` keeps all BN modules in eval mode, so they stay on their COCO
  running stats.
  - Laptop check: train-mode and eval-mode losses are now identical.
  - ViTPose++-B goes 89.5 → **92.2 Test100 after one epoch**.
- **Lesson:** a model ported for inference can carry BN layers whose batch statistics are
  inappropriate for small-batch fine-tuning. Check train-mode vs eval-mode loss on one batch
  before launching.

**Job handling.**
- The user approved cancelling all 6 jobs: 4 training runs and their 2 evals.
- The runs were resubmitted under new tags `vit2-*`:

| Run | Job |
|---|---|
| vit2-l-lr5e-5-s1 | 11395455 |
| vit2-h-lr5e-5-s1 | 11395479 |
| vit2-l-lr1e-5-s1 | 11395499 |
| vit2-h-lr1e-5-s1 | 11395504 |
| eval500-D2b-l | 11395527 |
| eval500-D2b-h | 11395547 |

- **Duplicate eval:** the D2b-h eval first reported "Unexpected message received" and was not in
  squeue after more than 60 s, so I resubmitted it once. The original then appeared as 11395551.
  The user approved cancelling that duplicate.
  - Lesson: wait longer than a minute before treating a failed sbatch as not queued.

### Round 8b results (BN fixed), single seed each

| Run | Epochs run | Test100 best (ep) | **Test400 final** | Test400 best-ckpt | Test100 final |
|---|---|---|---|---|---|
| ViTPose++-L, LR 5e-5 | 40 | 94.0 (3) | 90.5 | 91.9 | 93.1 |
| **ViTPose++-L, LR 1e-5** | 40 | 94.9 (8) | **92.0** | 92.3 | 94.5 |
| ViTPose++-H, LR 5e-5 | 24/40 (TIMEOUT) | 95.0 (4) | -- | 92.6 | -- |
| **ViTPose++-H, LR 1e-5** | 24/40 (TIMEOUT) | 95.7 (10) | -- | **92.7** | -- |

- **Fine-tuning ViTPose++ beats every HRNet result.**
  - ViTPose++-L at LR 1e-5, final checkpoint: Test400 92.0 (+1.3 over the released hrnet_fidip at
    90.7) and Test100 94.5 (+0.9 over the paper's 93.6).
  - The best-ckpt numbers are selected on Test100 only, so their Test400 scores remain unbiased.
    For ViTPose++-H that gives 92.7 Test400.
- **LR 5e-5 overfits.** It peaks at epoch 3-4, then Test400 decays: L reaches 90.5 at the final
  checkpoint.
- **LR 1e-5 also peaks early.** Best epochs were 8 for L and 10 for H.
- **ViTPose++-H is too slow for the planned schedule.** It takes about 3.8 min per epoch on a
  V100, so 40 epochs (about 2.5 h) does not fit in 1.5 h.

**Follow-ups submitted:**

| Run | Job | Model / LR | Schedule |
|---|---|---|---|
| vit2-l-lr1e-5-s2 | 11398128 | ViTPose++-L, LR 1e-5 | 40 ep |
| vit2-l-lr1e-5-s3 | 11398145 | ViTPose++-L, LR 1e-5 | 40 ep |
| vit2-h20-lr1e-5-s1 | 11398580 | ViTPose++-H, LR 1e-5 | 20 ep, LR step at 15, TIME 1:45 |
| vit2-h20-lr1e-5-s2 | 11398625 | ViTPose++-H, LR 1e-5 | 20 ep, LR step at 15, TIME 1:45 |

- Evals: eval500-D2c-l (11398156) and eval500-D2d-h (11398679).
- vit2-l-lr1e-5-s3 reported "Unexpected message" but did queue.

### Round 8 final results (2026-10-27): stage D2 complete

| Recipe | n | **Test400 final** | Test100 final | Δ vs released hrnet_fidip (Test400 / Test100) |
|---|---|---|---|---|
| **ViTPose++-H, LR 1e-5, 20 ep** | 2 | **92.8 ± 0.1** (92.7, 92.8) | **95.2 ± 0.2** | **+2.1 / +1.6** |
| ViTPose++-L, LR 1e-5, 40 ep | 3 | 91.9 ± 0.3 (92.0, 91.5, 92.1) | 94.5 ± 0.1 | +1.2 / +0.9 |
| HRNet soup A1-four-bn (round 7) | 1 | 91.1 | 92.9 | +0.4 / −0.7 |
| HRNet ft40 (our best trained, round 4) | 2 | 90.2 ± 0.2 | ~92.0 | −0.5 / −1.6 |
| released hrnet_fidip | 1 | 90.7 | 93.6 | -- |

Jobs:
- L seeds 2 and 3: 11398128 and 11398145, evaluated by 11398156.
- H20 seeds: 11398580 and 11398625, evaluated by 11398679.
- All completed within their time limits.
  - The H20 runs took about 80 min each.
  - The L seeds took about 90 min. They shared node d1002, which made them slower than s1.

**Conclusions**
- **PLAN.md hypothesis D2 is confirmed.** Fine-tuned ViTPose++-H beats the released FiDIP
  model by +2.1 AP on Test400, the held-out set.
  - That is far beyond the ≥ 0.5 rule; the seed std is 0.1.
  - It also wins on the paper's own Test100: 95.2 vs 93.6.
  - It is our new best configuration.
- **Model size matters.**
  - H beats L by +0.9 Test400, with fewer epochs.
  - Zero-shot (no infant training) the gap was 0.7. Fine-tuning adds about +2.4 to L and +2.7
    to H on Test400.
- **Low LR and short schedules.**
  - Test100 peaks around epoch 8-12 at LR 1e-5 and around epoch 3-4 at LR 5e-5. LR 5e-5
    overfits.
  - The 20-epoch H schedule has final ≈ best, so no early-stopping pick is needed.
- **Why it works (interpretation).** The gain comes from a stronger adult-pose prior: COCO + AIC
  + MPII ViTPose++ pretraining at scale. The domain-adaptation machinery is not the source.
  - This agrees with Jahn et al. (2025), and with our rounds 1-4 finding that FiDIP's
    adversarial term does not help.
- **Still untested:**
  - weight-soups of the ViT seeds (stage A applied to D)
  - layer-wise LR decay
  - stages B and C on the ViT
  - E, which is still gated on unlabelled data

## 2026-10-27 -- Round 9: delete BN-bug outputs; soup the ViTPose seeds (stage A on D)

User: "Yes, delete them and soup the H20 seeds."

**Cleanup.**
- I listed the targets first, then deleted them by literal path:
  - `output/vit-{h,l}-lr{1e-5,5e-5}-s1`, on /projects, about 7.6 GB
  - the matching `log/vit-*`
- All `vit2-*` runs are untouched.

**Soups.**
- `scripts/vit_soups.sbatch` (job 11400326, short partition, CPU, 2 min) builds uniform averages
  of the `final_state.pth` files:
  - **D-soup-h20**: H20 s1+s2. This was pre-registered as the soup we report.
  - D-soup-l3: L lr1e-5 s1+s2+s3.
- No BN recalibration. The ViTPose decoder BN is frozen during fine-tuning, so its stats are
  already the COCO ones. `weight_average.py` now refuses `--recal-bn` for ViTPose.
- Scored by eval500-Dsoup-h/l (11400360, 11400382).

| Soup | Test400 | vs members' mean | Test100 | vs members' mean |
|---|---|---|---|---|
| **D-soup-h20** | **93.1** | +0.3 | **95.7** | +0.4 |
| D-soup-l3 | 92.2 | +0.4 | 94.7 | +0.2 |

**Conclusions**
- Souping helps ViTPose a little: +0.3 to +0.4 Test400, in the same direction for both model
  sizes.
  - That is below our 0.5 rule, so it counts as a consistent small gain, not a decisive one.
  - It is smaller than the +0.9 seen for HRNet. Our reading: HRNet seeds were noisier
    (std 0.2-0.3, and including the aug-cj variants), so averaging had more variance to remove.
- **Best single model of the project: `output/soups/D-soup-h20.pth`.**
  - **Test400 93.1, Test100 95.7.** The released hrnet_fidip scores 90.7 / 93.6, so this is
    **+2.4 / +2.1**.
  - It is a pre-registered uniform average of the two H20 seeds, not a pick by test score.
- **The reported recipe number stays the 2-seed H20 mean: 92.8 ± 0.1 Test400.** The soup is
  reported alongside it.

## 2026-10-27 -- Round 10: stages C2 (layer-wise LR decay) and B (real:synthetic mix) on ViTPose++-H

User: "Move into C2 then B next."

Baseline for both stages is the H20 recipe:
- ViTPose++-H, LR 1e-5, 20 epochs, LR step at epoch 15
- 2 seeds score Test400 92.8 ± 0.1 and Test100 95.2 ± 0.2
- the soup scores 93.1

Each config gets one seed first, as a screen. Configs that beat the baseline by ≥ 0.3 get a
second seed; a claim still needs ≥ 0.5 with 2 seeds.

**C2: layer-wise LR decay** (`TRAIN.LAYER_DECAY`, already in `train_vitpose.py`; groups checked
on the laptop).
- Block i of 32 gets LR × decay^(33−i). The embeddings get decay^33, which is 0.18 at 0.95 and
  0.03 at 0.9. The decoder keeps the full LR.
- Hypothesis: smaller updates to the generic early layers reduce overfitting to 1200 images.

| Run | Job | Settings |
|---|---|---|
| c2-ld95-lr1e-5-s1 | 11400434 | gentle decay |
| c2-ld90-lr1e-5-s1 | 11400442 | strong decay; nearly freezes early blocks |
| c2-ld90-lr3e-5-s1 | 11400455 | strong decay + 3× base LR, so middle blocks get about the baseline LR and the head gets more |
| eval500-C2 | 11400473 | evaluation |

**B: real:synthetic mix, re-scoped after reading the sampler.**
- `utils/sampler.BalancedBatchSampler` already makes every epoch 50% real / 50% synthetic:
  1000 synthetic + 200 real × 5. So "oversample the real images" (the original B1) is already
  the baseline.
- That sampler also **never shuffles**. Every epoch visits images in the same order; only the
  augmentation changes.
- New `MixSampler` (`TRAIN.REAL_FRACTION`):
  - Same epoch length (2000).
  - The chosen share of real images, with each pool cycled in a fresh random order.
  - Fully shuffled.
  - Composition checked on the laptop: at f = 0.5, every real image appears exactly 5 times.

| Run | Job | Real fraction | Purpose |
|---|---|---|---|
| b-rf17-s1 | 11400506 | 1/6 | the dataset's natural mix, no oversampling |
| b-rf50-s1 | 11400510 | 0.5 | control: baseline mix + shuffling, isolates the shuffle effect |
| b-rf75-s1 | 11400521 | 0.75 | |
| b-rf100-s1 | 11400537 | 1.0 | real only = B2 |

- The eval for the B runs will be submitted once a slot frees up; the 8-job cap is reached.

### Round 10 results (2026-10-27): C2 and B are both null

All 7 runs and both evals (11400473, 11401502) completed. Full table:
`results/eval500_summary.md`.

| Run | Test400 final | Test100 final |
|---|---|---|
| baseline H20 (2 seeds) | 92.7, 92.8 | 95.1, 95.4 |
| c2-ld95 | 92.7 | 95.6 |
| c2-ld90 | 92.7 | 94.7 |
| c2-ld90-lr3e-5 | 92.9 | 95.3 |
| b-rf17 (natural 1/6) | 92.6 | 95.6 |
| b-rf50 (shuffled 50/50) | 93.0 | 95.7 |
| b-rf75 | 92.8 | 94.9 |
| b-rf100 (real only) | 92.7 | 94.7 |

**Every config is within −0.2 to +0.2 of the baseline on Test400.**
- None reaches the +0.3 screen, so no second seeds (pre-registered rule).
- The spread across all 7 runs is 92.6-93.0. That is consistent with seed noise of about
  ±0.1-0.2.

**C2, layer-wise LR decay: no effect.**
- Even nearly freezing the early blocks (decay 0.9 at LR 1e-5) leaves Test400 unchanged.
  Strong decay with the base LR does learn more slowly on Test100: 94.7 final, and the best epoch
  is 13 rather than 8-12.
- This matches the WiSE-FT result in round 7. Limiting how far the backbone moves from its
  pretrained weights neither helps nor hurts.
- Interpretation: at LR 1e-5 for 20 epochs, the fine-tune is already gentle enough.

**B, real:synthetic mix: no effect on Test400.**
- **Synthetic data adds nothing measurable for ViTPose++-H.** Training on the 200 real images
  alone (b-rf100) gives Test400 92.7, the same as the baseline.
- The natural 1/6 mix without oversampling gives 92.6.
- There is a weak hint on Test100: rf75 and rf100 finish at 94.7-94.9 vs 95.1-95.7 with more
  synthetic data. But that is single-seed on a 100-image set, below our noise threshold, and it
  is absent on Test400.
- Shuffling (rf50 vs the unshuffled baseline) gives +0.2, which is not significant.
- Implication for PLAN.md:
  - §7.1 (generating more synthetic infants with `syn_generation/`) is unlikely to pay off for
    this backbone.
  - The original FiDIP paper's synthetic-data gain was measured on a much weaker adult-pose prior.

**Where this leaves the project.** The recipe is saturated around Test400 92.8-93.0, and the
soup is at 93.1.
- The remaining levers are data the model has never seen (stage E, unlabelled real infant
  images) and a different architecture or resolution.
- Training hyper-parameters, layer-wise LR and data mix are not levers here.

## Round 11 (2026-10-27): real-only second seed + ViTPose++-H at 384x288

User request: "Run both optional next steps."

**B2 confirmation.** `b-rf100-s2` (job 11410487) repeats b-rf100-s1 exactly: ViTPose++-H,
LR 1e-5, 20 epochs, `TRAIN.REAL_FRACTION 1.0` (200 real images only). A 2-seed real-only mean
within 0.5 of the 92.8 baseline would confirm "synthetic data adds nothing for ViTPose++-H".
Evaluated by eval500-B2 (11410741).

**D3: input resolution 384x288 (was 256x192).**
- HF `VitPosePatchEmbeddings` raises on any input size other than the checkpoint's. So
  `adaptive_pose_vitpose.py` now bicubic-interpolates the position embeddings from a 16x12 to a
  24x18 grid, and updates the size the decoder head uses to reshape tokens.
  - Laptop check: ViTPose++-B zero-shot Test100 is 89.3 at 384x288 vs 89.8 native. So the
    interpolation is sane, though not free before fine-tuning.
- Config `experiments/syrip/vitpose/vitpose_plus_384x288_infant.yaml`: heatmap 96x72, sigma 3,
  same as ViTPose's own 384x288 configs. Recipe is otherwise H20 (LR 1e-5, 20 ep, step at 15).
- **Memory.** Rough estimate: batch 32 at 432 tokens needs about 35 GB of activations plus
  10 GB of weights and AdamW state. That does not fit a 32 GB V100, so the run uses
  `TRAIN.GRAD_CKPT True` (gradient checkpointing). This costs ~30% more compute and has no
  effect on the maths.
- **Wall time.** Measured on a V100: 10.5 s/iteration, about 11-12 min/epoch
  (~4 h for 20 epochs). Zero-shot ++-H at 384x288 scores Test100 91.3 (92.7 native). To keep the short-job habit, `train_vitpose.py` now supports
  `AUTO_RESUME`:
  - It saves `checkpoint.pth` (model, AdamW, scheduler, scaler, best score, losses) after every
    epoch, via an atomic rename.
  - It resumes from that checkpoint, and deletes it when the run finishes.
  - A spare chained job that starts after `final_state.pth` exists exits without training.
  - Laptop test (ViTPose++-B): killed after epoch 0, then resumed. The resumed run starts at
    epoch 1 with the stepped LR and the saved best score, and finishes normally.
- Jobs: `d3-h384-lr1e-5-s1` 11410716, then -p2 11410725 and -p3 11410732 (each 1:45,
  `afterany`), then eval500-D3 11410743 (384 cfg).
- Screening rule unchanged: second seed only if Test400 ≥ 93.1 (+0.3).

### Round 11 results (2026-10-27): both null

| Run | Test400 final | Test100 final |
|---|---|---|
| baseline H20 256x192 (2 seeds) | 92.7, 92.8 → **92.8 ± 0.1** | 95.1, 95.4 → 95.3 |
| real only (b-rf100-s1, s2) | 92.7, 92.9 → **92.8 ± 0.1** | 94.7, 94.4 → 94.6 |
| ViTPose++-H 384x288 (d3-h384, 1 seed) | **92.7** (best ckpt 92.8) | 95.0 (best 95.5, ep 6) |

**B2 confirmed with 2 seeds: synthetic data adds nothing on Test400 for ViTPose++-H.**
- On Test100 (the paper's set), dropping synthetic data costs 0.7. That is now consistent
  across both seeds, and slightly above Test100's ±0.5 noise.
- The effect does not appear on the 4× larger, unbiased Test400. So it is reported as a
  Test100-only effect, not as a held-out benefit of synthetic data.

**D3, input resolution 384x288: no gain, at about 2.8× the GPU time.**
- Zero-shot drops at the higher resolution, because of the interpolated position embeddings:
  Test100 91.3 vs 92.7.
- Fine-tuning recovers to the baseline level, but not beyond it.
- Test400 92.7 is below the +0.3 screen, so no second seed.
- **Ops notes:**
  - The chained resumable jobs worked as designed. Part 1 timed out after epoch 7, part 2
    resumed at epoch 8 and timed out during epoch 16, and part 3 finished epochs 16-19.
  - The 10.8 GB `checkpoint.pth` was deleted at the end.
  - Peak memory with gradient checkpointing fit a 32 GB V100.

**Conclusion after round 11.** Every recipe-level change on ViTPose++-H lands at Test400 92.6-93.0:
- layer-wise LR decay (C2)
- real:synthetic mix (B, B2)
- input resolution (D3)

The best results remain:
- the H20 recipe, 92.8 ± 0.1 (Test100 95.2)
- the D-soup-h20 seed soup, 93.1 / 95.7 (output/soups/D-soup-h20.pth)

The only untested lever in PLAN.md is stage E (unlabelled real infant images).

## 2026-10-27 -- Round 12: the user's improvement plan before acquiring new video

**Asked:** "Parse through the attached plan .md. Proceed with the plan as documented there"
(`draft_plan_fidip.md`, now summarised with the pre-registration in PLAN.md "Round 12").
Experiments, in the plan's order:
- 0: evaluation/pipeline audit
- 1: training-label audit
- 2: ranking loss
- 3: mean teacher on the existing unannotated images
- 4: Sapiens2 screen

### Experiment 0: evaluation and pipeline audit (done)

**Provenance / overlap** (`scripts/audit_overlap.py`, laptop, 8 min). It compares the 200 labelled real
training images, the 300 unannotated images in `images/train_infant` and Test500 by:
- YouTube video id
- 64-bit DCT perceptual hash (distance ≤ 10)
- ORB + RANSAC homography inliers (≥ 40)

Checked visually: hash-0 pairs are the same frame, and 40-60 inlier pairs are other frames of the same video.

| Finding | Count |
|---|---|
| Train YouTube videos that also supply test images | 23 of 25 |
| Test400 images with a direct link (video/hash/scene) to a labelled training image | 147 (Test400_indep = 253) |
| Test100 images with such a link | 58 |
| Near-identical train/test pairs (hash ≤ 4) | 6 (e.g. train00315/test42, train00340/test50) |
| Unannotated images that are near-exact copies of a Test500 image | 204 of 300 |
| Unannotated images sharing a source with any test image | 245 of 300 |
| Unannotated images with no link to test or labelled images (eligible) | **52** |

This leakage is in the original SyRIP split. Every number reported so far is affected the same way.
New reporting: Test400_indep alongside Test400.

**Retrospective bootstrap** (`scripts/bootstrap_ap.py`): paired, resampling source groups,
2000 resamples, recomputing AP each time. File: `output/audit12/bootstrap_retro.csv`.

| Comparison (B − A) | Test400 | Test400_indep | Test100 |
|---|---|---|---|
| H20-s1 − released | +2.0 [+1.2, +3.0] | +2.4 [+1.2, +3.3] | +1.5 [+0.1, +3.0] |
| D-soup-h20 − released | +2.4 [+1.5, +3.2] | +2.7 [+1.4, +3.5] | +2.0 [+0.7, +3.3] |
| D-soup-h20 − H20-s1 | +0.4 [+0.0, +0.5] | +0.3 [+0.0, +0.5] | +0.6 [+0.0, +1.1] |
| H20-s2 − H20-s1 (seed noise) | +0.1 [−0.4, +0.5] | +0.0 [−0.3, +0.2] | +0.3 [−0.5, +1.1] |
| b-rf100-s2 − H20-s2 (real only) | +0.0 [−0.8, +0.7] | +0.3 [−0.4, +1.0] | −1.0 [−2.5, −0.1] |
| d3-h384 − H20-s1 | −0.0 [−0.8, +0.7] | −0.3 [−0.9, +0.4] | −0.1 [−0.8, +0.7] |
| c2-ld90 − H20-s1 | −0.0 [−0.5, +0.5] | +0.4 [−0.2, +0.7] | −0.4 [−1.1, +0.4] |

- The ViTPose gain over FiDIP holds on the train-independent subset.
- The soup gain is small but its CI excludes 0.
- Seed-to-seed CIs are about ±0.4, so the +0.5 rule is about one evaluation-noise width.

**Pipeline parity** (`tools/parity_vitpose.py`, zero-shot ++-B, laptop):
- Encode→decode round trip: repo 0.008 px, HF 0.0001 px. Repo targets decoded by HF post-processing: 1.45 px.
- The repo centres the box at x+(w−1)/2; HF centres it at x+w/2.
- DARK blur: cv2 11×11 (σ≈2) in the repo vs σ=0.8 in HF.

| crop + refinement | Test400 | Test100 |
|---|---|---|
| repo + repo (our eval) | 87.82 | 89.50 |
| repo + HF | 87.45 | 89.95 |
| HF + HF (tools/test_vitpose.py) | 87.47 | 89.79 |
| HF + repo | 87.64 | 89.25 |

±0.3-0.4 AP with no consistent winner. There is no bug, but decoder choices alone move AP by our
decision-threshold size. Decoder frozen.

**Validation for method selection:** 5 grouped folds over the 200 labelled real images
(`scripts/make_cv_folds.py`, groups from direct train-train links, 124 groups, 40 images per fold).
`data/syrip_cv/f<k>/` (gitignored) holds the fold jsons; test set `heldout_infant`
(`lib/dataset/syrip.py`). Metric: pooled OOF AP of final-epoch checkpoints.

### Implemented for experiments 1-4

- `lib/core/rank_loss.py`: RSPose-style S-Rank + S-Sort (+ I-Sort), `LOSS.TYPE rspose`.
  - No official code exists, so this is our implementation from the paper's equations, with AP-Loss/RS-Loss identity-update gradients.
  - The vectorised gradients match a loop port of Oksuz et al.'s RankSort to 1e-9.
  - Positives are pixels with target > e^-4.5 (our targets are untruncated Gaussians).
- `lib/dataset/unlabeled.py` + `TRAIN.MT_*`: mean teacher.
  - Weak crop goes to the flip-averaged EMA teacher; peaks above the threshold become Gaussian targets in the strong (scale/rot/flip/colour) view.
  - Unit test: the pseudo-target peak lands within 0.61 heatmap px of the analytically transformed point over 40 random augmentations, 16 of them flipped (left/right swapped).
  - `tools/estimate_boxes.py` gives the unannotated images person boxes (2-pass zero-shot ++-H).
- `tools/label_audit.py`: flags swap / far / visibility / outside-box joints from OOF predictions plus a zero-shot second opinion.
  - Adds 20 random unflagged images.
  - Writes a **local** review page (`output/audit12/labels/review.html`, SyRIP images, never published) that exports decisions.json.
- `scripts/sapiens2_setup.sbatch`: separate venv `~/venvs/sapiens2` (torch 2.7.1 cu126, transformers 5.x) and the 0.4B/1B checkpoints.
  - Sapiens2 is on HF ungated (`facebook/sapiens2-pose-*`, `Sapiens2ForPoseEstimation`, input 1024x768, 308 Goliath keypoints).

**Laptop smoke tests** (++-B, fold 0, 40 held-out images, batch 16):

| Run | ep0 | ep1 | Notes |
|---|---|---|---|
| zero-shot | 88.3 | | |
| MSE | 88.9 | 89.7 | |
| RS spatial | 87.9 | 89.6 | |
| RS + I-Sort | 86.4 | | 1 epoch only |
| mean teacher (labelled pool only) | 88.0 | | 92% of pseudo-joints accepted, ~1.7x step time |

### Submitted

| Tag | Job | What |
|---|---|---|
| cv-h20-f0..f4 | 11422061 11422072 11422076 11422081 11422084 | baseline H20, folds 0-4 (TIME 1:45) |
| cv-rs-f0, f1 | 11422319 11422326 | E2 spatial RS, folds 0-1 (f2-f4 when slots free) |
| sapiens2-setup | 11422150 | venv + checkpoints (short partition) |
| r12-prep | (after setup) | boxes for the 52 eligible images; zero-shot ++-H on 200R |

### Results (2026-10-28)

**Baseline CV (cv-h20):** pooled OOF AP **94.39**; folds 92.9 / 94.0 / 95.2 / 94.7 / 95.0.

**E2, spatial RSPose (cv-rs):** pooled 94.03, **-0.36** vs baseline. Fold diffs -0.9 / +0.1 / -0.1 / -0.3 / -0.4, better in 1/5. **No advance.**
E2b (I-Sort) is skipped: the pre-registered condition was that spatial RS looks promising.

**E4, Sapiens2 zero-shot** (`tools/test_sapiens2.py`, GT boxes, flip test, job 11424809):

| Model | real200 (clean) | Test400 | Test100 | Test400_indep |
|---|---|---|---|---|
| 0.4b | 94.1 | 91.6 | 92.9 | 92.0 |
| 1b | 94.8 | 92.1 | 93.4 | 92.5 |
| H20-s1 (ref) | 94.39 (OOF) | 92.7 | 95.1 | 92.6 |

- The overlays confirm the Goliath->COCO mapping.
- Sapiens2-1b has never seen SyRIP, yet it matches the fine-tuned H20 on clean data:
  - real200: +0.37, better in 3/5 folds;
  - Test400_indep: -0.06, CI [-1.0, +1.6].
- It trails on the linked subsets (Test400_linked -1.5, Test100 -1.7). That is consistent with H20 benefiting from train/test source overlap.

**E4 follow-up, keypoint ensemble** (`scripts/ensemble_kpts.py`):
- Rule: 0.5/0.5 average of coordinates and scores; instance score recomputed. Fixed before scoring, no weight tuning.
- This combination was not a pre-registered candidate. It answers the plan's E4 question ("stronger or *complementary*?").

| Comparison | Set | Diff [95% CI] |
|---|---|---|
| OOF H20 + S2-1b vs cv-h20 | real200 CV | **+1.55** pooled, 5/5 folds better (94.3 / 96.3 / 96.1 / 96.9 / 95.7) |
| H20-s1 + S2-1b vs H20-s1 | Test400 | **93.8**, +1.06 [+0.68, +1.62] |
| | Test400_indep | +1.32 [+0.65, +1.73] |
| | Test100 | 95.3, +0.19 [-0.57, +1.01] |
| H20-s2 + S2-1b vs H20-s2 | Test400 | **93.8**, +0.96 [+0.58, +1.73] |
| | Test400_indep | +1.18 [+0.55, +1.91] |
| | Test100 | +0.17 (n.s.) |
| H20-s1 + S2-1b vs D-soup-h20 | Test400 | +0.70 [+0.39, +1.36] |
| | Test400_indep | +1.04 |
| | Test100 | -0.39 (n.s.) |
| D-soup + S2-1b vs D-soup-h20 | Test400 | **94.0**, +0.94 [+0.42, +1.48] |
| | Test400_indep | +1.10 |
| | Test100 | -0.24 (n.s.) |

- The gain is largest on Test400_indep, the subset with no source overlap. It is absent on Test100.
- Cost: inference runs two models. Sapiens2-1b runs at 1024x768 in a separate venv.

**E3 mean teacher:**
- Folds needed 2-3 chained jobs: about 10 min/epoch with GRAD_CKPT.
- cv-mt-f0-p2 (11430056) was killed with signal 9; MaxRSS was 47 GB against a 32 GB request.
- It was resubmitted as p3 (11434194) with AUTO_RESUME.

**E1 label audit:**
- Flags: 16 images, 19 flags (17 swap, 1 far, 1 vis), plus 20 random images.
- Waiting for human adjudication.
