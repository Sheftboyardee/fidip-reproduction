# Improvement plan: beating the FiDIP baseline (literature review + hypotheses)

Written 2026-10-26 after rounds 1-6 (see AGENT.md). MobileNet is parked; everything below is HRNet
or a stronger backbone.

## Where we stand

| Model | Test100 AP | **Test400 AP** (primary) |
|---|---|---|
| Released `hrnet_fidip.pth` (target to beat) | 93.6 | **90.7** |
| Our best: ft40 (fine-tune 40 ep) | ~93.0 | 90.2 ± 0.2 |
| aug-cj (+ colour jitter 0.4) | -- | 90.3 ± 0.0 |
| FiDIP recipe as published | 91.7 | 88.8 |

Facts from our own runs that shape the plan:
- The domain-adversarial part of FiDIP does not help in our hands. Plain fine-tuning matches or
  beats it, and a large LAMBDA collapses training. Further work on adversarial loss variants is a
  low priority.
- **Noise floor:** seed-to-seed spread on Test400 is ±0.2-0.3 AP. On Test100 it is ±0.5 or more,
  and one lucky Test100 score (aug-geo-s2, 93.7) did not transfer to Test400.
  - **Decision rule:** a change counts as real only if its 2-seed Test400 mean beats the
    comparison by ≥ 0.5 AP. Test100 is reported only for comparison with the paper.
- Training set is 1200 images: 200 real + 1000 synthetic. HRNet runs take 20-40 min, so cheap
  experiments really are cheap.
- Flip test and DARK decoding are already on, so there are no free test-time wins hiding there.

## Literature review (what's new since FiDIP, 2021)

1. **Generic large pose models beat infant-specialised ones on infants.**
   - Jahn et al. (Behav. Res. Methods 2025) compared 7 methods on infant video. ViTPose was best
     *without* fine-tuning, and the specialised infant models had no clear advantage.
   - Jahn et al. (Sci. Rep. 2025) found the same thing: ViTPose was the strongest, and retraining
     on infant data helped further but showed signs of overfitting.
   - The SMaL under-the-cover study (Ostadabbas lab, 2022) also found ViTPose best, and found
     that pretraining on domain-similar data helps.
   - **Takeaway:** the backbone/pretraining is the largest lever, and HRNet-W48 (2019) is no
     longer the strongest starting point.
2. **Human-centric foundation models.**
   - Sapiens (Meta, 2024) is a ViT pretrained on 300M human images (0.3B-2B params). It is
     reported to generalise well when labels are scarce or synthetic.
   - A 2026 paper distils Sapiens-2 pseudo-labels from *unlabelled* infant video into another
     model.
   - Caveat: the Sapiens pose heads are 308-keypoint (Goliath). Whether a COCO-17 checkpoint
     exists still needs checking.
3. **Self-training / mean-teacher beats adversarial alignment for pose domain adaptation.**
   - SHIFT (CVPRW 2025) uses a mean-teacher with pseudo-labels, an infant pose-manifold prior
     (PoseNDF-style) and a keypoint-to-segmentation visibility consistency term. It reports beating
     UniFrame (ECCV 2022) and RegDA, and beating FiDIP on SyRIP→MINI-RGBD.
   - Semi-supervised 2D pose work points the same way: Dual-network (ICCV 2021) addresses the
     collapse problem, and SSPCM (CVPR 2023) does pseudo-label correction.
   - These methods all need *unlabelled target images*, which we don't currently have beyond the
     test set.
4. **Robust fine-tuning / weight averaging.**
   - WiSE-FT (interpolating pretrained and fine-tuned weights) and model soups (averaging seeds)
     improve robustness to distribution shift at zero inference cost.
   - They have not been reported on pose estimation (a gap we can test for free).
   - Our two seeds share an init (COCO W48), and so does the released model, so their weights
     should be compatible for averaging.
5. **Noisy keypoint labels.** REMIND (2026) filters noisy keypoints using their loss
   trajectories in neonatal pose. It reports cutting the noise-induced mAP drop from 22.5% to
   1.8%. This is relevant if SyRIP's real annotations are noisy, but that is unverified for SyRIP.

## Hypotheses, in execution order

Stages are ordered by cost. Each stage's result decides whether the next one is worth running.

### Stage A: free wins from weights we already have (eval-only, ~1 GPU-h)

| ID | Hypothesis | Test |
|---|---|---|
| A1 | **Seed soup:** averaging the weights of fine-tuned seeds that share a COCO init beats either seed. | Uniform average of ft40-s1+s2, then of aug-cj-s1+s2, then all four. BN stats re-estimated on the train set. |
| A2 | **WiSE-FT:** interpolating θ = (1−α)·θ_COCO + α·θ_ft40 keeps adult-pose generality and improves held-out infant AP. | α ∈ {0.5, 0.7, 0.8, 0.9, 1.0}. |
| A3 | **Soup with the released model:** averaging ours with `hrnet_fidip.pth` (same COCO init) beats both. | 50/50 and 70/30. |

Runs needed:
- A new `scripts/weight_average.py` (state-dict lerp plus a BN recalibration pass).
- Scoring with the existing `eval500.sbatch`.

**Go/no-go:** a soup is a valid "our model" if it beats 90.7 on Test400. The plain average of the
4 seeds counts as the result, because a weight chosen by its Test400 score would leak the test set.

### Stage B: what is the synthetic data actually worth? (cheap training, 2 seeds each)

| ID | Hypothesis | Test |
|---|---|---|
| B1 | The 1000 synthetic images dilute the 200 real ones, so oversampling real images helps. | Real images sampled ×3 and ×5 per epoch. |
| B2 | Synthetic data helps at all. | Real-only (200 images) vs real + synthetic. |

This informs whether guide §7.1 (generating more synthetic data with `syn_generation/`) is worth
its considerable cost. If B2 shows synthetic data adds less than 0.5 AP, we skip it.

### Stage C: regularised fine-tuning (cheap, 2 seeds each)

| ID | Hypothesis | Test |
|---|---|---|
| C1 | Freezing the HRNet stem + stage 1-2 reduces overfitting to 1200 images. | Frozen layers. |
| C2 | Layer-wise LR decay achieves the same more softly. | Decay 0.7 per stage. |
| C3 | L2-SP (an L2 pull toward the COCO weights) helps. | λ ∈ {1e-4, 1e-3}. |

These are the training-time counterparts of A2. If A2 shows a gain, C should too, and may combine
with it.

### Stage D: stronger pretrained backbone (the largest expected gain; most engineering)

| ID | Hypothesis | Test |
|---|---|---|
| D1 | **Zero-shot:** COCO-trained ViTPose (B / L / H, Hugging Face `usyd-community/vitpose-*`) without any infant training beats COCO HRNet-W48 zero-shot on Test400. | Our harness with GT boxes and the same OKS scoring. |
| D2 | **Fine-tune ViTPose on SyRIP train** (1200 images) with our best recipe from A-C. ViTPose-B/L should exceed the 90.7 baseline. | Expected because of the literature in items 1-2 above. |
| D3 | *(stretch)* A Sapiens-0.3B backbone with a COCO-17 head, if a checkpoint is available. Otherwise a ViTPose head on Sapiens features. | |

Engineering:
- A small adapter that runs ViTPose inside `lib/` (or a stand-alone `tools/test_vitpose.py`)
  writing the same results JSON, so that `subset_ap.py` scores it unchanged.
- D1 needs no training and decides whether D2 is worth building.

GPU budget:
- ViTPose-L at 256×192 fits easily on one V100/H200.
- Fine-tuning 1200 images × 40-100 epochs is about 1 h, which fits the short-wall-time habit.

### Stage E: self-training on unlabelled infant images (gated on data)

| ID | Hypothesis | Test |
|---|---|---|
| E1 | A mean-teacher (EMA 0.999, pseudo-label confidence 0.5, as in SHIFT) on unlabelled real infant images improves on the best D model. | |
| E2 | Adding SHIFT's pose-manifold prior removes anatomically implausible pseudo-labels and adds gains on top of E1. | |

**Gate:** we need unlabelled real infant images that are *not* Test400. Options:
- Expanded SyRIP: ask the authors what it contains.
- Public infant video with a suitable licence.

Using Test400 images unlabelled (transductive) would only be reportable as a separate, clearly
labelled protocol. Not started until the data question is answered.

### Stage F (stretch): label-noise audit

F1: some SyRIP real training labels are noisy, and down-weighting high-loss keypoints (REMIND
style) helps. First do a cheap check: per-keypoint loss trajectories from an existing run.

## Bookkeeping

- Every run is logged in AGENT.md (hypothesis, result, verdict), and in MANIFEST.md if it produces
  a checkpoint we cite.
- The final report compares, on both Test100 and Test400:
  - the released FiDIP
  - our reproduction
  - the best result of each stage

## Sources

- Huang et al., FiDIP / SyRIP (FG 2021): https://arxiv.org/abs/2010.06100
- Jahn et al., *Automatic infant 2D pose estimation from videos: comparing seven deep neural network methods* (Behav. Res. Methods 2025): https://arxiv.org/abs/2406.17382
- Jahn et al., *Comparison of marker-less 2D image-based methods for infant pose estimation* (Sci. Rep. 2025): https://www.nature.com/articles/s41598-025-96206-0
- *Under the Cover Infant Pose Estimation using Multimodal Data* (SMaL): https://arxiv.org/abs/2210.00662
- Cao et al., AggPose (IJCAI 2022): https://arxiv.org/abs/2205.05277
- Xu et al., ViTPose: https://arxiv.org/abs/2204.12484; HF docs: https://huggingface.co/docs/transformers/main/model_doc/vitpose
- Khirodkar et al., Sapiens (2024): https://arxiv.org/abs/2408.12569
- *Cross-Model Distillation of a Human-Pose Foundation Model from Unannotated Infant Video* (2026): https://arxiv.org/abs/2609.01840
- Bose et al., SHIFT (CVPRW 2025): https://arxiv.org/abs/2504.05789
- Kim et al., UniFrame (ECCV 2022): https://arxiv.org/abs/2204.00172
- Xie et al., Dual-network semi-supervised pose (ICCV 2021): https://arxiv.org/abs/2011.12498
- Huang et al., SSPCM (CVPR 2023): https://arxiv.org/abs/2303.04346
- Wortsman et al., WiSE-FT: https://arxiv.org/abs/2109.01903; Model soups: https://arxiv.org/abs/2203.05482
- REMIND, *The Blind Spot in 2D Infants' Pose Estimation* (2026): https://arxiv.org/abs/2609.04009

## Status (updated 2026-10-27)

| Stage | Status | Result (Test400) |
|---|---|---|
| A: soups / WiSE-FT | done (round 7) | Uniform 4-seed HRNet soup 91.1. That is +0.9 over its members but only +0.4 over released; WiSE-FT gives no gain. |
| D1: ViTPose zero-shot | done (round 7) | ++-H 90.1 with no infant training; COCO HRNet zero-shot 86.7 |
| **D2: ViTPose fine-tune** | **done (round 8)** | **++-H LR 1e-5 20 ep: 92.8 ± 0.1 (Test100 95.2), +2.1 over released FiDIP**; ++-L: 91.9 ± 0.3 |
| A on D: ViTPose seed soups | done (round 9) | **D-soup-h20: 93.1 (Test100 95.7)**, +0.3 over its seeds; L soup 92.2 (+0.4) |
| C2: layer-wise LR decay | done (round 10) | null: 92.7-92.9 vs 92.8 (decay 0.95 / 0.9 / 0.9 at 3x LR) |
| B: real:synthetic mix | done (round 10) | null: 92.6-93.0; **real-only (no synthetic) = 92.7, same as baseline** |
| B2 confirm (2nd real-only seed) | done (round 11) | real-only 2-seed 92.8 ± 0.1 = baseline; Test100 −0.7 |
| D3: 384x288 input | done (round 11) | null: 92.7 (Test100 95.0) at ~2.8x GPU time |
| C1, C3, E, F | not started | C1/C3 deprioritised (C2 and WiSE-FT null); E is the main remaining lever (needs unlabelled data) |

Candidate next steps on the new baseline, cheapest first:
- ~~**A on D:** soup the two H20 seeds~~ (done: 93.1).
- ~~**C2:** layer-wise LR decay~~ (null).
- ~~**B:** real-image oversampling~~ (null; synthetic data adds nothing for ViTPose++-H).
- ~~**384x288 input**~~ (null).
- **E:** mean-teacher, once unlabelled infant images are available.

## Round 12 (2026-10-27): improvement plan before acquiring new infant video

Adopted from the user's draft plan (`draft_plan_fidip.md`, literature search 2026-10-27). Its
conclusion: the completed sweeps show a plateau for the tested ViTPose++-H recipe, not a ceiling
for the existing data. Before acquiring video, run a bounded round on annotation quality,
alternative supervision, the unannotated images we already have, and a newer pretrained model.

| Priority | Experiment | Main question | Initial scope |
|---|---|---|---|
| 0 | Evaluation and pipeline audit | Can the next difference be interpreted reliably? | splits, decoding, metrics, provenance |
| 1 | Training-label audit | Are unreliable annotations limiting learning? | all 200 real training images, human-adjudicated |
| 2 | Ranking loss (RSPose, Keles et al. 2025) | Does MSE leave localisation/ranking information unused? | MSE vs spatial rank+sort; instance-sort as ablation |
| 3 | Mean teacher on existing unannotated images | Can the 300 unannotated images help? | simple EMA teacher, pseudo-label threshold |
| 4 | Sapiens2 pose (2026) | Is a newer pretrained representation stronger or complementary? | zero-shot screen of 0.4B / 1B |

Deprioritised (practical decisions, not proofs): more adversarial FiDIP sweeps, small real:synthetic
changes, more LLRD settings, repeating interpolated 384x288, ImageNet-init MobileNet.

### Experiment 0 findings that change the protocol

- **Train/test source overlap in the original SyRIP split.** 23 of the 25 YouTube videos behind
  training images also supply test images. By video id, perceptual hash or ORB scene matching
  (`scripts/audit_overlap.py`), 147 of the Test400 images and 58 of the Test100 images share a source with a
  labelled training image; a few test images (e.g. test20/42/46/50) are near-identical to training images.
  -> Report **Test400_indep** (253 images with no direct link to a labelled training image) beside Test400.
- **The "300 unannotated images" are mostly test material.** 204 are near-exact copies of Test500
  images and 245 share a source with a test image; only **52** have no link to any test or labelled image.
  Training on the others would be transductive and is excluded from the main protocol.
- **Test400 is now a development benchmark.** It was used to choose methods in rounds 8-11.
  Method selection now uses grouped cross-validation on the 200 labelled real images instead.
- **Pipeline parity** (`tools/parity_vitpose.py`, zero-shot ++-B): the repo pipeline is internally
  consistent (encode/decode round trip 0.008 px); mixing its targets with HF decoding costs 1.45 px.
  Swapping crop or peak refinement between repo and HF moves AP by ±0.3-0.4 with no consistent
  winner, the same size as our +0.5 decision rule. **The decoder is frozen (repo pipeline) for all comparisons.**

### Pre-registered protocol (written before any round 12 training result)

- **Validation (method selection):** 5 grouped folds of the 200 labelled real images
  (`scripts/make_cv_folds.py`; images linked by video/hash/scene stay together; 40 per fold).
  Each fold trains on the other 160 real images + the 1000 synthetic images with the H20 recipe
  (ViTPose++-H, LR 1e-5, 20 epochs, step at 15). Metric: **pooled out-of-fold AP over all 200
  images, final-epoch checkpoints** (area = bbox w*h), plus the per-fold differences.
- **Advance rule:** pooled OOF AP >= baseline + 0.5 **and** better in >= 4 of 5 folds.
- **Frozen confirmation (only for candidates that advance):** train on all data, 2 seeds,
  final checkpoints. Report Test400, Test400_indep and Test100, plus a paired source-group bootstrap 95% CI
  (`scripts/bootstrap_ap.py`) against the H20 seeds. Report soups separately against D-soup-h20 (93.1).
- **Candidates and settings, fixed in advance:**
  - E2: `LOSS.TYPE rspose`, paper's ViTPose-H row (S-Rank δ 0.4 c 1.0, S-Sort δ 1.5 c 2.0); E2b adds I-Sort (δ 3.0, c 1.5).
  - E3: mean teacher, pool = the 52 eligible unannotated images (boxes from zero-shot ++-H,
    `tools/estimate_boxes.py`) + the fold's labelled real images with labels unused. EMA 0.999,
    threshold 0.5 on the flip-averaged teacher peak, weight 1, 16 crops per step. Student final checkpoint is
    the primary model; the EMA teacher is secondary.
    With only 52 new images this is mainly a consistency-regularisation test (the plan's
    weaker-evidence variant). The result must not be read as evidence about unlabelled data in general.
  - E1: unchanged recipe on human-adjudicated corrected labels (corrections recorded with reasons).
  - E4: Sapiens2 zero-shot on all 200 labelled real images (clean for a model never trained on them) and on Test500.
    COCO-17 from Goliath-308 indices [0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14], checked with overlays.
- **Budget:** the plan's 8-12-run cap is revised after the smoke tests, because clean validation
  needs 5 fold runs per screen. Planned: baseline CV + E2 + E3 (+ E2b, E1 if warranted) = 15-25 fold
  runs of ~1.5 GPU-h, plus <= 4 confirmation runs. No hyperparameter sweeps inside a screen.

### Round 12 status (2026-10-28)

| Exp | Result | Decision |
|---|---|---|
| CV baseline (H20) | pooled OOF 94.39 | reference |
| E2 RSPose spatial | -0.36, 1/5 folds | no advance; E2b skipped |
| E3 mean teacher | -0.78, 0/5 folds | no advance |
| E4 Sapiens2-1b zero-shot | +0.37, 3/5 folds; Test400_indep tie | competitive without fine-tuning |
| E4 follow-up: H20 + S2-1b ensemble (fixed 0.5/0.5, not pre-registered) | +1.55, 5/5 folds | Test400 93.8 (both H20 seeds), Test400_indep +1.2-1.3, Test100 n.s. Best system so far |
| E1 corrected labels (18 joints / 16 images, user-adjudicated; 0/20 controls wrong) | -0.11 (3/5) on original labels, -0.13 (2/5) on corrected labels | no advance |

Next candidates:
- Fine-tune Sapiens2 (expensive at 1024x768; must be CV-screened).
- E1-corrected H20 inside the ensemble: -0.22, not pursued.
- Most promising next steps:
  - CV-screened Sapiens2 fine-tune (or a COCO-17 head on Sapiens2 features).
  - Learned/weighted ensembling, which must be tuned inside CV folds only.
  - New infant video (stage E), deduplicated against Test500.
