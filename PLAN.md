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
