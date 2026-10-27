### HRNet-W48 (384x288)

| Setup | n | **Test400** (final ckpt) | Test400 (best ckpt) | Test100 (final) | Test100 (best) |
|---|---|---|---|---|---|
| Released FiDIP (`hrnet_fidip.pth`) | 1 | -- | 90.7 | -- | 93.6 |
| Fine-tune, 40 ep + aug: colour jitter 0.4 | 2 | 90.3 ± 0.0 | 90.0 ± 0.3 | 92.3 ± 0.2 | 92.8 ± 0.5 |
| Fine-tune, 40 ep + LR step 30 (baseline recipe) | 2 | 90.2 ± 0.2 | 90.0 ± 0.6 | 92.0 ± 0.5 | 92.8 ± 0.5 |
| Fine-tune, 40 ep + aug: rot 60, scale 0.5 | 2 | 90.1 ± 0.3 | 89.9 ± 0.7 | 93.3 ± 0.6 | 93.6 ± 0.4 |
| Adversarial only, 20 ep | 3 | 90.0 ± 0.3 | 90.0 ± 0.3 | 92.0 ± 0.6 | 92.5 ± 0.4 |
| Adversarial only, 40 ep + LR step 30 | 2 | 89.8 ± 0.2 | 89.1 ± 0.4 | 91.7 ± 0.1 | 92.1 ± 0.2 |
| Fine-tune (shipped cfg), 20 ep | 3 | 89.7 ± 0.4 | 89.7 ± 0.3 | 92.3 ± 0.8 | 92.7 ± 0.2 |
| Pre-training only, 20 ep | 1 | 89.5 | 89.5 | 92.0 | 92.0 |
| FiDIP settings (LAMBDA 5e-4, PRE 10), 20 ep | 3 | 88.8 ± 0.5 | 88.9 ± 0.3 | 90.9 ± 0.4 | 91.7 ± 0.4 |
| FiDIP + PRE_BN_FREEZE, 20 ep | 3 | 88.2 ± 1.1 | 89.3 ± 0.4 | 90.6 ± 1.3 | 91.9 ± 0.6 |
| FiDIP + LR_STEP [10,15], 20 ep | 1 | 88.5 | 88.5 | 91.2 | 91.2 |
| Fine-tune, 40 ep + aug: light (rot 15, scale 0.15, no half-body) | 2 | 88.4 ± 0.1 | 88.3 ± 0.2 | 90.6 ± 0.2 | 91.3 ± 0.4 |
| FiDIP + confusion loss 5e-4, 20 ep | 1 | 83.9 | 86.7 | 86.4 | 91.1 |

### Pose-MobileNetV2 (224x224)

| Setup | n | **Test400** (final ckpt) | Test400 (best ckpt) | Test100 (final) | Test100 (best) |
|---|---|---|---|---|---|
| Released FiDIP (`mobile_fidip.pth`, COCO-pretrained) | 1 | -- | 68.3 | -- | 79.2 |
| MobileNetV2 ImageNet init, fine-tune, 200 ep (LR steps 150, 180) | 2 | 45.4 ± 1.0 | 45.7 ± 0.8 | 58.5 ± 0.6 | 60.6 ± 0.3 |
| MobileNetV2 ImageNet init, fine-tune, 120 ep (LR steps 90, 110) | 2 | 43.8 ± 0.1 | 45.2 ± 1.7 | 56.3 ± 0.5 | 57.9 ± 0.1 |
| MobileNetV2 ImageNet init, fine-tune, 40 ep | 1 | 43.5 | 43.5 | 57.3 | 57.3 |
| MobileNetV2 ImageNet init, FiDIP (LAMBDA 5e-4, PRE 10), 40 ep | 1 | 8.7 | 12.1 | 9.0 | 17.2 |

adv40-s1 (timed out on a slow node) and aug-cj-s1 (cancelled on a slow node) are excluded. Released models have a single checkpoint.

### Round 7: stage A weight soups (HRNet-W48, no training) -- single measurement each

Members: ft40-s1/s2 Test400 90.3/90.0, aug-cj-s2/s3 90.3/90.3 (mean 90.2). `-bn` = BN stats recalibrated on train_infant, `-raw` = BN stats averaged.

| Soup | Test400 raw / **bn** | Test100 raw / bn |
|---|---|---|
| A1 ft40 s1+s2 | 90.7 / 90.8 | 92.6 / 93.2 |
| A1 aug-cj s2+s3 | 90.5 / 91.1 | 92.3 / 92.9 |
| **A1 four (pre-registered report)** | 90.7 / **91.1** | 92.5 / 92.9 |
| A1 six (+ aug-geo s1/s2) | 91.0 / 91.0 | 93.0 / 93.3 |
| A2 WiSE-FT α=0.5 (s1, s2) | 89.8, 89.4 / 89.2, 89.3 | 92.5, 93.0 / 92.3, 92.5 |
| A2 WiSE-FT α=0.7 | 90.5, 90.7 / 90.4, 90.2 | 93.2, 93.3 / 92.8, 93.2 |
| A2 WiSE-FT α=0.8 | 90.5, 90.9 / 90.8, 90.6 | 93.3, 93.3 / 92.8, 93.5 |
| A2 WiSE-FT α=0.9 | 90.7, 90.5 / 90.7, 90.7 | 92.8, 92.6 / 93.0, 93.2 |
| A3 released 50% + ft40 (s1, s2) | 91.1, 90.9 / 90.9, 90.9 | 93.6, 93.6 / 93.1, 93.2 |
| A3 released 70% + ft40 (s1, s2) | 91.1, 91.0 / 90.3, 90.7 | 94.0, 93.6 / 93.0, 93.2 |

### Round 7: stage D1 zero-shot (COCO-trained, no infant training, GT boxes, flip test)

| Model | Test400 | Test100 |
|---|---|---|
| HRNet-W48 DarkPose COCO (`w48_384×288.pth`) | 86.7 | 88.5 |
| ViTPose-B simple decoder | 84.8 | 86.1 |
| ViTPose-B | 84.2 | 87.3 |
| ViTPose++-B (COCO expert) | 87.5 | 89.8 |
| ViTPose++-L | 89.4 | 92.3 |
| **ViTPose++-H** | **90.1** | **92.7** |

### Round 8: stage D2, fine-tuned ViTPose++ (256x192, AdamW, decoder BN frozen; HRNet data pipeline)

"best" = checkpoint picked on Test100 during training, so its Test400 score is still held out. Mean ± sample std over seeds.

| Setup | n | **Test400** (final ckpt) | Test400 (best ckpt) | Test100 (final) | Test100 (best) |
|---|---|---|---|---|---|
| **ViTPose++-H, LR 1e-5, 20 ep (LR step 15)** | 2 | **92.8 ± 0.1** | 92.8 ± 0.3 | **95.2 ± 0.2** | 95.6 ± 0.0 |
| ViTPose++-H, LR 1e-5, 40 ep (timed out at ep 24) | 1 | -- | 92.7 | -- | 95.7 |
| ViTPose++-H, LR 5e-5, 40 ep (timed out at ep 24) | 1 | -- | 92.6 | -- | 95.0 |
| ViTPose++-L, LR 1e-5, 40 ep (LR step 30) | 3 | 91.9 ± 0.3 | 92.3 ± 0.1 | 94.5 ± 0.1 | 94.9 ± 0.2 |
| ViTPose++-L, LR 5e-5, 40 ep | 1 | 90.5 | 91.9 | 93.1 | 94.0 |
| *reference: released hrnet_fidip* | 1 | 90.7 | -- | 93.6 | -- |

### Round 9: stage A on D, uniform seed soups of fine-tuned ViTPose++ (no BN recalibration) -- single measurement each

| Soup | Members' Test400 (mean) | **Test400** | Members' Test100 (mean) | Test100 |
|---|---|---|---|---|
| **D-soup-h20** (H20 s1+s2) | 92.7, 92.8 (92.8) | **93.1** | 95.1, 95.4 (95.2) | **95.7** |
| D-soup-l3 (L lr1e-5 s1+s2+s3) | 92.0, 91.5, 92.1 (91.9) | 92.2 | 94.5, 94.4, 94.6 (94.5) | 94.7 |

### Round 10: stages C2 (layer-wise LR decay) and B (real:synthetic mix) on ViTPose++-H, 20 ep, single seed each (screen)

Baseline is H20, LR 1e-5, with the original 50/50 unshuffled sampler and no decay: Test400 92.7 / 92.8 (2 seeds), Test100 95.1 / 95.4.

| Setup | **Test400** (final) | Test400 (best) | Test100 (final) | Test100 (best) |
|---|---|---|---|---|
| C2 layer decay 0.95, LR 1e-5 | 92.7 | 92.9 | 95.6 | 95.7 |
| C2 layer decay 0.90, LR 1e-5 | 92.7 | 92.6 | 94.7 | 94.7 |
| C2 layer decay 0.90, LR 3e-5 | 92.9 | 93.0 | 95.3 | 95.6 |
| B real fraction 1/6 (natural mix, shuffled) | 92.6 | 92.7 | 95.6 | 95.9 |
| B real fraction 0.5 (shuffled control) | 93.0 | 93.0 | 95.7 | 95.9 |
| B real fraction 0.75 | 92.8 | 93.1 | 94.9 | 95.6 |
| B real fraction 1.0 (real only, no synthetic) | 92.7 | 93.1 | 94.7 | 95.6 |

### Round 11: B2 confirmation, second seed of real-only training (ViTPose++-H, H20 recipe, REAL_FRACTION 1.0)

| Setup | Seeds | **Test400** (final) | Test100 (final) |
|---|---|---|---|
| Baseline H20 (50/50 real:synthetic) | 2 | **92.8 ± 0.1** (92.7, 92.8) | 95.3 ± 0.2 (95.1, 95.4) |
| Real only (no synthetic) | 2 | **92.8 ± 0.1** (92.7, 92.9) | 94.6 ± 0.2 (94.7, 94.4) |

Test400: no difference, so synthetic data adds nothing measurable on held-out infants. Test100: −0.7 without synthetic, consistent across both seeds but absent on Test400.

### Round 11: D3, ViTPose++-H at 384x288 (H20 recipe, gradient checkpointing, 3 chained jobs) -- single seed (screen)

| Setup | **Test400** (final) | Test400 (best) | Test100 (final) | Test100 (best, ep 6) |
|---|---|---|---|---|
| Baseline H20 at 256x192 (2 seeds) | **92.8 ± 0.1** | 92.8 | 95.3 | -- |
| ViTPose++-H at 384x288 | **92.7** | 92.8 | 95.0 | 95.5 |

Zero-shot at 384x288 (interpolated position embeddings): Test100 91.3 vs 92.7 at 256x192. Null: below the +0.3 screen, so no second seed. It costs about 2.8x the GPU time.
