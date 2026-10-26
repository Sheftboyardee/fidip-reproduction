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
