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
