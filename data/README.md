# Data

Latency tables, learned thresholds and the placement of every method on the budgets, as used in the paper.

| Tag | Model (timm unless noted) | Budgets (ToMe r) |
|---|---|---|
| deits | deit_small_patch16_224.fb_in1k | 8, 12, 16 |
| deits_orin_cpu | deit_small_patch16_224.fb_in1k, Jetson Orin Nano CPU | 8, 12, 16 |
| deitt | deit_tiny_patch16_224.fb_in1k | 8, 12, 16 |
| deitb | deit_base_patch16_224.fb_in1k | 8, 12, 16 |
| augs | vit_small_patch16_224.augreg_in21k_ft_in1k | 8, 12, 16 |
| augb | vit_base_patch16_224.augreg_in21k_ft_in1k | 8, 12, 16 |
| maeb | mae_vit_base_patch16_224.ft_in1k | 8, 12, 16 |
| mael | mae_vit_large_patch16_224.ft_in1k | 4, 6, 8 |
| d3s384 | deit3_small_patch16_384.fb_in1k | 24, 36, 48 |
| clipb16 | vit_base_patch16_clip_224.openai (image encoder) | 8, 12, 16 |
| blip | BLIP ITM base COCO (image encoder) | 24, 36, 48 |
| segvits | Segmenter ViT-S mask, ADE20K | 40, 60, 80 |

## Files

- `latency_tables/<tag>.json`: measured CPU time of each block part against the number of tokens (one thread, batch size one). The desktop tables are measured on an AMD Ryzen 9 9950X.
- `thresholds/<tag>.json`: for each budget `r<r>` and seed `s<k>`, the learned thresholds `theta`, the caps, the two shifts, the thresholds `low` and `high` after the shifts and the caps, and the probability `p` of `low`. Load `low`, `high` and `p` as in the README and pass them to `thresholds_for`. `thresholds/deits.json` holds `theta` only. The DeiT-S placement is in the two files below.
- `deits_desktop_placement.json`: DeiT-S on the desktop CPU. For every method and budget, the two measured settings, their measured latencies and the probability `p` of the first one. The budgets are the median measured latencies of ToMe. Threshold methods also give the thresholds, the shifts, `low` and `high`.
- `deits_orin_placement.json`: the same on the CPU of a Jetson Orin Nano. The thresholds of ours are trained with the table `latency_tables/deits_orin_cpu.json`.

The setting of each image is chosen by `tmt.place.choose` on the index of the image in the evaluation set.
