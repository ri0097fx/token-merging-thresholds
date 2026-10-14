# Token Merging Thresholds

Learn one token merging threshold per block for a frozen vision transformer (timm) under a CPU latency budget.

- Each block merges every token whose best-match cosine similarity (ToMe bipartite matching on the attention keys) is above its threshold, so the number of merged tokens follows each image.
- The latency of an image is read from a table of measured CPU times per block part and token count.
- The thresholds are trained with parameter-exploring policy gradients (forward passes only, no labels) to keep the output close to the original model, with a latency price updated by dual ascent.
- The trained thresholds are placed exactly on the budget by a common shift, mixing two adjacent shifts per image.

## Install

```
pip install -e .
```

## Usage

```
tmt-train --model deit_small_patch16_224.fb_in1k --train /path/to/imagenet/train --heldout /path/to/heldout --r 12 --out thresholds.json
```

The budget is the table latency of ToMe with the given `r`. The first run measures the latency table on the CPU (one thread, batch size one) and saves it to `latency_table.json`. Pass `--table latency_table.json` to reuse it. Use `--distance cos` for embedding models.

```python
import json, torch
from tmt import ThresholdViT, thresholds_for

vit = ThresholdViT("deit_small_patch16_224.fb_in1k")
placed = json.load(open("thresholds.json"))
placed = dict(low=torch.tensor(placed["low"]), high=torch.tensor(placed["high"]), p=placed["p"])
thresholds = thresholds_for(image_ids, placed)
logits, n_in, k = vit(vit.normalize(images), thresholds)
logits, counts = vit.forward_single(vit.normalize(images[:1]), thresholds[0])
```

## Citation

```
@article{ishibashi2026thresholds,
  title   = {Learning Token Merging Thresholds for Vision Transformers under a CPU Latency Budget},
  author  = {Ishibashi, Ryuto and Meng, Lin},
  journal = {Computers},
  year    = {2026}
}
```
