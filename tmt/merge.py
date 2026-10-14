import torch
import torch.nn.functional as F
import timm


class ThresholdViT(torch.nn.Module):
    def __init__(self, name, pretrained=True, **kw):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, **kw).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.data_config = cfg

    @property
    def depth(self):
        return len(self.model.blocks)

    @property
    def tokens(self):
        return self.model.patch_embed.num_patches + 1

    def normalize(self, img):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return (img - self.mean) / self.std

    def embed(self, img):
        m = self.model
        x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(img))))
        return x, torch.ones(x.shape[:2], device=x.device)

    def head(self, x, s):
        m = self.model
        if m.global_pool != "avg":
            return m.forward_head(m.norm(x))
        x = m.norm(x)
        w = s[:, 1:].to(x.dtype)
        p = (x[:, 1:] * w[..., None]).sum(1) / w.sum(1, keepdim=True)
        return m.head(m.head_drop(m.fc_norm(p)))

    @torch.no_grad()
    def forward(self, img, thresholds=None, counts=None, hook=None):
        x, s = self.embed(img)
        B, T = x.shape[:2]
        n = torch.full((B,), T, dtype=torch.long, device=x.device)
        th = None if thresholds is None else thresholds.to(x.device).float().expand(B, -1)
        n_in, ks = [], []
        for l, blk in enumerate(self.model.blocks):
            x, metric = attention(blk, x, s)
            sc = scores(metric, n)
            best = sc.max(-1).values
            if hook is not None:
                hook(l, best)
            k = torch.full_like(n, counts[l]) if th is None else (best > th[:, l:l + 1]).sum(-1)
            k = torch.minimum(k, (n - 1) // 2)
            x, s = merge(x, s, assign(sc, n, k, T), T)
            n_in.append(n)
            ks.append(k)
            n = n - k
            x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        return self.head(x, s), torch.stack(n_in, 1), torch.stack(ks, 1)

    @torch.no_grad()
    def forward_single(self, img, thresholds):
        x, s = self.embed(img)
        ks = []
        for l, blk in enumerate(self.model.blocks):
            x, metric = attention(blk, x, s)
            x, s, k = merge_single(x, s, metric, lambda best: int((best > float(thresholds[l])).sum()))
            ks.append(k)
            x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        return self.head(x, s), ks

def attention(blk, x, s):
    a = blk.attn
    h = blk.norm1(x)
    B, T, C = h.shape
    q, k, v = a.qkv(h).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    logits = (q * a.scale) @ k.transpose(-2, -1) + s.log()[:, None, None, :].to(q.dtype)
    out = (logits.float().softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, T, C)
    return x + blk.ls1(a.proj(a.norm(out))), k.mean(1)


def scores(metric, n):
    m = metric.float()
    m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sc = m[:, ::2] @ m[:, 1::2].transpose(1, 2)
    T = metric.shape[1]
    dead_a = torch.arange(0, T, 2, device=m.device)[None] >= n[:, None]
    dead_b = torch.arange(1, T, 2, device=m.device)[None] >= n[:, None]
    sc = sc.masked_fill(dead_a[:, :, None] | dead_b[:, None, :], float("-inf"))
    sc[:, 0, :] = float("-inf")
    return sc


def assign(sc, n, k, T):
    B, dev = sc.shape[0], sc.device
    best, partner = sc.max(-1)
    order = best.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(order.shape[1], device=dev).expand(B, -1))
    pos_a = torch.arange(0, T, 2, device=dev)[None]
    pos_b = torch.arange(1, T, 2, device=dev)[None]
    src = rank < k[:, None]
    keep = (pos_a < n[:, None]) & ~src
    kept = keep.sum(1, keepdim=True)
    out = (n - k)[:, None]
    slot_a = torch.where(keep, torch.cumsum(keep.long(), 1) - 1, out + pos_a - n[:, None])
    slot_a = torch.where(src, kept + partner, slot_a)
    slot_b = torch.where(pos_b < n[:, None], kept + torch.arange(pos_b.shape[1], device=dev)[None], out + pos_b - n[:, None])
    slots = torch.empty(B, T, dtype=torch.long, device=dev)
    slots[:, 0::2], slots[:, 1::2] = slot_a, slot_b
    return torch.where((k == 0)[:, None], torch.arange(T, device=dev).expand(B, -1), slots)


def merge(x, s, slots, n_out):
    with torch.autocast(device_type=x.device.type, enabled=False):
        P = F.one_hot(slots, n_out).transpose(1, 2).float()
        s_new = (P @ s.float().unsqueeze(-1))[..., 0]
        x_sum = P @ (x.float() * s.float().unsqueeze(-1))
        x_new = torch.where(s_new[..., None] > 0, x_sum / s_new.clamp_min(1e-12)[..., None], torch.zeros_like(x_sum))
    return x_new, s_new


def merge_single(x, s, metric, count):
    T = x.shape[1]
    if T < 3:
        return x, s, 0
    best, partner = scores(metric, torch.tensor([T], device=x.device)).max(-1)
    k = int(min(count(best), (T - 1) // 2))
    if k <= 0:
        return x, s, 0
    order = best.argsort(dim=-1, descending=True, stable=True)
    keep, src = order[0, k:].sort()[0], order[0, :k]
    dst = partner[0, src]
    xs = x[0] * s[0, :, None]
    a_x, b_x = xs[::2], xs[1::2].clone()
    a_s, b_s = s[0, ::2], s[0, 1::2].clone()
    b_x.index_add_(0, dst, a_x[src])
    b_s.index_add_(0, dst, a_s[src])
    s_new = torch.cat([a_s[keep], b_s])
    return (torch.cat([a_x[keep], b_x]) / s_new[:, None])[None], s_new[None], k


def tome_counts(tokens, depth, r):
    n, ks = tokens, []
    for _ in range(depth):
        k = min(r, (n - 1) // 2)
        ks.append(k)
        n -= k
    return ks
