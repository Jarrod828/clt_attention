# %%
#pip install datasets

# %%
#%pip install ipywidgets
## If you're using JupyterLab:
#%pip install jupyterlab_widgets
#from tqdm.auto import tqdm   # <- remove this
#from tqdm import tqdm           # <- use this instead (no widgets)

# %%
#pip install "huggingface_hub[hf_xet]"

# %%
#import os
#os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# %%
# === CELL 0: imports, paths, seeding, logging, normalization ===
from tqdm import tqdm           # <- use this instead (no widgets)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import os, json, time, logging, random, math
from pathlib import Path
from dataclasses import dataclass, asdict
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---- YOUR ROOT OUTPUT PATH ----
ROOT = Path(r"C:\Users\jarro\OneDrive\Documents\GitHub\explainable_ai\xai_diff_data\clt_attention_approach")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8'); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)

    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def norm_minmax_np(x, eps=1e-9):
    x = x.astype(np.float64)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin < eps: return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def ensure_dirs(root: Path, dataset: str, method: str, seed: int):
    run_dir = root / dataset / method / f"seed_{seed:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# %%
# === CELL 1: IMDB loader with 70/15/15 split + IDF ===
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("Please install `datasets` package: pip install datasets") from e

def simple_tokenize(text):
    return text.lower().split()

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts; self.labels = labels
        self.vocab = vocab; self.unk = vocab.get("<unk>", 1)
        self.max_len = max_len

    def encode(self, text):
        toks = simple_tokenize(text)
        ids = [self.vocab.get(t, self.unk) for t in toks][:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0]*(self.max_len - len(ids))  # 0 = <pad>
        return np.array(ids, dtype=np.int64)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        y = int(self.labels[idx])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_vocab(train_texts, min_freq=2, max_size=40000):
    from collections import Counter
    cnt = Counter()
    for t in train_texts:
        cnt.update(simple_tokenize(t))
    vocab = {"<pad>":0, "<unk>":1}
    for tok, c in cnt.most_common():
        if c < min_freq: break
        if tok in vocab: continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_size: break
    return vocab

def compute_idf(train_texts, vocab):
    from collections import defaultdict
    N = len(train_texts)
    df = defaultdict(int)
    for t in train_texts:
        toks = set(simple_tokenize(t))
        for tok in toks:
            if tok in vocab:
                df[tok] += 1
    idf = np.zeros(len(vocab), dtype=np.float64)
    for tok, idx in vocab.items():
        if tok in ("<pad>", "<unk>"):
            idf[idx] = 0.0
        else:
            idf[idx] = math.log(N / (1.0 + df.get(tok, 0)))
    idf = norm_minmax_np(idf)  # normalize to [0,1] once
    return idf

def get_dataloaders_70_15_15_IMDB(seed, batch_size=32, max_len=256, num_workers=0):
    set_seed(seed)
    raw = load_dataset("imdb")
    tr_texts = list(raw["train"]["text"]); tr_labels = list(raw["train"]["label"])
    te_texts = list(raw["test"]["text"]);  te_labels = list(raw["test"]["label"])

    # merge train/test then reshuffle for a clean 70/15/15 (simple & consistent with our protocol)
    texts = tr_texts + te_texts
    labels = tr_labels + te_labels
    idx = np.arange(len(texts)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    texts = [texts[i] for i in idx]; labels = [labels[i] for i in idx]

    N = len(texts)
    n_train = int(0.70 * N); n_val = int(0.15 * N)
    train_texts = texts[:n_train];      train_labels = labels[:n_train]
    val_texts   = texts[n_train:n_train+n_val]; val_labels = labels[n_train:n_train+n_val]
    test_texts  = texts[n_train+n_val:];        test_labels = labels[n_train+n_val:]

    vocab = build_vocab(train_texts, min_freq=2, max_size=40000)
    idf_vec = compute_idf(train_texts, vocab)  # fixed scale, derived from train only

    train_ds = TextDataset(train_texts, train_labels, vocab, max_len)
    val_ds   = TextDataset(val_texts,   val_labels,   vocab, max_len)
    test_ds  = TextDataset(test_texts,  test_labels,  vocab, max_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def idf_provider(batch_tokens):  # batch_tokens: [B, Seq] LongTensor
        ids = batch_tokens.cpu().numpy()
        return idf_vec[ids]  # [B, Seq]

    meta = {"vocab_size": len(vocab), "num_classes": 2, "max_len": max_len, "idf_vec": idf_vec}
    return train_dl, val_dl, test_dl, idf_provider, meta


# %%
# === CELL 2: Model (Transformer encoder) with baseline/CLT attention ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):
        S = x.size(1)
        return x + self.pe[:, :S, :]

def attention_entropy(attn_probs: torch.Tensor) -> torch.Tensor:
    p = attn_probs.clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=-1)  # [B, H, S]
    H = H.mean(dim=1)               # [B, S]
    return H

def norm_minmax_t(t):
    t_min = t.amin(dim=-1, keepdim=True)
    t_max = t.amax(dim=-1, keepdim=True)
    denom = (t_max - t_min).clamp_min(1e-8)
    return (t - t_min) / denom

def compute_load(attn_probs, out_probs, idf_seq, weights):
    # attention entropy -> [B,S]
    p = attn_probs.clamp_min(1e-12)
    H_i = (-(p * p.log()).sum(dim=-1)).mean(dim=1)           # [B,S]
    H_i = norm_minmax_t(H_i)

    # margin -> [B,S]
    top2 = torch.topk(out_probs, k=2, dim=-1).values         # [B,2]
    margin = (top2[..., 0] - top2[..., 1]).clamp_min(0.0).unsqueeze(1).expand(-1, attn_probs.shape[2])
    margin = norm_minmax_t(margin)
    load_from_margin = 1.0 - margin

    # idf -> [B,S]
    if idf_seq is None:
        IDF = torch.zeros(attn_probs.shape[0], attn_probs.shape[2], dtype=attn_probs.dtype, device=attn_probs.device)
    else:
        IDF = idf_seq

    L = (weights['attn_entropy'] * H_i
       + weights['margin']       * load_from_margin
       + weights['idf']          * IDF)
    return norm_minmax_t(L)

def map_load_to_budget(L, bmin, bmax):
    return bmin + (bmax - bmin) * L  # [B,S] -> [B,S]

def apply_budget_cap(attn_probs, B):
    # attn_probs: [B,H,S,S], B: [B,S]
    row_sum = attn_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B,H,S,1]
    scale = (B.unsqueeze(1).unsqueeze(-1) / row_sum).clamp(max=1.0)  # [B,1,S,1]
    return attn_probs * scale  # keep grads
    
# --- Grad-safe positional encodings ---
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: [B,S,D]
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)  # NO detach
        
class BudgetedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, mode="baseline",
                 bmin=0.30, bmax=1.00, wE=0.4, wM=0.4, wI=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model; self.n_heads = n_heads; self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.bmin = bmin; self.bmax = bmax
        self.weights = dict(attn_entropy=wE, margin=wM, idf=wI)

    def forward(self, x, out_probs=None, idf_seq=None):
        B, S, D = x.shape
    
        # projections
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,d_k]
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,d_k]
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,d_k]
    
        # scaled dot-product attention
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)          # [B,H,S,S]
        attn_probs  = F.softmax(attn_logits, dim=-1)                            # [B,H,S,S]
    
        # CLT re-weighting (keep grads)
        if self.mode.startswith("CLT"):
            assert out_probs is not None and idf_seq is not None, "CLT mode requires out_probs and idf_seq"
            L   = compute_load(attn_probs, out_probs, idf_seq, self.weights)    # [B,S]
            Bgt = map_load_to_budget(L, self.bmin, self.bmax)                   # [B,S]
            attn_probs = apply_budget_cap(attn_probs, Bgt)                      # grad-preserving
    
        # dropout on probs is fine; keeps grads
        attn_probs = self.dropout(attn_probs)
    
        # attention output
        y = (attn_probs @ v).transpose(1, 2).contiguous().view(B, S, D)
        y = self.o_proj(y)
    
        # return a DETACHED COPY for logging only
        attn_probs_for_log = attn_probs.detach()
        return y, attn_probs_for_log



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff=4, dropout=0.1, mode="baseline",
                 bmin=0.30, bmax=1.00, wE=0.4, wM=0.4, wI=0.2):
        super().__init__()
        self.attn = BudgetedMultiHeadAttention(d_model, n_heads, dropout, mode, bmin,bmax,wE,wM,wI)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_model*dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, out_probs=None, idf_seq=None):
        attn_out, attn_probs = self.attn(x, out_probs=out_probs, idf_seq=idf_seq)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x, attn_probs

class TextClassifierCLT(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, n_heads=4, n_layers=2,
             dropout=0.1, mode="baseline", bmin=0.30, bmax=1.00, wE=0.4, wM=0.4, wI=0.2,
             max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)  # <-- use the new PE
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout, mode=mode,
                             bmin=bmin, bmax=bmax, wE=wE, wM=wM, wI=wI)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, num_classes)
        self.mode = mode

    def forward_with_attention(self, tokens, targets=None, idf_seq=None):
        # embeddings & positional
        x = self.emb(tokens)           # [B,S,D]
        x = self.pos(x)                # grad-safe PE
    
        # provisional probs for CLT (initial)
        pooled = x.mean(dim=1)
        logits = self.cls(self.dropout(pooled))
        probs  = F.softmax(logits, dim=-1)
    
        attn_last = None
        for layer in self.layers:
            x, attn_probs = layer(
                x,
                out_probs=probs,
                idf_seq=idf_seq if self.mode.startswith("CLT") else None
            )
            attn_last = attn_probs  # (already detached copy from the layer)
    
            # refresh provisional probs for next layer
            pooled = x.mean(dim=1)
            logits = self.cls(self.dropout(pooled))
            probs  = F.softmax(logits, dim=-1)
    
        return {"logits": logits, "attn_probs": attn_last, "targets": targets}




# %%
# === CELL 3: metrics & plotting ===
def eval_core_classification(model, loader, device="cpu", idf_provider=None):
    """
    If the model is in CLT mode, we pass idf_seq during eval too.
    """
    model.eval()
    correct = 0; total = 0; loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    use_idf = getattr(model, "mode", "baseline").startswith("CLT") and (idf_provider is not None)

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device); labels = labels.to(device)
            idf_seq = (torch.tensor(idf_provider(tokens), dtype=torch.float32, device=device)
                       if use_idf else None)
            out = model.forward_with_attention(tokens, targets=labels, idf_seq=idf_seq)
            logits = out["logits"]
            loss = ce(logits, labels)
            loss_sum += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    avg_loss = loss_sum / total
    acc = correct / total
    return {"loss": avg_loss, "accuracy": acc}

def save_core_metrics(run_dir: Path, rows: list):
    pd.DataFrame(rows).to_csv(run_dir / "metrics_core.csv", index=False)

def save_alloc_metrics(run_dir: Path, rows: list):
    pd.DataFrame(rows).to_csv(run_dir / "metrics_alloc.csv", index=False)

def plot_budget_histogram(run_dir: Path, B_values, title="Budget histogram"):
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots()
    # histogram with visible bin edges
    n, bins, patches = ax.hist(B_values, bins=30, edgecolor='black', linewidth=0.8)

    # thin lines at each bin edge
    for be in bins:
        ax.axvline(be, linestyle='-', linewidth=0.3, alpha=0.6)

    # integer y-axis ticks
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # data labels on bars
    for count, patch in zip(n, patches):
        if count <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        ax.text(x, y, f"{int(count)}", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("B_i")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(run_dir / "budget_hist.png", dpi=150)
    plt.close(fig)


def rankdata_np(a):
    return pd.Series(a).rank(method='average').to_numpy()

def compute_allocation_metrics(attn_probs, L, B):
    # attn_probs: [B,H,S,S], L,B: [B,S]
    p = attn_probs.clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=-1).mean(dim=1)  # [B,S]
    mean_entropy = float(H.mean().item())

    B_np = B.detach().cpu().numpy()
    mean_B = float(B_np.mean())
    pct_at_min = float((B_np <= (B_np.min()+1e-6)).mean())
    pct_at_max = float((B_np >= (B_np.max()-1e-6)).mean())

    # outgoing attention mass per query token -> [B,S]
    total_mass = attn_probs.sum(dim=-1).mean(dim=1)  # [B,S]

    L_np = L.detach().cpu().numpy().ravel()
    M_np = total_mass.detach().cpu().numpy().ravel()

    # robust "spearman-like" rank correlation (avoid NaNs on constants)
    def _rank(arr):
        return pd.Series(arr).rank(method='average').to_numpy()
    def _safe_corr(a, b):
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return float("nan")
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):  # constant vector => define 0
            return 0.0
        return float(np.corrcoef(_rank(a), _rank(b))[0,1])

    rho = _safe_corr(L_np, M_np)

    return {
        "mean_attention_entropy": mean_entropy,
        "mean_budget": mean_B,
        "pct_budget_at_min": pct_at_min,
        "pct_budget_at_max": pct_at_max,
        "spearman_like_rho_L_vs_mass": rho,
    }



# %%
# === CELL 4: runner for one dataset (IMDB) ===
@dataclass
class RunConfig:
    dataset_name: str
    method_name: str
    seed: int
    bmin: float = 0.30
    bmax: float = 1.00
    w_entropy: float = 0.40
    w_margin: float = 0.40
    w_idf: float = 0.20

def run_all_methods_for_IMDB(
    methods: list,
    seeds: list = [42,43,44,45,46,47,48,49,50,51],
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=5,
    batch_size=32,
    max_len=256,
    quick_test=False,                 # <- NEW: enable quick sanity run
    max_train_batches=500,            # <- NEW: cap train batches/epoch when quick_test
    max_val_batches=100,              # <- NEW
    max_test_batches=100,             # <- NEW (for alloc metrics too)
):
    dataset_name = "IMDB"
    dataset_root = ROOT / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    # if quick test: one seed, one epoch, smaller caps
    if quick_test:
        seeds = [seeds[0]]
        epochs = 1
        max_train_batches = max(50, min(max_train_batches, 200))
        max_val_batches   = max(20, min(max_val_batches,   50))
        max_test_batches  = max(20, min(max_test_batches,  50))
        print(f"[QUICK TEST] device={device} | epochs={epochs} | seeds={seeds} | "
              f"caps: train={max_train_batches} val={max_val_batches} test={max_test_batches}")

    for m in tqdm(methods, desc=f"[{dataset_name}] Methods", position=0, leave=True):
        method_name = m["name"]
        bmin = m.get("bmin", 0.30); bmax = m.get("bmax", 1.00)
        wE  = m.get("w_entropy", 0.40); wM = m.get("w_margin", 0.40); wI = m.get("w_idf", 0.20)

        all_core_rows = []
        all_alloc_rows = []

        for seed in tqdm(seeds, desc=f"[{dataset_name}/{method_name}] Seeds", position=1, leave=False):
            run_dir = ensure_dirs(ROOT, dataset_name, method_name, seed)
            logger = make_logger(run_dir / "log.txt")
            logger.info(f"Starting {dataset_name} / {method_name} / seed {seed}")

            try:
                set_seed(seed)

                # dataloaders (70/15/15)
                train_dl, val_dl, test_dl, idf_provider, meta = get_dataloaders_70_15_15_IMDB(
                    seed, batch_size=batch_size, max_len=max_len, num_workers=0
                )
                vocab_size = meta["vocab_size"]; num_classes = meta["num_classes"]

                mode = "baseline" if method_name=="baseline" else "CLT"
                model = TextClassifierCLT(
                    vocab_size=vocab_size, num_classes=num_classes,
                    d_model=256, n_heads=4, n_layers=2, dropout=0.1,
                    mode=mode, bmin=bmin, bmax=bmax, wE=wE, wM=wM, wI=wI,
                    max_len=max_len
                ).to(device)

                opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()
                
                # ---- ensure autograd ON ----
                torch.set_grad_enabled(True)  # reset global grad mode in case a prior cell disabled it

                # ---- LR scheduler: cosine with warmup ----
                # compute how many train batches per epoch (honor quick_test caps)
                steps_per_epoch = len(train_dl)
                if quick_test:
                    steps_per_epoch = min(steps_per_epoch, max_train_batches)
                
                total_steps  = max(1, steps_per_epoch * epochs)
                warmup_steps = max(10, int(0.05 * total_steps))  # 5% warmup
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))  # linear warmup
                    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))   # cosine decay
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
                global_step = 0


                # ---- preflight sanity check: one batch forward (fast fail) ----
                model.eval()
                with torch.no_grad():
                    btokens, blabels = next(iter(train_dl))
                    btokens = btokens.to(device); blabels = blabels.to(device)
                    idf_seq = (torch.tensor(idf_provider(btokens), dtype=torch.float32, device=device)
                               if mode == "CLT" else None)
                    _out = model.forward_with_attention(btokens, targets=blabels, idf_seq=idf_seq)
                    _ = _out["logits"]; _ = _out["attn_probs"]
                    logger.info("Preflight OK: forward_with_attention returned logits & attn_probs.")

                best_val = float('inf'); best_state = None
                best_epoch = 0
                no_improve = 0
                patience = 5           # <--- choose your patience (3–8 is common)
                max_epochs = epochs    # keep original request for logging
                t0 = time.time()

                # ---- training ----
                for ep in tqdm(range(1, epochs+1), desc=f"[{dataset_name}/{method_name}/seed{seed}] Epochs", position=2, leave=False):
                    model.train()
                    train_losses = []
                    for b_idx, (tokens, labels) in enumerate(train_dl, start=1):
                        tokens = tokens.to(device); labels = labels.to(device)
                        idf_seq = (torch.tensor(idf_provider(tokens), dtype=torch.float32, device=device)
                                   if mode=="CLT" else None)
                        # always compute train forward/backward with grad enabled
                        with torch.enable_grad():
                            out = model.forward_with_attention(tokens, targets=labels, idf_seq=idf_seq)
                            logits = out["logits"]
                            loss = criterion(logits, labels)
                        
                            # optional diag (remove once stable)
                            # assert logits.requires_grad, "Logits lost grad — unexpected no_grad around training forward."
                        
                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # if you added clipping
                            opt.step()
                            # if you added a scheduler:
                            scheduler.step()
                        train_losses.append(loss.item())
                        if quick_test and b_idx >= max_train_batches:
                            break

                    # ---- validation (NO grad; NO optimizer steps) ----
                    model.eval()
                    val_loss = 0.0
                    val_total = 0
                    with torch.no_grad():
                        for b_idx, (tokens, labels) in enumerate(val_dl, start=1):
                            tokens = tokens.to(device); labels = labels.to(device)
                            idf_seq = (torch.tensor(idf_provider(tokens), dtype=torch.float32, device=device)
                                       if mode == "CLT" else None)
                    
                            out = model.forward_with_attention(tokens, targets=labels, idf_seq=idf_seq)
                            logits = out["logits"]
                            loss = criterion(logits, labels)
                    
                            # accumulate (sum) to average later
                            val_loss += loss.item() * labels.size(0)
                            val_total += labels.size(0)
                    
                            if quick_test and b_idx >= max_val_batches:
                                break
                    
                    # finalize val loss
                    val_loss = val_loss / max(1, val_total)
                    
                    logger.info(f"Epoch {ep}/{epochs} | train_loss={np.mean(train_losses):.4f} | val_loss={val_loss:.4f}")


                    if val_loss < best_val - 1e-6:  # tiny margin to avoid churn
                        best_val = val_loss
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        best_epoch = ep
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    # early stop?
                    if no_improve >= patience:
                        logger.info(f"Early stopping at epoch {ep} (best @ epoch {best_epoch}, val_loss={best_val:.4f})")
                        break

                train_time = time.time() - t0
                torch.save(best_state, run_dir / "best_model_state.pt")

                # ---- test (pass idf_provider so CLT gets idf_seq) ----
                model.load_state_dict({k: v.to(device) for k,v in best_state.items()})
                core = eval_core_classification(model, test_dl, device=device, idf_provider=idf_provider)
                core["epochs_run"] = best_epoch
                core["val_loss_best"] = float(best_val)
                core["seed"] = seed; core["train_time_sec"] = train_time
                pd.DataFrame([core]).to_csv(run_dir / "metrics_core.csv", index=False)
                all_core_rows.append(core)
                logger.info(f"TEST | acc={core['accuracy']:.4f} | loss={core['loss']:.4f} | time={train_time:.1f}s")

                # ---- allocation metrics (sample first N batches) ----
                alloc_accum = []
                with torch.no_grad():
                    for b_idx, (tokens, labels) in enumerate(test_dl, start=1):
                        if quick_test and b_idx > max_test_batches: break
                        tokens = tokens.to(device); labels = labels.to(device)
                        idf_seq = (torch.tensor(idf_provider(tokens), dtype=torch.float32, device=device)
                                   if mode=="CLT" else None)
                        out = model.forward_with_attention(tokens, targets=labels, idf_seq=idf_seq)
                        logits = out["logits"]
                        probs = F.softmax(logits, dim=-1)
                        attn_probs = out["attn_probs"]
                        if mode == "CLT":
                            L = compute_load(attn_probs, probs, idf_seq, {'attn_entropy': wE, 'margin': wM, 'idf': wI})
                        else:
                            # make a zero-IDF tensor of shape [B,S] for baseline
                            Bsize, _, S, _ = attn_probs.shape
                            idf_zero = torch.zeros(Bsize, S, dtype=attn_probs.dtype, device=attn_probs.device)
                            L = compute_load(attn_probs, probs, idf_zero, {'attn_entropy': 1.0, 'margin': 0.0, 'idf': 0.0})
                        B = map_load_to_budget(L, bmin, bmax)
                        # sanity: avoid NaNs
                        assert not torch.isnan(L).any(), "NaN in load L"
                        assert not torch.isnan(B).any(), "NaN in budget B"
                        m_alloc = compute_allocation_metrics(attn_probs, L, B)
                        alloc_accum.append(m_alloc)

                if alloc_accum:
                    alloc_mean = {k: float(np.mean([d[k] for d in alloc_accum])) for k in alloc_accum[0]}
                else:
                    alloc_mean = {}
                alloc_mean["seed"] = seed
                pd.DataFrame([alloc_mean]).to_csv(run_dir / "metrics_alloc.csv", index=False)
                all_alloc_rows.append(alloc_mean)   # <-- add this line

                # ---- quick budget histogram ----
                if alloc_accum:
                    # reuse last B from the loop
                    B_values = B.detach().cpu().numpy().ravel()
                    plot_budget_histogram(run_dir, B_values, title=f"{dataset_name}/{method_name} seed {seed} budget")
                else:
                    logger.warning("Skipping budget histogram for this seed (no allocation batches collected).")

            except Exception as e:
                logger.exception(f"Seed {seed} FAILED with error: {e}")
                (run_dir / "_FAILED.txt").write_text(str(e))
                continue

        # averaged summary (skip if all seeds failed)
        if not all_core_rows:
            print(f"[{dataset_name}/{method_name}] No successful seeds; skipping averages.")
            continue

        avg_dir = dataset_root / method_name / "averaged"
        avg_dir.mkdir(parents=True, exist_ok=True)
        df_core = pd.DataFrame(all_core_rows); df_core.to_csv(avg_dir / "tables_core.csv", index=False)

        df_alloc = pd.DataFrame(all_alloc_rows) if all_alloc_rows else pd.DataFrame()
        if not df_alloc.empty:
            df_alloc.to_csv(avg_dir / "tables_alloc.csv", index=False)

        core_summary = {col: {"mean": float(df_core[col].mean()), "std": float(df_core[col].std(ddof=1))}
                        for col in df_core.columns if col not in ["seed"]}
        alloc_summary = ({col: {"mean": float(df_alloc[col].mean()), "std": float(df_alloc[col].std(ddof=1))}
                          for col in df_alloc.columns if col not in ["seed"]} if not df_alloc.empty else {})

        dump_json({"core": core_summary, "allocation": alloc_summary}, avg_dir / "summary_mean_std.json")
        print(f"[{dataset_name}/{method_name}] averaged summary written to {avg_dir}")



# %%
# === CELL 5: define variants & run IMDB (full) ===
methods_to_run = [
    {"name": "baseline"},
    {"name": "CLT-B030-E40M40I20", "bmin": 0.30, "bmax": 1.00, "w_entropy": 0.40, "w_margin": 0.40, "w_idf": 0.20},
    {"name": "CLT-B050-E40M40I20", "bmin": 0.50, "bmax": 1.00, "w_entropy": 0.40, "w_margin": 0.40, "w_idf": 0.20},
    {"name": "CLT-B030-E50M50I00", "bmin": 0.30, "bmax": 1.00, "w_entropy": 0.50, "w_margin": 0.50, "w_idf": 0.00},
    {"name": "CLT-B050-E50M50I00", "bmin": 0.50, "bmax": 1.00, "w_entropy": 0.50, "w_margin": 0.50, "w_idf": 0.00},
    {"name": "CLT-E",              "bmin": 0.30, "bmax": 1.00, "w_entropy": 1.00, "w_margin": 0.00, "w_idf": 0.00},
    {"name": "CLT-E-tight",        "bmin": 0.50, "bmax": 1.00, "w_entropy": 1.00, "w_margin": 0.00, "w_idf": 0.00},
]

run_all_methods_for_IMDB(
    methods=methods_to_run,
    seeds=[42,43,44,45,46,47,48,49,50,51],  # 10 seeds like your prior protocol
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,              # start with 10; early stopping is active
    batch_size=32,
    max_len=256,
    quick_test=False        # FULL run
)


# methods_to_run = [
#     {"name": "baseline"},
#     {"name": "CLT-B030-E40M40I20", "bmin": 0.30, "bmax": 1.00, "w_entropy": 0.40, "w_margin": 0.40, "w_idf": 0.20},
#     {"name": "CLT-B050-E40M40I20", "bmin": 0.50, "bmax": 1.00, "w_entropy": 0.40, "w_margin": 0.40, "w_idf": 0.20},
#     {"name": "CLT-E-tight",        "bmin": 0.50, "bmax": 1.00, "w_entropy": 1.00, "w_margin": 0.00, "w_idf": 0.00},
#     # run baseline only first in quick mode; add CLT after baseline looks sane
# ]

# run_all_methods_for_IMDB(
#     methods=methods_to_run,
#     seeds=[42],
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     epochs=1,
#     batch_size=32,
#     max_len=256,
#     quick_test=True,          # <- turn on
#     max_train_batches=100,    # <- caps per epoch
#     max_val_batches=30,
#     max_test_batches=30,
# )



# %%


# %%


# %%


# %%



