# NeronRank — Minimal, Reproducible ResNet Pruning Benchmarks (CUDA)

> **Goal**: Provide a clean, minimal codebase to compare **Magnitude (MB)** vs **NeuronRank (NR; TF‑IDF activation scoring)** vs **First‑Order (FO; Taylor |g·w|)** for **MLP (neuron) pruning** on **ResNet** with pre‑trained **Hugging Face** checkpoints.  
> We measure **zero‑shot** and **+1‑epoch fine‑tune (FT) recovery**, log **compute costs**, and plot **Accuracy vs *Parameter Count*** (X axis = *percent of params pruned*, evenly spaced for comparison).  
> We implement **two statistics modes**:  
> 1) **before**: record **module inputs** (default; our existing behavior),  
> 2) **post**: record **post‑activation** activity,  
> 3) **all**: compute both.  
> Use `--statistics {before|post|all}` to switch.

**CUDA required**. Designed to be **as small as possible** but **solid enough** to support a short paper submission (journal/proceedings).

> **Reference:** This implementation is a fresh, clean codebase inspired by our internal NeuronRank work in **#Deep‑CompressionV4**. New code here lives under **`dc2/NeronRank`** in this repository.

---

## TL;DR (What you get)

- **Methods**:  
  - **MB** (Magnitude): per‑neuron L1/RMS of outgoing weights (columns of `fc.weight`).  
  - **NR** (NeuronRank): TF‑IDF activation scoring × weight magnitude:
    \[\mathbf{NR}_j = \lVert W_{\cdot j}\rVert^{\alpha} \cdot \underbrace{\mathrm{TF}_j}_{\text{mean|act|}}^{\ \beta} \cdot \underbrace{\mathrm{IDF}_j=\log\frac{N+1}{\mathrm{DF}_j+1}}_{\text{rarity}}^{\ \gamma}\]
    Defaults: \(\alpha=\beta=\gamma=1\).  
    `--statistics before` uses **inputs to the masked module**; `--statistics post` uses **post‑activation** signals.
  - **FO** (First‑Order / Taylor): per‑weight \(|g\cdot w|\) aggregated to neuron (column‑wise), on a small calibration set (one fwd+bwd pass).

- **Targets**: **MLP (neuron) pruning** of ResNet’s **final classifier input features** (the 2048‑d vector feeding `fc`), plus optional support for late “expansion” MLP inside a bottleneck block.

- **Sparsities**: sweep up to **95%** neuron pruning (we’ll log all; you can choose what to include in the paper).

- **Metrics & outputs**:
  - **CSV** with **zero‑shot** and **+1‑epoch FT** accuracy, **scoring time**, **FT epoch time**, **param counts**, etc.  
  - **Plot**: **Accuracy vs Parameter Count** (Y vs X).  
  - Optional **Spearman ρ** between scores and true Δloss for random neuron ablations (sanity).

- **Pre‑activation vs Post‑activation** statistics: default **before** (inputs to masked module), switchable via CLI.

---

## Environment

- Python ≥ 3.10  
- CUDA‑enabled PyTorch & torchvision  
- Hugging Face: `transformers`, `datasets` (for CIFAR)  
- Optional: `timm` (alternative HF models), `scipy` (for least‑squares “surgery”, if enabled)  
- Plotting: `matplotlib`, `pandas`

**Install (pip):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # choose your CUDA wheel
pip install transformers datasets timm pandas matplotlib scipy tqdm
```

---

## Folder layout (what to create under `dc2/`)

```
dc2/
└─ NeuronRank/
   ├─ README.md                         # this file
   ├─ requirements.txt                  # optional; pin later
   ├─ scripts/
   │   ├─ run_resnet_cifar10.sh         # single-command sanity run (CUDA)
   │   ├─ run_resnet_imagenet.sh        # if you have ImageNet val locally
   │   └─ plot_results.sh               # convenience wrapper for plotting
   └─ neuronrank/
       ├─ __init__.py
       ├─ cli.py                        # main entrypoint (argparse)
       ├─ config.py                     # default config, seeds
       ├─ data.py                       # CIFAR-10 (HF datasets) and ImageNet(val) loaders
       ├─ models/
       │   ├─ __init__.py
       │   └─ resnet_loader.py          # HF Hub -> ResNet (or timm/torchvision fallback)
       ├─ pruning/
       │   ├─ __init__.py
       │   ├─ hooks.py                  # register stats hooks (before/post)
       │   ├─ scoring.py                # MB, NR, FO implementations
       │   ├─ mask.py                   # build/apply prune masks; physical neuron removal
       │   └─ surgery.py                # (optional) least-squares recovery per layer
       ├─ eval/
       │   ├─ imagenet_eval.py          # top-1/top-5 eval
       │   └─ metrics.py                # timers, param count, Spearman rho
       ├─ viz/
       │   └─ plots.py                  # Accuracy vs Params plot
       └─ utils/
           ├─ logging.py                # CSV writer
           └─ seed.py                   # deterministic runs
```

---

## Datasets

- **Default sanity**: **CIFAR‑10** via `datasets` (auto‑download).  
  - **HF model** default: `edadaltocg/resnet18_cifar10` (public).  
  - Switch model via `--hf-model-id`.

- **ImageNet‑1k (val)**: use if you have local path.  
  - Set `--dataset imagenet --imagenet-val <path_to_val>`.  
  - Suggested HF model: `timm/resnet50.a1_in1k` (weights hosted on HF via `timm`).  
  - Or any HF vision model compatible with `AutoModelForImageClassification`.

---

## What we prune (ResNet)

We prune **neurons feeding the final FC**:

- Let `z ∈ R^{C}` be the pooled feature vector before `fc` (C=512/2048 depending on model).  
- The **neuron j** corresponds to the **j‑th input feature** of `fc`; pruning neuron `j` removes **column j** in `fc.weight` (and slices the pooled vector accordingly).

**Statistics modes**:

- `--statistics before` (default): capture **inputs to `fc`** (forward_pre_hook on `fc`) → collect TF/DF per neuron from `x_fc ∈ R^{B×C}`.
- `--statistics post`: capture **post‑activation** before pooling by hooking **`avgpool` (forward_pre_hook)** → you get `U ∈ R^{B×C×H×W}` (post‑ReLU). Compute `v = mean_{H,W}(|U|)` to align with `fc` inputs.
- `--statistics all`: compute both and log both (decisions default to `before`).

This mapping (channel ↔ fc input) is 1:1 in standard ResNet: `avgpool(mean)` over `[B, C, H, W]` → `[B, C]` → `fc`.

---

## CLI

**Primary entrypoint:**
```bash
python -m neuronrank.cli \
  --hf-model-id edadaltocg/resnet18_cifar10 \
  --dataset cifar10 \
  --methods NR,MB,FO \
  --statistics before \

  --sparsities 0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \

  --recover-epochs 1 \
  --calib-size 4096 \
  --batch-size 128 \
  --lr 1e-4 \
  --output-dir runs/resnet18-cifar10 \
  --cuda
```

### Scope options (`--scope`)

- `fc` *(default)* — prune only the classifier input features (per-neuron pruning of the final linear layer).
- `cl` — **layer-level** pruning. Aggregates NR/MB/FO scores per residual block,
  plans which blocks to disable, and rewrites the model to preserve residual
  alignment while dropping entire blocks.
- `all` — structured channel pruning across all residual blocks with per-layer sparsity caps.

**Switch to post‑activation statistics:**
```bash
--statistics post
```

**Compute both (“all”) and compare:**
```bash
--statistics all
```

**Run on ImageNet‑val with ResNet‑50:**
```bash
python -m neuronrank.cli \
  --hf-model-id timm/resnet50.a1_in1k \
  --dataset imagenet \
  --imagenet-val /path/imagenet/val \
  --methods NR,MB,FO \
  --statistics before \

  --sparsities 0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \

  --recover-epochs 1 \
  --calib-size 4096 \
  --batch-size 256 \
  --lr 1e-4 \
  --output-dir runs/resnet50-imagenet \
  --cuda
```

**Key flags (summary):**
- `--hf-model-id` : HF Hub model id (default CIFAR‑10 ResNet18).  
- `--dataset {cifar10|imagenet}` and `--imagenet-val <path>` if ImageNet.  
- `--methods {NR,MB,FO}` comma‑sep.  
- `--statistics {before|post|all}` (default: `before`).  
- `--sparsities` : comma‑sep list of neuron pruning ratios (0–0.99 supported; defaults include fine 0.96–0.99 steps).
- `--calib-size` : #samples for stats (NR/FO) and timers.  
- `--recover-epochs` : epochs for short FT (0 to disable; default 1).  
- `--batch-size`, `--lr`, `--wd` : training knobs.  
- `--cuda` : force CUDA; error if not available.

---

## Implementation details

### 1) Statistics hooks (`pruning/hooks.py`)
- **before**: `forward_pre_hook` on `model.fc` → record input `x_fc ∈ R^{B×C}`; update **TF** (mean|x|) and **DF** (count(|x|>τ)) per neuron; stop at `calib_size` images.
- **post**: `forward_pre_hook` on `model.avgpool` → record feature map `U ∈ R^{B×C×H×W}` (post‑ReLU); compute `v = mean_{H,W}(|U|)`; update TF/DF per neuron.
- **all**: compute both collectors in one pass.

**Threshold τ (for DF)**:  
Per‑neuron robust threshold: `τ_j = 0.5 * median(|activation_j| over calib)` (use EMA or two‑pass).

**Return**: dict with per‑neuron `{TF, DF, N}` for requested mode(s).

### 2) Scoring (`pruning/scoring.py`)
- **MB (Magnitude)**: `mb_j = ||W[:, j]||_1` (L1 per column by default).  
- **NR (NeuronRank)**:
  - `TF_j = mean|act_j|`, `DF_j = count(|act_j|>τ_j)`, `IDF_j = log((N+1)/(DF_j+1))`.  
  - `nr_j = (mb_j**α) * (TF_j**β) * (IDF_j**γ)`; defaults α=β=γ=1.  
  - If `--statistics all`, compute `nr_before`, `nr_post` and log both (decisions default to `before`).  
- **FO (First‑Order / Taylor)**:
  - One **forward+backward** over calibration with CrossEntropy loss.  
  - Accumulate per‑weight `|grad * weight|`, then **aggregate column‑wise** to per‑neuron `fo_j = Σ_i |(∂L/∂W_{i,j}) * W_{i,j}|`.

Each scorer returns `scores`, `score_time_seconds`, `peak_mem_mb` (use `torch.cuda.max_memory_allocated` if CUDA).

### 3) Mask building & application (`pruning/mask.py`)
- For sparsity `s`, **keep top (1−s)** neurons by score; build keep indices `keep_idx` (sorted).  
- **Zeroing**: set masked columns of `fc.weight` to zero for zero‑shot eval.  
- **Structural prune**: physically shrink the classifier by:
  - replacing `fc` with a new `Linear(len(keep_idx), out_features)` and copying weights from kept cols,  
  - introducing a small **Slicer** module that slices the pooled vector `z` to `z[keep_idx]` before `fc` in the forward pass (so earlier convs stay intact).

### 4) Optional “surgery” (`pruning/surgery.py`)
- After pruning to `keep_idx`, recover `fc.weight_S` by least‑squares on calibration:  
  `W_S ← argmin_W || X_S W − Y ||_2^2`, where `X_S` are kept pooled features and `Y` are original logits.  
- Implement with `torch.linalg.lstsq` (optional flag `--do-surgery`).

### 5) Evaluation (`eval/imagenet_eval.py`, `eval/metrics.py`)
- **Zero‑shot**: evaluate immediately after pruning.  
- **+FT (1 epoch)**: if `--recover-epochs > 0`, run one epoch (SGD/AdamW) and re‑evaluate.  
- Measure:
  - `zero_shot_acc_top1`, `zero_shot_eval_time_s`
  - `ft_acc_top1`, `ft_total_time_s`, `ft_epoch_time_avg_s`
  - `kept_params` (connections kept within the pruned scope)
  - `compression_rate` (kept ÷ original connections for that scope)

### 6) CSV logging (`utils/logging.py`)

Row per (method × statistics × sparsity):

| column | description |
|---|---|
| timestamp | ISO time |
| seed | int |
| device | cuda:0 or cpu |
| dataset | cifar10 or imagenet |
| hf_model_id | string |
| layer | scope identifier (e.g., “fc”, “layers”, “all”) |
| method | MB / NR / FO |
| statistics | before / post |
| sparsity | float (e.g., 0.7) |
| kept_params | int (connections remaining in the pruned scope) |
| compression_rate | float (kept_params / original_scope_params) |
| zero_shot_acc_top1 | float |
| zero_shot_eval_time_s | float |
| score_time_s | float |
| score_peak_mem_mb | float |
| ft_epochs | int (0 or 1) |
| ft_epoch_time_avg_s | float |
| ft_total_time_s | float |
| ft_acc_top1 | float |
| notes | free text |

Saved to: `--output-dir/metrics.csv`

### 7) Plots (`viz/plots.py`)
- **Accuracy vs Parameter Count**: one line per method (MB, NR, FO), for the chosen `--statistics` mode.
  - X = evenly spaced `% parameters pruned` tick labels so sparsities align across methods;
    Y = `accuracy` (zero-shot and FT as separate markers or styles).
  - Outputs: `--output-dir/acc_vs_params.png` plus per‑statistics variants (e.g. `acc_vs_params_post.png`).

---

## Example scripts (`scripts/`)

### `scripts/run_resnet_cifar10.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

OUT="runs/resnet18-cifar10"
python -m neuronrank.cli \
  --hf-model-id edadaltocg/resnet18_cifar10 \
  --dataset cifar10 \
  --methods NR,MB,FO \
  --statistics before \

  --sparsities 0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \

  --calib-size 4096 \
  --batch-size 128 \
  --recover-epochs 1 \
  --lr 1e-4 \
  --output-dir "${OUT}" \
  --cuda


# CLI auto-saves plots; rerun manually for custom filters:

python -m neuronrank.viz.plots --csv "${OUT}/metrics.csv" --out "${OUT}/acc_vs_params.png"
```

### `scripts/run_resnet_imagenet.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

IMNET_VAL="/path/to/imagenet/val"
OUT="runs/resnet50-imagenet"

python -m neuronrank.cli \
  --hf-model-id timm/resnet50.a1_in1k \
  --dataset imagenet \
  --imagenet-val "${IMNET_VAL}" \
  --methods NR,MB,FO \
  --statistics before \

  --sparsities 0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \

  --calib-size 4096 \
  --batch-size 256 \
  --recover-epochs 1 \
  --lr 1e-4 \
  --output-dir "${OUT}" \
  --cuda


# CLI auto-saves plots; rerun manually for custom filters:

python -m neuronrank.viz.plots --csv "${OUT}/metrics.csv" --out "${OUT}/acc_vs_params.png"
```

---

## Validation strategy

1) **Sanity (CIFAR‑10, ResNet‑18)**

  - Run MB / NR / FO at sparsities `0.8, 0.9, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99`.

   - Compare **zero‑shot** and **+1 epoch FT** accuracy; ensure CSV is populated.
   - Confirm plots `acc_vs_params.png` and `acc_vs_params_post.png` (if applicable) exist.

2) **Optional ImageNet‑val (ResNet‑50)**  
   - Repeat with `timm/resnet50.a1_in1k` and local val path.

3) **Post‑activation vs Before**  
   - Re‑run with `--statistics post` and `--statistics all`.  
   - Expect post‑activation NR to be equal or slightly better at higher sparsities.

---

## What to report

- **Table**: per method (MB, NR, FO) and sparsity:
  zero‑shot acc, +1‑epoch acc, score_time_s, ft_epoch_time_s, kept_params, compression_rate.
- **Figure**: Accuracy vs Parameter Count (two series: zero‑shot and +FT).

At **high sparsity (≥0.8–0.95)**, **NR should clearly outperform MB**; We keep all rows even if on high sparcity results are not nice.

---

## Notes & guardrails

- **Reproducibility**: fix seeds; fix a calibration subset (store indices).  
- **GPU**: use mixed precision off by default; add AMP later if desired.  
- **Extending to BERT**: identical pattern—hook **post‑GeLU** in FFN for `post`, and **input to W₂** for `before`; prune FFN neurons (columns/rows).

---

## License

MIT (or repository default). Please ensure compatibility with upstream model licenses from Hugging Face Hub.
