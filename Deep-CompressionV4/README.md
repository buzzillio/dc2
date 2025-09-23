# Deep-Compression-PyTorch
PyTorch implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding'  by Song Han, Huizi Mao, William J. Dally

This implementation implements three core methods in the paper - Deep Compression
- Pruning
- Weight sharing
- Huffman Encoding

## Requirements
Following packages are required for this project
- Python3.6+
- tqdm
- numpy
- pytorch, torchvision
- scipy
- scikit-learn
- matplotlib (for benchmarking visualisations)
- transformers (for GPT-2 fine-tuning)
- datasets (WikiText-2 loading)

or just use docker
``` bash
$ docker pull tonyapplekim/deepcompressionpytorch
```

## Usage
### Pruning
``` bash
# Train and prune in a single pass
$ python pruning.py --model lenet --pruning-method neuronrank
```
This command
- trains LeNet-300-100 model with MNIST dataset
- prunes weight values that has low absolute value
- retrains the model with MNIST dataset
- prints out non-zero statistics for each weights in the layer

You can split the workflow in two steps to reuse trained checkpoints:

```bash
# Train only (saves checkpoint and, optionally, NeuronRank stats)
python pruning.py --mode train --model vgg --epochs 100 \
  --output-checkpoint checkpoints/vgg_e100.pt \
  --save-activation-stats stats/vgg_e100.pt

# Prune/retrain from the saved checkpoint
python pruning.py --mode prune --model vgg --checkpoint checkpoints/vgg_e100.pt \
  --activation-stats stats/vgg_e100.pt --target-sparsity 0.9 --retrain-epochs 100
```

You can control other values such as
- random seed
- epochs
- sensitivity
- batch size
- learning rate
- and others
For more, type `python pruning.py --help`

### Automated benchmarking

Use `benchmark.py` to compare standard magnitude pruning with our NeuronRank (TF-IDF inspired)
scoring. The helper script trains each milestone once, caches the checkpoint and
activation statistics, then reuses them across all sparsity targets and pruning
methods—so even wide sweeps finish much faster.

Example (LeNet, default milestones `50 100` and sparsity targets `0.5 0.8 0.9`):

```bash
python benchmark.py --model lenet --device cuda
```

For VGG you can run:

```bash
python benchmark.py --model vgg --device cuda --workers 8 --sparsity-targets 0.5 0.75 0.9
```

Outputs are written to `benchmark_outputs/` (configurable via `--output-dir`) and
include a JSONL file with raw metrics, a CSV export, and PNG plots showing
accuracy vs. compression for each epoch milestone.

### SqueezeNet + CIFAR-10

SqueezeNet plugs into the same CIFAR-10 pipeline as VGG. Install the dependencies listed
above (or via pip):

```bash
pip install torch torchvision tqdm numpy matplotlib
```

Run a one-shot prune using the baked-in CIFAR-10 defaults (SGD + momentum, augmentation, etc.):

```bash
python pruning.py --model squeezenet --mode full --epochs 100 --target-sparsity 0.9
```

To sweep sparsities and seeds automatically:

```bash
python benchmark.py --model squeezenet --epochs 50 100 --device cuda
```

The benchmark helper forwards additional arguments (e.g. `--pruning-args`) to `pruning.py` so you
can reuse NeuronRank tuning without rewriting commands.

### GPT-2 + NeuronRank on WikiText-2

Phase 1-3 add a full GPT-2 pipeline: fine-tuning, pruning, and benchmarking.

*Fine-tuning*

```bash
# Full fine-tune (3 epochs) saving checkpoints and metrics
python gpt2_finetune.py --epochs 3 --output-dir checkpoints/gpt2_wikitext2

# Quick smoke test (~minutes on CPU)
python gpt2_finetune.py --epochs 1 --gpt2-max-eval-batches 5 --output-dir tmp/gpt2_smoke
```

*Pruning with NeuronRank*

```bash
# Train + prune in one pass (collects activation stats automatically)
python pruning.py --model gpt2 --mode full --epochs 3 --target-sparsity 0.9 \
  --gpt2-model-name gpt2 --gpt2-max-eval-batches 50 --log-interval 20

# Two-step workflow if you want to reuse checkpoints/stats
python pruning.py --model gpt2 --mode train --epochs 3 \
  --output-checkpoint checkpoints/gpt2_e3.pt \
  --save-activation-stats stats/gpt2_e3.pt

python pruning.py --model gpt2 --mode prune --epochs 3 \
  --checkpoint checkpoints/gpt2_e3.pt --activation-stats stats/gpt2_e3.pt \
  --target-sparsity 0.9 --gpt2-max-eval-batches 50 --retrain-epochs 3
```

Perplexity is reported after fine-tuning, after prune, and after the retrain phase. Expect
~35–40 perplexity after a short 3-epoch CPU run (lower with GPUs or longer training).

*Benchmarking*

```bash
# Compare std vs NeuronRank at multiple sparsities (per seed / milestone)
python benchmark.py --model gpt2 --epochs 1 2 3 --sparsity-targets 0.5 0.8 0.9 \
  --gpt2-max-eval-batches 20 --output-dir benchmark_outputs/gpt2
```

Plots now show perplexity vs compression when `--model gpt2` is selected and summaries
print the same metric. Use `--gpt2-max-eval-batches` to keep turnaround short when
experimenting on CPU or limited hardware.

#### Gradient-aware NeuronRank for GPT-style models

NeuronRank now collects activation *and* gradient statistics when `--neuronrank-gradients`
is set to `on` (the default `auto` mode enables it automatically for GPT-2/NanoGPT).
Gradient magnitudes act as a contextual rarity signal—rare-but-impactful neurons receive
larger scores—so pruning focuses on redundant units instead of those steering perplexity.

Key knobs:

- `--neuronrank-grad-threshold`: sets the minimum gradient magnitude counted in the
  gradient document frequency (defaults to `1e-3`).
- `--neuronrank-grad-mix`: blends the classic TF-IDF scores with the new gradient-aware
  component. `0` disables gradients, `1` trusts them fully (default `0.75`).
- `--neuronrank-grad-tf-power`, `--neuronrank-grad-idf-power`, and
  `--neuronrank-grad-power`: shape the influence of the gradient activation, specificity,
  and blended score respectively.

These additions significantly tighten perplexity after pruning on language models while
retaining backwards compatibility—existing activation stats keep working, and CNN workflows
stay untouched unless you opt in.

### Weight sharing
``` bash
$ python weight_share.py saves/model_after_retraining.ptmodel
```
This command
* Applies K-means clustering algorithm for the data portion of CSC or CSR matrix representation for each weight
* Then, every non-zero weight is now clustered into (2**bits) groups.
(Default is 32 groups - using 5 bits)
- This modified model is saved to
`saves/model_after_weight_sharing.ptmodel`

### Huffman coding
``` bash
$ python huffman_encode.py saves/model_after_weight_sharing.ptmodel
```
This command
- Applies Huffman coding algorithm for each of the weights in the network
- Saves each weight to `encodings/` folder
- Prints statistics for improvement



## Note
Note that I didn’t apply pruning nor weight sharing nor Huffman coding  for bias values. Maybe it’s better if I apply those to the biases as well, I haven’t try this out yet.

Note that this work was done when I was employed at http://nota.ai
