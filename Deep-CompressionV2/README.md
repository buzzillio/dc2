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

or just use docker
``` bash
$ docker pull tonyapplekim/deepcompressionpytorch
```

## Usage
### Pruning
``` bash
$ python pruning.py
```
This command
- trains LeNet-300-100 model with MNIST dataset
- prunes weight values that has low absolute value
- retrains the model with MNIST dataset
- prints out non-zero statistics for each weights in the layer

You can control other values such as
- random seed
- epochs
- sensitivity
- batch size
- learning rate
- and others
For more, type `python pruning.py --help`

### Automated benchmarking

Use `benchmark.py` to compare standard magnitude pruning with the TF-IDF inspired
ranking. The helper script trains models at a series of epoch milestones,
applies both pruning rules at matched sparsity levels, and plots the final
accuracy against the achieved compression ratios.

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
