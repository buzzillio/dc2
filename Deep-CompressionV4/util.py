import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def collect_nonzero_stats(model):
    """Return aggregate sparsity metrics for a pruned model."""
    nonzero = total = 0
    per_param = []

    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = param.data.detach().cpu().numpy()
        nz_count = int(np.count_nonzero(tensor))
        total_params = int(np.prod(tensor.shape))
        nonzero += nz_count
        total += total_params
        per_param.append({
            'name': name,
            'nonzero': nz_count,
            'total': total_params,
            'sparsity': 0.0 if total_params == 0 else 1.0 - (nz_count / total_params),
            'shape': tuple(int(dim) for dim in tensor.shape),
        })

    if total == 0:
        return {
            'alive': 0,
            'pruned': 0,
            'total': 0,
            'sparsity': 0.0,
            'compression_ratio': 1.0,
            'per_param': per_param,
        }

    sparsity = 1.0 - (nonzero / total)
    compression_ratio = float('inf') if nonzero == 0 else total / nonzero
    return {
        'alive': int(nonzero),
        'pruned': int(total - nonzero),
        'total': int(total),
        'sparsity': sparsity,
        'compression_ratio': compression_ratio,
        'per_param': per_param,
    }


def print_nonzeros(model):
    stats = collect_nonzero_stats(model)
    for entry in stats['per_param']:
        name = entry['name']
        nz_count = entry['nonzero']
        total_params = entry['total']
        sparsity = entry['sparsity']
        print(
            f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * (1 - sparsity):6.2f}%) "
            f"| total_pruned = {total_params - nz_count :7} | shape = {entry['shape']}"
        )
    if stats['total'] > 0:
        print(
            f"alive: {stats['alive']}, pruned : {stats['pruned']}, total: {stats['total']}, "
            f"Compression rate : {stats['compression_ratio']:10.2f}x  ({100 * stats['sparsity']:6.2f}% pruned)"
        )
    else:
        print('Model contains no parameters to evaluate sparsity.')


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy
