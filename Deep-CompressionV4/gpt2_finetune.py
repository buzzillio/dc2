"""Fine-tune masked GPT-2 on WikiText-2 with checkpointing and perplexity reporting."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from gpt import MaskedGPT2LMHeadModel, build_wikitext2_dataloaders


@dataclass
class TrainState:
    epoch: int
    global_step: int
    best_eval_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on WikiText-2 with pruning-ready masks.')
    parser.add_argument('--model-name-or-path', type=str, default='gpt2',
                        help='Hugging Face model identifier or checkpoint path (default: gpt2).')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Optional cache directory for datasets and models.')
    parser.add_argument('--output-dir', type=str, default='checkpoints/gpt2_wikitext2',
                        help='Directory to store checkpoints and logs.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size per step.')
    parser.add_argument('--eval-batch-size', type=int, default=None, help='Evaluation batch size (defaults to train batch size).')
    parser.add_argument('--block-size', type=int, default=1024, help='Sequence length for language modelling blocks.')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='AdamW learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='AdamW weight decay.')
    parser.add_argument('--warmup-steps', type=int, default=0, help='Linear warmup steps.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Steps to accumulate gradients before optimizer.step().')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping value (0 to disable).')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Evaluate every N optimizer steps (defaults to once per epoch).')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Save checkpoint every N optimizer steps (defaults to once per epoch).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                        help='Force device (auto-detected when omitted).')
    parser.add_argument('--max-train-steps', type=int, default=None,
                        help='Limit total optimizer steps (overrides epochs when set).')
    parser.add_argument('--log-interval', type=int, default=10, help='Training progress log interval (in optimizer steps).')
    parser.add_argument('--metrics-file', type=str, default=None,
                        help='Optional JSON lines file to append training/eval metrics.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from a previous checkpoint directory.')
    parser.add_argument('--eval-only', action='store_true', help='Skip training and only run evaluation.')
    parser.add_argument('--max-eval-batches', type=int, default=None,
                        help='Limit evaluation batches for quick smoke tests.')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(choice: Optional[str] = None) -> torch.device:
    if choice == 'cpu':
        return torch.device('cpu')
    if choice == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if choice == 'mps':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(output_dir: str, model: MaskedGPT2LMHeadModel, tokenizer: AutoTokenizer,
                    optimizer: torch.optim.Optimizer, scheduler, state: TrainState) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'train_state': asdict(state),
    }, os.path.join(output_dir, 'trainer_state.pt'))


def load_trainer_state(path: str):
    trainer_path = os.path.join(path, 'trainer_state.pt')
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f'No trainer_state.pt found in {path}')
    return torch.load(trainer_path, map_location='cpu')


def evaluate(model: MaskedGPT2LMHeadModel, dataloader: DataLoader, device: torch.device,
             max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    total_tokens = 0
    loss_sum = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            token_count = inputs['input_ids'].numel()
            loss_sum += loss.item() * token_count
            total_tokens += token_count
    mean_loss = loss_sum / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    return {'eval_loss': mean_loss, 'eval_perplexity': ppl}


def append_metrics(path: Optional[str], payload: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload) + '\n')


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = auto_device(args.device)
    print(f'Using device: {device}')

    train_loader, eval_loader, tokenizer = build_wikitext2_dataloaders(
        tokenizer_name=args.model_name_or_path,
        cache_dir=args.cache_dir,
        block_size=args.block_size,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        shuffle=not args.eval_only,
    )

    model_source = args.resume_from or args.model_name_or_path
    model = MaskedGPT2LMHeadModel.from_pretrained(model_source, cache_dir=args.cache_dir)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = args.max_train_steps
    if total_steps is None:
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_steps = args.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    state = TrainState(epoch=0, global_step=0, best_eval_loss=float('inf'))
    if args.resume_from:
        print(f'Resuming from {args.resume_from}')
        checkpoint = load_trainer_state(args.resume_from)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        state = TrainState(**checkpoint['train_state'])
        if state.global_step >= total_steps:
            total_steps = state.global_step

    if args.eval_only:
        metrics = evaluate(model, eval_loader, device, args.max_eval_batches)
        print(f"Eval loss: {metrics['eval_loss']:.4f}, perplexity: {metrics['eval_perplexity']:.2f}")
        append_metrics(args.metrics_file, {**metrics, 'mode': 'eval'})
        return

    progress = tqdm(total=total_steps, desc='Training steps', leave=True, initial=state.global_step)
    optimizer.zero_grad()

    while state.global_step < total_steps:
        for step, batch in enumerate(train_loader):
            model.train()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss_value = outputs.loss.detach().item()
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            should_step = ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_loader))

            if should_step:
                if args.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state.global_step += 1
                progress.update(1)

                if args.log_interval and state.global_step % args.log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    effective_loss = loss_value
                    print(f'Step {state.global_step}/{total_steps}: loss={effective_loss:.4f} lr={current_lr:.2e}')
                    append_metrics(args.metrics_file, {
                        'mode': 'train_step',
                        'step': state.global_step,
                        'loss': effective_loss,
                        'lr': current_lr,
                    })

                if args.eval_interval and state.global_step % args.eval_interval == 0:
                    metrics = evaluate(model, eval_loader, device, args.max_eval_batches)
                    append_metrics(args.metrics_file, {**metrics, 'mode': 'eval', 'step': state.global_step})
                    print(
                        f"Eval @ step {state.global_step}: loss={metrics['eval_loss']:.4f}, "
                        f"perplexity={metrics['eval_perplexity']:.2f}"
                    )
                    if metrics['eval_loss'] < state.best_eval_loss:
                        state.best_eval_loss = metrics['eval_loss']
                        save_checkpoint(os.path.join(args.output_dir, 'best'), model, tokenizer, optimizer, scheduler, state)

                if args.save_interval and state.global_step % args.save_interval == 0:
                    save_checkpoint(os.path.join(args.output_dir, f'step_{state.global_step}'),
                                    model, tokenizer, optimizer, scheduler, state)

                if state.global_step >= total_steps:
                    break
        state.epoch += 1
        if args.eval_interval is None:
            metrics = evaluate(model, eval_loader, device, args.max_eval_batches)
            append_metrics(args.metrics_file, {**metrics, 'mode': 'eval', 'epoch': state.epoch})
            print(
                f"Epoch {state.epoch} eval: loss={metrics['eval_loss']:.4f}, "
                f"perplexity={metrics['eval_perplexity']:.2f}"
            )
            if metrics['eval_loss'] < state.best_eval_loss:
                state.best_eval_loss = metrics['eval_loss']
                save_checkpoint(os.path.join(args.output_dir, 'best'),
                                model, tokenizer, optimizer, scheduler, state)
        if args.save_interval is None:
            save_checkpoint(os.path.join(args.output_dir, f'epoch_{state.epoch}'),
                            model, tokenizer, optimizer, scheduler, state)

    progress.close()

    final_metrics = evaluate(model, eval_loader, device, args.max_eval_batches)
    append_metrics(args.metrics_file, {**final_metrics, 'mode': 'final'})
    save_checkpoint(os.path.join(args.output_dir, 'final'), model, tokenizer, optimizer, scheduler, state)
    print(f"Training complete. Final loss={final_metrics['eval_loss']:.4f}, "
          f"perplexity={final_metrics['eval_perplexity']:.2f}")


if __name__ == '__main__':
    arguments = parse_args()
    train(arguments)
