'''
核心训练代码
'''
import os
import time

import argparse
import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from modules import (get_batch, TransformerLM, AdamW, SGD, cross_entropy_loss, gradient_clipping, save_checkpoint, \
                     load_checkpoint, cosine_annealing)

def cal_grad_norm(parameters):
    """
    Args:
        parameters:
    Returns:
        L2 grad_norm: float
    """
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    s = sum(torch.sum(torch.pow(grad, 2)).item() for grad in grads) # 每个grad都是矩阵，先内部求和。
    norm = s ** 0.5
    return norm

@torch.no_grad()
def run_validation(model, dataset, batch_size, context_length, device, num_batches=10):
    total_loss = 0.0
    model.eval()
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / num_batches
    return avg_loss

def main(args):
    if args.use_wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)

    # 1. 选择device和生成权重保存路径
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 使用np.load(..., mmap_mode='r')加载数据集
    print(f"Loading training data from {args.train_data_path}...")
    train_dataset = np.load(args.train_data_path, mmap_mode='r')
    print(f"Loading validation data from {args.val_data_path}...")
    val_dataset = np.load(args.val_data_path, mmap_mode='r')

    # 3. 初始化模型
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # 4.初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. 加载检查点
    start_step = 0
    if args.load_from_checkpoint and args.checkpoint_path:
        try:
            start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
            print(f"Resuming training from step {start_step}")
        except FileNotFoundError:
            print("Failed to load checkpoint. Starting training from scratch.")

    # 6. 开始训练
    print("Starting training...")
    model.train()

    start_time = t0 = time.time()
    train_tokens = args.batch_size * args.context_length

    for step in range(start_step, args.max_steps):
        # 6.1 使用cosine_annealing确定学习率
        lr = cosine_annealing(
            step,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_steps,
            args.cosine_cycle_iters
        )

        # 6.2 修改param_groups“操作手册”中的学习率；这里的optimizer.param_groups是一个列表，元素为字典
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 6.3 获取batch数据
        inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, device)

        # 6.4 前向传播
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        # 6.5 反向传播
        optimizer.zero_grad()
        loss.backward() # 计算梯度
        # 6.5.1 计算grad_norm，用于检查是否出现梯度爆炸的问题
        grad_norm = cal_grad_norm(model.parameters())

        # 6.6 裁剪梯度; model.parameters() -> Tensor: weight, bias, -> Tensor: grad
        gradient_clipping(model.parameters(), args.grad_clip)
        # 6.7 更新参数
        optimizer.step()

        # 7. 日志记录
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if step % args.log_interval == 0:
            print(f"Step {step}: loss {loss.item():.4f}, lr {lr:.2e}, time {dt:.2f}s")
            throughput = train_tokens / dt
            if args.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/step": step,
                    "train/token_per_sec": throughput,
                    "train/time_elapsed": time.time() - start_time,
                    "train/grad_norm": grad_norm,
                })

        # 8. 验证中间结果
        if step > 0 and step % args.val_interval == 0:
            val_loss = run_validation(model, val_dataset, args.batch_size, args.context_length, device)
            print(f"--- Validation loss at step {step}: {val_loss:.4f} ---")
            if args.use_wandb and HAS_WANDB:
                wandb.log({
                    "val/loss": val_loss,
                    "val/step": step,
                    "val/time_elapsed": time.time() - start_time,
                })

            # 9. 保存检查点
            save_path = os.path.join(args.save_dir, f"checkpoint_{step}.pth")
            save_checkpoint(model, optimizer, step, save_path)
            print(f"Saved checkpoint to {save_path}")

    print("Training finished.")
    if args.use_wandb and HAS_WANDB:
        wandb.finish()

if __name__ == '__main__':
    # main()
    # run_validation()
    parser = argparse.ArgumentParser(description="CS336 Assignment-1 5.3 Training Loop")
    # 数据和权重路径相关
    def clearfy_true_false(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif s.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data(.npy).")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data(.npy).")
    parser.add_argument("--load_from_checkpoint", type=clearfy_true_false, default=True, help="Whether to load checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pretrained checkpoint.")
    parser.add_argument("--save_dir", type=str, required=True, help="Dir path to save checkpoint.")
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=64, help="Context length.")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")
    # 优化器超参数
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_learning_rate", type=float, default=1e-3, help="Max LR for cosine schedule")
    parser.add_argument("--min_learning_rate", type=float, default=1e-4, help="Min LR (end of training)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000, help="Steps for cosine cycle to reach min_lr")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max L2 norm")
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_steps", type=int, default=200, help="Total number of training steps.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use(cuda/cpu).")
    # --- 日志与验证 (Logging & Validation) ---
    parser.add_argument("--log_interval", type=int, default=10, help="Print logs every N steps")
    parser.add_argument("--val_interval", type=int, default=500, help="Run validation every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    main(args)

    print('OK!')


"""
How to train:
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --save_dir checkpoints/run1 \
    --vocab_size 10000 \
    --context_length 128 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --max_steps 5000 \
    --batch_size 32 \
    --use_wandb
"""