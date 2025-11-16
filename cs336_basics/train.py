'''
核心训练代码
'''
import os

import argparse
import numpy as np
import torch

from modules import (get_batch, TransformerLM, AdamW, SGD, cross_entropy_loss, gradient_clipping, save_checkpoint, \
                     load_checkpoint, cosine_annealing)

@torch.no_grad()
def run_validation(model, dataset, batch_size, context_length, device, num_batches=10):
    total_loss = 0.0
    model.eval()
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs, targets)
        total_loss += loss.item()
        print(f"Validation loss: {loss.item():.4f}")
    model.train()
    return total_loss / num_batches

def main(args):
    # 1. 选择device和生成权重保存路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 使用np.load(..., mmap_mode='r')加载数据集
    print("Loading data...")
    train_dataset = np.load(args.train_data_path, mmap_mode='r')
    val_dataset = np.load(args.val_data_path, mmap_mode='r')

    # 3. 初始化模型
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.dff,
        rope_theta=args.rope_theta
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # 4.初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. 加载检查点
    start_step = 0
    if args.load_from_checkpoint:
        try:
            start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
            print(f"Resuming training from step {start_step}")
        except FileNotFoundError:
            print("Failed to load checkpoint. Starting training from scratch.")

    # 6. 开始训练
    print("Starting training...")
    model.train()

    for step in range(start_step, args.max_steps):
        # 6.1 使用cosine_annealing确定学习率
        lr = cosine_annealing(step, args.max_learning_rate, args.min_learning_rate, args.warmup_steps, args.cosine_cycle_iters)
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
        # 6.6 裁剪梯度; model.parameters() -> Tensor: weight, bias, -> Tensor: grad
        gradient_clipping(model.parameters(), args.grad_clip)
        # 6.7 更新参数
        optimizer.step()

        # 7. 验证中间结果
        if step > 0 and step % args.val_interval == 0:
            val_loss = run_validation(model, val_dataset, args.batch_size, args.context_length, device)
            print(f"--- Validation loss at step {step}: {val_loss:.4f} ---")

            # 8. 保存检查点
            save_path = os.path.join(args.save_dir, f"checkpoint_{step}.pth")
            save_checkpoint(model, optimizer, step, save_path)

    print("Training finished.")

if __name__ == '__main__':
    # main()
    # run_validation()
    parser = argparse.ArgumentParser(description="CS336 Assignment-1 5.3 Training Loop")
    # 数据和权重路径相关
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data(.npy).")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data(.npy).")
    parser.add_argument("--load_from_checkpoint", type=bool, default=False, help="Whether to load checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument("--save_dir", type=str, required=True, help="Dir path to save checkpoint.")
    parser.add_argument("--context_length", type=int, default=64, help="Context length.")
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--max_seq_len", type=int, default=)
    # 优化器超参数
    pass
    # 训练超参数
    pass

    print('OK!')