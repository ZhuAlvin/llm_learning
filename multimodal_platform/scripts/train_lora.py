"""Train LoRA script: command-line entry point for LoRA fine-tuning."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lora_manager import inject_lora_to_vit


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for ViT-Tiny")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Synthetic data for demo
    torch.manual_seed(42)
    n_train = args.num_classes * args.num_shots
    n_val = args.num_classes * 20

    train_x = torch.randn(n_train, 3, 224, 224)
    train_y = torch.repeat_interleave(torch.arange(args.num_classes), args.num_shots)
    val_x = torch.randn(n_val, 3, 224, 224)
    val_y = torch.repeat_interleave(torch.arange(args.num_classes), 20)

    train_ld = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_ld = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # Load base model
    print(f"Loading ViT-Tiny (num_classes={args.num_classes})...")
    base = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=args.num_classes)

    # Inject LoRA
    lora_params = inject_lora_to_vit(base, rank=args.rank, alpha=args.alpha)
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"LoRA trainable params: {trainable:,}")

    # Train
    base.train()
    opt = optim.AdamW(filter(lambda p: p.requires_grad, base.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        total_loss = 0
        for x, y in train_ld:
            opt.zero_grad()
            loss = crit(base(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{args.epochs} Loss: {total_loss/len(train_ld):.4f}")

    # Evaluate
    base.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_ld:
            correct += (base(x).argmax(1) == y).sum().item()
    acc = correct / n_val
    print(f"Validation accuracy: {correct}/{n_val} = {acc:.4f}")

    # Save LoRA weights
    lora_sd = {k: v.clone() for k, v in base.state_dict().items() if "lora_" in k}
    output_path = os.path.join(args.output_dir, "lora_trained.pt")
    torch.save({"state_dict": lora_sd, "accuracy": acc, "rank": args.rank}, output_path)
    print(f"Saved LoRA weights to {output_path}")
    print(f"File size: {os.path.getsize(output_path)/1024:.1f} KB")


if __name__ == "__main__":
    main()
