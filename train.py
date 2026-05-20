import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import resnet20


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_PATH = PROJECT_ROOT / "RESULT.md"
SUMMARY_JSON = PROJECT_ROOT / "run_summary.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(data_dir: Path, batch_size: int, train_size: int, test_size: int):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    if train_size and train_size < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(train_size)))
    if test_size and test_size < len(test_dataset):
        test_dataset = Subset(test_dataset, list(range(test_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Test", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return total_loss / total, 100.0 * correct / total, report


def write_report(args, device, train_history, test_loss, test_acc, report, elapsed_time):
    rows = pd.DataFrame(train_history)
    rows["train_loss"] = rows["train_loss"].map(lambda x: f"{x:.4f}")
    rows["train_acc"] = rows["train_acc"].map(lambda x: f"{x:.2f}%")
    table_lines = ["| epoch | train_loss | train_acc |", "| --- | --- | --- |"]
    for row in rows.itertuples(index=False):
        table_lines.append(f"| {row.epoch} | {row.train_loss} | {row.train_acc} |")
    table = "\n".join(table_lines)

    md = f"""# ResNet CIFAR-10 复现结果

## 实验配置
- 模型：ResNet-20（适配 CIFAR-10 的小型残差网络）
- 数据集：CIFAR-10
- 训练设备：{device}
- Epochs：{args.epochs}
- Batch Size：{args.batch_size}
- 学习率：{args.lr}
- 优化器：SGD(momentum=0.9, weight_decay=1e-4)
- 训练样本数：{args.train_size}
- 测试样本数：{args.test_size}
- 随机种子：{args.seed}
- 总耗时：{elapsed_time:.2f} 秒

## 训练过程
{table}

## 测试结果
- Test Loss：{test_loss:.4f}
- Test Accuracy：{test_acc:.2f}%

## 分类报告
```text
{report}
```

## 说明
- 这是课程作业可运行版复现，目标是复现 ResNet 的核心思想与训练流程。
- 本次运行使用了轻量配置，便于在普通 CPU 环境下完成演示。
- 若要更接近论文结果，应增加训练轮数，并在更强硬件上使用更完整的数据与训练策略。
"""
    RESULT_PATH.write_text(md, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Train a small ResNet on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_dataloaders(args.data_dir, args.batch_size, args.train_size, args.test_size)
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    model = resnet20(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    start_time = time.time()
    train_history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%")

    test_loss, test_acc, report = evaluate(model, test_loader, criterion, device, class_names)
    elapsed_time = time.time() - start_time
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), PROJECT_ROOT / "resnet20_cifar10_demo.pth")
    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "elapsed_time_sec": elapsed_time,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args, device, train_history, test_loss, test_acc, report, elapsed_time)


if __name__ == "__main__":
    main()


