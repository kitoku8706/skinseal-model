"""
Acne 전용 이진 분류 학습 스크립트 (PyTorch, EfficientNet-B0)
- 데이터: skinseal-model/train/* 를 사용하여 Acne(양성=1) vs 비여드름(음성=0)
- 검증: train 에서 stratified split (기본 0.15)
- 저장: skinseal-pythonAI/models/best_acne_model.pth

실행 예 (Windows PowerShell):
python .\skinseal-model\train_acne_binary.py -e 10 -b 32 -o .\skinseal-pythonAI\models\best_acne_model.pth

주의:
- 사전학습 가중치 다운로드가 제한된 환경을 고려하여 기본 weights=None, 실패 시에도 작동합니다.
- GPU가 있으면 자동으로 사용, 없으면 CPU 사용.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models


@dataclass
class TrainConfig:
    train_dir: Path
    val_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    img_size: Tuple[int, int] = (224, 224)
    out_path: Path = Path("skinseal-pythonAI/models/best_acne_model.pth")
    seed: int = 42
    use_pretrained: bool = False  # 온라인 제약 고려하여 기본 False


class BinaryWrapper(Dataset):
    """ImageFolder 기반을 이진 레이블(Non-Acne=0, Acne=1)로 변환하는 래퍼"""
    def __init__(self, base: datasets.ImageFolder):
        self.base = base
        # 원본 클래스명 역매핑
        idx_to_class = {v: k for k, v in base.class_to_idx.items()}
        self.targets = [1 if idx_to_class[y] == "Acne" else 0 for y in base.targets]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        y = self.targets[idx]
        return x, y


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(num_classes: int = 2, use_pretrained: bool = False) -> nn.Module:
    try:
        if use_pretrained:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            m = models.efficientnet_b0(weights=None)
    except Exception:
        m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return m


def split_stratified_indices(labels, val_ratio: float, seed: int):
    from collections import defaultdict
    import random

    n = len(labels)
    idxs = list(range(n))
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    val_idx = []
    train_idx = []
    rnd = random.Random(seed)
    for _, arr in by_class.items():
        rnd.shuffle(arr)
        k = max(1, int(len(arr) * val_ratio))
        val_idx.extend(arr[:k])
        train_idx.extend(arr[k:])
    return train_idx, val_idx


def accuracy_and_f1(outputs: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        acc = correct / targets.numel()
        # F1 (positive class=1)
        tp = int(((preds == 1) & (targets == 1)).sum().item())
        fp = int(((preds == 1) & (targets == 0)).sum().item())
        fn = int(((preds == 0) & (targets == 1)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, f1


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 경로 준비
    train_dir = cfg.train_dir
    assert train_dir.exists(), f"Train dir not found: {train_dir}"

    # 변환 정의
    train_tf = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base = datasets.ImageFolder(str(train_dir), transform=train_tf)
    bin_full = BinaryWrapper(base)

    train_idx, val_idx = split_stratified_indices(bin_full.targets, cfg.val_ratio, cfg.seed)
    train_ds = Subset(bin_full, train_idx)
    val_base = datasets.ImageFolder(str(train_dir), transform=val_tf)
    val_bin = BinaryWrapper(val_base)
    val_ds = Subset(val_bin, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(num_classes=2, use_pretrained=cfg.use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        # validation
        model.eval()
        val_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = torch.as_tensor(targets, dtype=torch.long, device=device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += float(loss.item()) * images.size(0)
                acc, f1 = accuracy_and_f1(outputs, targets)
                total_acc += acc * images.size(0)
                total_f1 += f1 * images.size(0)
                count += images.size(0)
        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_acc = total_acc / max(1, count)
        val_f1 = total_f1 / max(1, count)

        print(f"Epoch [{epoch}/{cfg.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                'epoch': epoch,
                'val_f1': best_f1,
                'model_state_dict': model.state_dict(),
            }

    # 저장 경로 보장
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {'epoch': 0, 'val_f1': 0.0, 'model_state_dict': model.state_dict()}
    torch.save(best_state, str(cfg.out_path))
    print(f"Saved best acne model → {cfg.out_path} (val_f1={best_state['val_f1']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train Acne binary classifier (EfficientNet-B0)")
    default_train = Path(__file__).resolve().parent / "train"
    default_out = Path(__file__).resolve().parents[1] / "skinseal-pythonAI" / "models" / "best_acne_model.pth"

    parser.add_argument('-t', '--train-dir', type=str, default=str(default_train), help='Train directory (ImageFolder)')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio from train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224], help='Image size H W')
    parser.add_argument('--workers', type=int, default=4, help='Num workers')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights (if available)')
    parser.add_argument('-o', '--out', type=str, default=str(default_out), help='Output model path (.pth)')

    args = parser.parse_args()

    cfg = TrainConfig(
        train_dir=Path(args.train_dir),
        val_ratio=float(args.val_ratio),
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        img_size=(int(args.img_size[0]), int(args.img_size[1])),
        out_path=Path(args.out),
        use_pretrained=bool(args.pretrained),
    )

    train(cfg)


if __name__ == "__main__":
    main()
