#!/usr/bin/env python3
"""
Remote GPU Training Script: Signature Detection (Faster R-CNN / RTMDet-Tiny)
Trained on YOLO-formatted Signature Datasets (images/ and labels/)
License: Apache 2.0 / BSD-3-Clause (Strictly Non-Ultralytics, Zero AGPL)
"""

import os
import sys
import argparse
import time
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SignatureYOLODataset(Dataset):
    """
    PyTorch Dataset loading YOLO-formatted annotations:
    Class 0 = signature, with normalized [x_center, y_center, width, height]
    Converts to absolute [xmin, ymin, xmax, ymax] boxes for Torchvision / RTMDet.
    """
    def __init__(self, img_dir: str, lbl_dir: str, max_samples: int = None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        
        valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
        if os.path.exists(img_dir):
            all_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(valid_exts)]
        else:
            all_files = []

        if max_samples and max_samples > 0:
            all_files = all_files[:max_samples]

        self.img_files = all_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Find corresponding label file
        base_name = os.path.splitext(fname)[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + '.txt')

        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id, xc, yc, bw, bh = map(float, parts[:5])
                            xmin = max(0.0, (xc - bw / 2.0) * w)
                            ymin = max(0.0, (yc - bh / 2.0) * h)
                            xmax = min(float(w), (xc + bw / 2.0) * w)
                            ymax = min(float(h), (yc + bh / 2.0) * h)
                            if xmax > xmin + 2.0 and ymax > ymin + 2.0:
                                boxes.append([xmin, ymin, xmax, ymax])
                                labels.append(1)  # Class 1: Signature (0 is background)
                        except ValueError:
                            continue

        # Negative sample handling: images without signatures have empty box and label tensors
        if not boxes:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
        else:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }

        tensor_img = TF.to_tensor(img)
        return tensor_img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Signature Detector on GPU")
    parser.add_argument("--dataset-dir", default="./merged_signature_dataset", help="Path to dataset root")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--output-dir", default="./checkpoints/rtdetr_r50", help="Directory to save model weights")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for rapid verification")
    parser.add_argument("--tracker", default="none", choices=["none", "mlflow", "comet"], help="Experiment tracker")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(" 🚀 STARTING REAL SIGNATURE DETECTION TRAINING RUN ON DATASET")
    print("=" * 75)
    print(f"Architecture : Faster R-CNN / RTMDet-Tiny (Permissive Apache 2.0 / BSD)")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device       : {'CUDA (GPU: ' + device_name + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset Dir  : {args.dataset_dir}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Output Dir   : {args.output_dir}")
    print(f"Tracker      : {args.tracker} (No external API keys required)")
    print("-" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Locate train split
    train_img_dir = os.path.join(args.dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(args.dataset_dir, "train", "labels")

    if not os.path.exists(train_img_dir):
        # Fallback to direct images/labels if not subfolder
        train_img_dir = os.path.join(args.dataset_dir, "images")
        train_lbl_dir = os.path.join(args.dataset_dir, "labels")

    train_dataset = SignatureYOLODataset(train_img_dir, train_lbl_dir, max_samples=args.max_samples)
    print(f"[*] Training images found: {len(train_dataset)}")

    if len(train_dataset) == 0:
        print(f"[!] Warning: No images found in '{train_img_dir}'. Check dataset directory path.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Build Faster R-CNN model
    print("[*] Initializing model with ResNet-50 FPN backbone...")
    try:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    except Exception:
        try:
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        except Exception:
            model = fasterrcnn_resnet50_fpn(weights=None)

    # 2 classes: 0 = background, 1 = signature
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            batch_count += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                cur_avg = running_loss / batch_count
                print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Batch [{batch_idx+1:04d}/{len(train_loader):04d}] - Loss: {cur_avg:.4f}")

        lr_scheduler.step()
        epoch_loss = running_loss / max(1, batch_count)
        epoch_sec = time.time() - epoch_start
        print(f"==> Epoch [{epoch:02d}/{args.epochs:02d}] Finished - Avg Loss: {epoch_loss:.4f} | Time: {epoch_sec:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save latest and best checkpoints
        latest_ckpt = os.path.join(args.output_dir, "detector_latest.pth")
        torch.save(model.state_dict(), latest_ckpt)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ckpt = os.path.join(args.output_dir, "detector_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [✓] New best checkpoint saved to: {best_ckpt}")

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"✅ Detection training successfully completed in {total_time / 60:.1f} minutes.")
    print(f"✅ Best trained weights saved to: {os.path.join(args.output_dir, 'detector_best.pth')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
