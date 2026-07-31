"""
Training/MViTv2/Scripts/mvit_trainer.py
==========================================
Script untuk fine-tune MViTv2 (baseline transformer).
- Dataset & DataLoader loading (raw video clip, lihat mvit_dataset.py)
- Training & Validation loop dengan mixed precision (AMP) -- perlu untuk muat di VRAM 8GB
- Differential learning rate (backbone kecil, head besar)
- Early stopping & LR scheduling
- Saving checkpoints
- TensorBoard logging

Struktur mengikuti Training/CNNsaBiLSTM/Scripts/trainer.py supaya konsisten, dengan
penyesuaian untuk model video yang jauh lebih berat (batch kecil, AMP, GradScaler).
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.MViTv2.Config.mvit_config import (
    MODEL_CONFIG, TRAIN_CONFIG,
    EARLY_STOPPING_CONFIG, LR_SCHEDULER_CONFIG, CHECKPOINT_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR,
    create_output_dirs, verify_data_exists,
)
from Training.MViTv2.Models.mvit_v2 import build_mvit_v2, get_param_groups
from Training.MViTv2.Scripts.mvit_dataset import build_mvit_datasets
from Training.Scripts.device_utils import get_device


class MViTTrainer:
    def __init__(self, device: torch.device):
        self.device = device

        # 1. Setup seed & directory
        torch.manual_seed(TRAIN_CONFIG["random_seed"])
        create_output_dirs()

        # 2. TensorBoard logger
        self.writer = SummaryWriter(log_dir=str(LOG_DIR))

        # 3. Load datasets (raw video clip, split shared dgn CNNsa-BiLSTM)
        print("\n[MViTTrainer] Loading datasets...")
        self.train_dataset, self.val_dataset, _ = build_mvit_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,  # Amankan dari multi-processing issues di Windows
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        print(f"  Train samples: {len(self.train_dataset)} (Batches: {len(self.train_loader)})")
        print(f"  Val samples  : {len(self.val_dataset)} (Batches: {len(self.val_loader)})")

        # 4. Initialize model
        print("\n[MViTTrainer] Initializing MViTv2 model...")
        self.model = build_mvit_v2(
            num_classes=MODEL_CONFIG["num_classes"],
            freeze_until_block=MODEL_CONFIG["freeze_until_block"],
            pretrained=MODEL_CONFIG["pretrained"],
            dropout=MODEL_CONFIG["dropout"],
        ).to(self.device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

        # 5. Loss & Optimizer (differential LR: backbone kecil, head besar)
        self.criterion = nn.CrossEntropyLoss()
        param_groups = get_param_groups(
            self.model,
            backbone_lr=TRAIN_CONFIG["backbone_lr"],
            head_lr=TRAIN_CONFIG["head_lr"],
        )
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=TRAIN_CONFIG["weight_decay"])

        # 6. Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min" if LR_SCHEDULER_CONFIG["monitor"] == "val_loss" else "max",
            factor=LR_SCHEDULER_CONFIG["factor"],
            patience=LR_SCHEDULER_CONFIG["patience"],
            min_lr=LR_SCHEDULER_CONFIG["min_lr"],
        )

        # 7. Mixed precision (AMP) -- wajib untuk muat MViTv2 + clip 16x224x224 di VRAM 8GB
        self.use_amp = TRAIN_CONFIG["use_amp"] and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if TRAIN_CONFIG["use_amp"] and not torch.cuda.is_available():
            print("  [WARN] use_amp=True tapi tidak ada GPU -- AMP dinonaktifkan (CPU tidak didukung).")

        # 8. Early stopping variables
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

    def _forward_backward(self, X_batch, y_batch, train: bool):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return outputs, loss

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            outputs, loss = self._forward_backward(X_batch, y_batch, train=True)

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        return running_loss / total, correct / total

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs, loss = self._forward_backward(X_batch, y_batch, train=False)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        return running_loss / total, correct / total

    def fit(self):
        max_epochs = TRAIN_CONFIG["max_epochs"]
        print(f"\n[MViTTrainer] Starting training on {self.device} for {max_epochs} epochs "
              f"(AMP={'on' if self.use_amp else 'off'})...")

        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_CONFIG["filename"]

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(f"Epoch {epoch:03d}/{max_epochs:03d} | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% | "
                  f"LR: {current_lr:.6f}")

            is_better = False
            monitor_metric = EARLY_STOPPING_CONFIG["monitor"]
            min_delta = EARLY_STOPPING_CONFIG["min_delta"]

            if monitor_metric == "val_accuracy":
                if val_acc > self.best_val_acc + min_delta:
                    self.best_val_acc = val_acc
                    is_better = True
            elif monitor_metric == "val_loss":
                if val_loss < self.best_val_loss - min_delta:
                    self.best_val_loss = val_loss
                    is_better = True

            if is_better:
                self.patience_counter = 0
                if CHECKPOINT_CONFIG["save_best_only"]:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "model_config": MODEL_CONFIG,
                    }, checkpoint_path)
                    print(f"  [BEST] New best model saved to {checkpoint_path.name} (Val Acc: {val_acc*100:.2f}%)")
            else:
                self.patience_counter += 1

            if LR_SCHEDULER_CONFIG["monitor"] == "val_loss":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(val_acc)

            if self.patience_counter >= EARLY_STOPPING_CONFIG["patience"]:
                print(f"\n[MViTTrainer] Early stopping triggered. No improvement for {self.patience_counter} epochs.")
                break

        self.writer.close()

        history_path = RESULTS_DIR / "mvit_v2_train_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[MViTTrainer] Training complete. History saved to {history_path}")


def run_training():
    if not verify_data_exists():
        return

    device = get_device()
    trainer = MViTTrainer(device)
    trainer.fit()


if __name__ == "__main__":
    run_training()
