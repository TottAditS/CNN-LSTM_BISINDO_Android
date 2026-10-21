"""
CNNsa-BiLSTM training: data loading, training and validation loop, early
stopping, LR scheduling, checkpointing, and TensorBoard logging.
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.CNNsaBiLSTM.Config.train_config import (
    TRAIN_NPZ, VAL_NPZ, MODEL_CONFIG, TRAIN_CONFIG,
    EARLY_STOPPING_CONFIG, LR_SCHEDULER_CONFIG, CHECKPOINT_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR,
    create_output_dirs, verify_data_exists
)
from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM
from Training.Scripts.device_utils import get_device


class KeypointDataset(Dataset):
    """Keypoint sequences loaded from a .npz file."""
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        # X shape: (N, T, 126), y shape: (N,)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)
        self.ids = data["ids"] if "ids" in data else [f"sample_{i}" for i in range(len(self.y))]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    def __init__(self, device: torch.device):
        self.device = device

        torch.manual_seed(TRAIN_CONFIG["random_seed"])
        np.random.seed(TRAIN_CONFIG["random_seed"])
        create_output_dirs()

        self.writer = SummaryWriter(log_dir=str(LOG_DIR))

        print("\n[Trainer] Loading datasets...")
        self.train_dataset = KeypointDataset(TRAIN_NPZ)
        self.val_dataset = KeypointDataset(VAL_NPZ)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0, # avoids multiprocessing issues on Windows
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"  Train samples: {len(self.train_dataset)} (Batches: {len(self.train_loader)})")
        print(f"  Val samples  : {len(self.val_dataset)} (Batches: {len(self.val_loader)})")

        print("\n[Trainer] Initializing CNNsa-BiLSTM model...")
        self.model = CNNsaBiLSTM(
            input_size=MODEL_CONFIG["input_size"],
            feature_dim=MODEL_CONFIG["feature_dim"],
            hidden_size=MODEL_CONFIG["hidden_size"],
            num_layers=MODEL_CONFIG["num_layers"],
            num_classes=MODEL_CONFIG["num_classes"],
            dropout=MODEL_CONFIG["dropout"],
            attn_reduction=MODEL_CONFIG["attn_reduction"]
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )

        # Newer PyTorch versions removed verbose=, so fit() prints the LR every epoch.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min" if LR_SCHEDULER_CONFIG["monitor"] == "val_loss" else "max",
            factor=LR_SCHEDULER_CONFIG["factor"],
            patience=LR_SCHEDULER_CONFIG["patience"],
            min_lr=LR_SCHEDULER_CONFIG["min_lr"],
        )

        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs, _ = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        max_epochs = TRAIN_CONFIG["max_epochs"]
        print(f"\n[Trainer] Starting training on {self.device} for {max_epochs} epochs...")

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

            # Checkpointing and early stopping on the configured metric
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
                print(f"\n[Trainer] Early stopping triggered. No improvement for {self.patience_counter} epochs.")
                break

        self.writer.close()

        history_path = RESULTS_DIR / "train_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[Trainer] Training complete. History saved to {history_path}")


def run_training():
    if not verify_data_exists():
        return

    device = get_device()
    trainer = Trainer(device)
    trainer.fit()


if __name__ == "__main__":
    run_training()
