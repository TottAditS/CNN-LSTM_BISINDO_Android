"""
Small hand-picked hyperparameter search for CNNsa-BiLSTM.

Each trial changes dropout, weight decay, hidden size, or learning rate, and is
compared with the baseline configuration on validation accuracy and the
train-validation gap. The test split is not used here.

Every run is written to its own folder,
Training/Output/hyperparameter_search/runs/<YYYYMMDD_HHMMSS>/:
    checkpoints/<trial>.pt      weights at the best validation epoch
    logs/<trial>/               TensorBoard logs
    history/<trial>.json        per-epoch loss, accuracy, and LR
    search_results.json         summary of all trials
    search_summary_table.csv    same summary as CSV
    search_comparison_chart.png validation accuracy per trial

The production checkpoint (cnnsa_bilstm_best.pt) is not modified. To adopt a
trial, copy its checkpoint over cnnsa_bilstm_best.pt, then rerun evaluator.py
and Export/Scripts/export_cnnsa_bilstm.py.

Usage (from the repository root):
    python Training/CNNsaBiLSTM/Scripts/hyperparameter_search.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.CNNsaBiLSTM.Config.train_config import TRAIN_NPZ, VAL_NPZ, MODEL_CONFIG as BASE_MODEL_CONFIG
from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM
from Training.CNNsaBiLSTM.Scripts.trainer import KeypointDataset
from Training.Scripts.device_utils import get_device

SEARCH_ROOT_DIR = Path(__file__).resolve().parents[3] / "Training" / "Output" / "hyperparameter_search"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = SEARCH_ROOT_DIR / "runs" / RUN_ID
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR / "logs"
HISTORY_DIR = RUN_DIR / "history"
RESULTS_PATH = RUN_DIR / "search_results.json"

RANDOM_SEED = 42
MAX_EPOCHS = 50          # shorter than the main training (100) to keep the search affordable
PATIENCE = 8             # main training uses 10
NUM_CLASSES = BASE_MODEL_CONFIG["num_classes"]
INPUT_SIZE = BASE_MODEL_CONFIG["input_size"]

# Hand-picked trials instead of a full grid. The first trial is the baseline.
TRIALS = [
    {"name": "baseline",              "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.4",           "dropout": 0.4, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.5",           "dropout": 0.5, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "wd_5e-4",               "dropout": 0.3, "weight_decay": 5e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "wd_1e-3",               "dropout": 0.3, "weight_decay": 1e-3, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.4_wd_5e-4",   "dropout": 0.4, "weight_decay": 5e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "hidden_128",            "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 128, "learning_rate": 1e-3},
    {"name": "hidden_128_dropout_0.4","dropout": 0.4, "weight_decay": 1e-4, "hidden_size": 128, "learning_rate": 1e-3},
    {"name": "lr_5e-4",               "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 5e-4},
    {"name": "lr_2e-3",               "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 2e-3},
    {"name": "hidden_128_wd_5e-4",    "dropout": 0.4, "weight_decay": 5e-4, "hidden_size": 128, "learning_rate": 1e-3},
]


def run_trial(trial_cfg: dict, device: torch.device, train_ds, val_ds) -> dict:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0,
                               pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    model = CNNsaBiLSTM(
        input_size=INPUT_SIZE,
        feature_dim=BASE_MODEL_CONFIG["feature_dim"],
        hidden_size=trial_cfg["hidden_size"],
        num_layers=BASE_MODEL_CONFIG["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=trial_cfg["dropout"],
        attn_reduction=BASE_MODEL_CONFIG["attn_reduction"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_cfg["learning_rate"],
                                  weight_decay=trial_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=str(LOG_DIR / trial_cfg["name"]))

    best_val_acc = 0.0
    best_train_acc_at_best = 0.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / f"{trial_cfg['name']}.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'='*70}\n[Trial] {trial_cfg['name']}: {trial_cfg}\n{'='*70}")
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_loss = running_loss / total
        val_acc = correct / total
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"  Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_train_acc_at_best = train_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "model_config": {
                    "input_size": INPUT_SIZE,
                    "feature_dim": BASE_MODEL_CONFIG["feature_dim"],
                    "hidden_size": trial_cfg["hidden_size"],
                    "num_layers": BASE_MODEL_CONFIG["num_layers"],
                    "num_classes": NUM_CLASSES,
                    "dropout": trial_cfg["dropout"],
                    "attn_reduction": BASE_MODEL_CONFIG["attn_reduction"],
                },
                "trial_hyperparams": trial_cfg,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience {PATIENCE})")
            break

    writer.close()
    with open(HISTORY_DIR / f"{trial_cfg['name']}.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t0
    gap = best_train_acc_at_best - best_val_acc
    print(f"[Trial] {trial_cfg['name']} finished in {elapsed:.1f}s, "
          f"best val_acc={best_val_acc*100:.2f}% @ epoch {best_epoch}, "
          f"train_acc={best_train_acc_at_best*100:.2f}%, gap={gap*100:.2f} points")

    return {
        "name": trial_cfg["name"],
        "hyperparams": trial_cfg,
        "best_val_acc": best_val_acc,
        "train_acc_at_best": best_train_acc_at_best,
        "gap_train_val": gap,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_acc"]),
        "elapsed_sec": elapsed,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(HISTORY_DIR / f"{trial_cfg['name']}.json"),
        "tensorboard_log_dir": str(LOG_DIR / trial_cfg["name"]),
    }


def generate_search_report(run_dir: Path, results: list[dict]) -> None:
    """Write the CSV summary and the validation accuracy chart of all trials to run_dir."""
    import csv

    results_sorted = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)

    csv_path = run_dir / "search_summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "dropout", "weight_decay", "hidden_size", "learning_rate",
                          "best_val_acc_pct", "train_acc_at_best_pct", "gap_points", "best_epoch"])
        for r in results_sorted:
            hp = r["hyperparams"]
            writer.writerow([
                r["name"], hp["dropout"], hp["weight_decay"], hp["hidden_size"], hp["learning_rate"],
                round(r["best_val_acc"] * 100, 2), round(r["train_acc_at_best"] * 100, 2),
                round((r["train_acc_at_best"] - r["best_val_acc"]) * 100, 2), r["best_epoch"],
            ])
    print(f"[HyperparameterSearch] Summary table saved to: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib is not installed, skipping the chart.")
        return

    names = [r["name"] for r in results_sorted]
    val_accs = [r["best_val_acc"] * 100 for r in results_sorted]
    colors = ["#d62728" if n == "baseline" else "#1f77b4" for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(names)), val_accs, color=colors)
    for bar, v in zip(bars, val_accs):
        ax.annotate(f"{v:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_ylim(min(val_accs) - 3, max(val_accs) + 3)
    ax.set_title("CNNsa-BiLSTM hyperparameter search (11 trials, sorted)")
    ax.grid(axis="y", alpha=0.3)

    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color="#d62728", label="baseline"),
                        mpatches.Patch(color="#1f77b4", label="hyperparameter trial")])
    fig.tight_layout()

    chart_path = run_dir / "search_comparison_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[HyperparameterSearch] Trial chart saved to: {chart_path}")


def backfill_report_from_existing_run(run_dir: Path) -> None:
    """Rebuild the CSV and chart of an earlier run from its search_results.json, without training."""
    results_path = run_dir / "search_results.json"
    with open(results_path) as f:
        data = json.load(f)
    results = data["trials"] if isinstance(data, dict) and "trials" in data else data
    generate_search_report(run_dir, results)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[HyperparameterSearch] Run ID  : {RUN_ID}")
    print(f"[HyperparameterSearch] Run dir : {RUN_DIR}")
    print(f"[HyperparameterSearch] Device  : {device}")
    print(f"[HyperparameterSearch] Running {len(TRIALS)} trials in sequence.")
    print(f"[HyperparameterSearch] The production checkpoint (cnnsa_bilstm_best.pt) is not modified.")

    train_ds = KeypointDataset(TRAIN_NPZ)
    val_ds = KeypointDataset(VAL_NPZ)
    print(f"[HyperparameterSearch] Train: {len(train_ds)} sequence | Val: {len(val_ds)} sequence")

    results = []
    for trial_cfg in TRIALS:
        result = run_trial(trial_cfg, device, train_ds, val_ds)
        results.append(result)
        # Save after every trial so finished trials survive an interrupted run
        with open(RESULTS_PATH, "w") as f:
            json.dump({"run_id": RUN_ID, "trials": results}, f, indent=2)

    results_sorted = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    print(f"\n{'='*70}\n[HyperparameterSearch] Run {RUN_ID}, sorted by validation accuracy\n{'='*70}")
    print(f"{'Trial':<22} {'Val Acc':>9} {'Train Acc':>10} {'Gap':>7} {'Epoch':>6}")
    for r in results_sorted:
        print(f"{r['name']:<22} {r['best_val_acc']*100:8.2f}% {r['train_acc_at_best']*100:9.2f}% "
              f"{r['gap_train_val']*100:6.2f}p {r['best_epoch']:6d}")

    generate_search_report(RUN_DIR, results)

    print(f"\nResults        : {RESULTS_PATH}")
    print(f"Checkpoint     : {CHECKPOINT_DIR}")
    print(f"History (JSON) : {HISTORY_DIR}")
    print(f"TensorBoard    : tensorboard --logdir \"{LOG_DIR}\"")
    print("\nThe production checkpoint (Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt)")
    print("is unchanged. See the module docstring for how to adopt a trial.")


if __name__ == "__main__":
    main()
