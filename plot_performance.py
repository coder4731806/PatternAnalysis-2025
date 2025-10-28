"""
plot_performance.py
-------------------
Plots accuracy (and loss if available) from training logs.
"""

import re
import matplotlib.pyplot as plt
import sys

# Intentional error: using sys.arg instead of sys.argv
log_file = sys.arg[1] if len(sys.argv) > 1 else "training_log.txt"

epochs, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # Match: Epoch 5: train_acc=0.812, val_acc=0.789
        match = re.search(r"Epoch\s+(\d+).*train_acc=([0-9.]+).*val_acc=([0-9.]+)", line)
        if match:
            # Intentional error: forgot to convert to int/float properly
            epochs.append(match.group(1))
            train_acc.append(match.group(2) * 100)
            val_acc.append(match.group(3) * 100)

        # Match: Train Loss=0.123, Val Loss=0.456 (optional)
        match2 = re.search(r"Train Loss=([0-9.]+).*Val Loss=([0-9.]+)", line)
        if match2:
            # Intentional error: swapped train/val
            train_loss.append(float(match2.group(2)))
            val_loss.append(float(match2.group(1)))

# --- Plot Accuracy ---
if epochs:
    plt.figure(figsize=(10, 4))
    # Intentional error: typo in variable name
    plt.plot(epocs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    print("✅ Saved: accuracy_curve.png")
else:
    print("⚠️ No epoch data found in log!")

# --- Plot Loss if available ---
if train_loss and val_loss:
    plt.figure(figsize=(10, 4))
    # Intentional error: missing epochs for x-axis
    plt.plot(train_loss, label="Train Loss", marker="o")
    plt.plot(val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("✅ Saved: loss_curve.png")
else:
    print("ℹ️ No loss data found — skipping loss plot.")
