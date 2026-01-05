#!/usr/bin/env python3
"""
Read a CSV training log (`epoch,loss`) and save a loss plot to `plots/loss.png`.
"""
import argparse
import csv
import os
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to CSV file with columns: epoch,loss")
    p.add_argument("--out", default="plots/loss.png", help="Output image path")
    args = p.parse_args()

    epochs = []
    losses = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8,4))
    plt.plot(epochs, losses, marker=".")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
