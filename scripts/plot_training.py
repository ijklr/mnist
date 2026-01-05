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
    p.add_argument("--out", default="plots/single_layer_loss.png", help="Output image path")
    p.add_argument("--title", default="Single-layer Perceptron Training Loss", help="Plot title")
    p.add_argument("--subtitle", default="", help="Subtitle or hyperparams to show on the plot")
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
    plt.title(args.title)
    # Optional subtitle / hyperparameter text below the title
    if args.subtitle:
        plt.figtext(0.5, 0.92, args.subtitle, ha='center', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
