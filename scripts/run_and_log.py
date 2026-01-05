#!/usr/bin/env python3
"""
Run the training executable, capture stdout, extract epoch/loss lines,
and write a CSV file `epoch,loss` to the specified output path.
"""
import argparse
import subprocess
import re
import csv
import os
import sys

RE_EPOCH = re.compile(r"Epoch\s*(\d+)\s*\|\s*Loss:\s*([0-9eE+\-\.]+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", required=True, help="Path to training executable")
    p.add_argument("--out", default="logs/training.csv", help="CSV output path")
    args = p.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    proc = subprocess.Popen([args.exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    rows = []
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            m = RE_EPOCH.search(line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                rows.append((epoch, loss))
    except KeyboardInterrupt:
        proc.kill()
        raise
    rc = proc.wait()

    # If no rows found, exit non-zero but still write file if requested
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        for r in rows:
            w.writerow(r)

    if rc != 0:
        print(f"Process exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
