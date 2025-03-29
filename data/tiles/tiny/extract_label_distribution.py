#!/usr/bin/env python3
import os
import sys


def extract_label_distribution(file_path):
    try:
        with open(file_path, "r") as f:
            in_label_section = False
            for line in f:
                stripped = line.strip()
                if stripped == "=== Label Distribution ===":
                    in_label_section = True
                    continue
                elif stripped.startswith("===") and in_label_section:
                    break
                if in_label_section and stripped.startswith("Label "):
                    print(stripped)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)


def main(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "stats.txt":
                file_path = os.path.join(root, file)
                print(f"File: {file_path}")
                extract_label_distribution(file_path)
                print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_label_distribution.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory or does not exist.")
        sys.exit(1)
    main(directory)
