#!/usr/bin/env bash
set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

DATASETS=("wikipedia" "reddit")

for DATASET in "${DATASETS[@]}"; do
    CSV_FILE="${DATA_DIR}/${DATASET}.csv"
    URL="https://snap.stanford.edu/jodie/${DATASET}.csv"

    if [ -f "$CSV_FILE" ]; then
        echo "Dataset $DATASET already exists at $CSV_FILE"
    else
        echo "Downloading $DATASET dataset from $URL..."
        curl -sSL "$URL" -o "$CSV_FILE"
        echo "Successfully downloaded $DATASET dataset to $CSV_FILE"
    fi
done

echo "Data download completed."
