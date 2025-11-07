#!/usr/bin/env bash
set -euo pipefail

USER="root"
HOSTS=(worker-0 worker-1 worker-2 worker-3)
SRC_DIR="data/processed"
DEST_DIR="/mnt/local-data/data/processed"
FILES=(vhh_200.tsv vhh_2K.tsv vhh_20K.tsv test_vhh_20K.tsv)

for H in "${HOSTS[@]}"; do
  TARGET="${USER:+$USER@}$H"
  echo "==> $H: create $DEST_DIR"
  ssh "$TARGET" "mkdir -p '$DEST_DIR'"

  echo "==> $H: copy files..."
  for f in "${FILES[@]}"; do
    SRC_PATH="$SRC_DIR/$f"
    if [[ ! -f "$SRC_PATH" ]]; then
      echo "ERROR: source file not found: $SRC_PATH" >&2
      exit 1
    fi
    scp "$SRC_PATH" "$TARGET:$DEST_DIR/"
  done

  echo "==> $H: done."
done

echo "All done."
