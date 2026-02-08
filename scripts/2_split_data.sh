#!/bin/bash
# ISD-Agent-Bench (English) - Split Train/Test Data
# Stratified sampling to split data into train (95%) and test (5%)

set -e

echo "=========================================="
echo "  Splitting Train/Test Data"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$BASE_DIR"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run split script
python scenarios/split_train_test.py --seed 42

echo ""
echo "Data split complete!"
echo "  Train: scenarios/train/"
echo "  Test: scenarios/test/"
