#!/bin/bash
# ISD-Agent-Bench (English) - Setup Script
# Install all dependencies for running the benchmark

set -e

echo "=========================================="
echo "  ISD-Agent-Bench (English) Setup"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$BASE_DIR"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base dependencies
echo "Installing base dependencies..."
pip install langchain langchain-core langgraph openai tqdm typer rich

# Install each agent
echo "Installing agents..."
for agent_dir in agents/*/; do
    if [ -f "${agent_dir}pyproject.toml" ] || [ -f "${agent_dir}setup.py" ]; then
        echo "  Installing $(basename $agent_dir)..."
        pip install -e "$agent_dir" 2>/dev/null || true
    fi
done

# Install evaluator
if [ -d "evaluator" ]; then
    echo "Installing evaluator..."
    pip install -e evaluator/ 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set your API key: export UPSTAGE_API_KEY='your-key'"
echo "  2. Split data: ./scripts/split_data.sh"
echo "  3. Run benchmark: ./scripts/run_benchmark.sh"
