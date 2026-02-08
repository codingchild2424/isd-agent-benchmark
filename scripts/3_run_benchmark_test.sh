#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null
export OPENROUTER_API_KEY

# Test: 1 scenario, all 6 agents, Claude Opus 4.5 via OpenRouter
AGENTS="baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent"
RATE_LIMIT="conservative"

MODEL_PROVIDER=openrouter MODEL_NAME=anthropic/claude-opus-4.5 \
python run_benchmark.py --scenario scenarios/test/scenario_bal_large2_0075.json --agents $AGENTS --rate-limit $RATE_LIMIT --verbose
