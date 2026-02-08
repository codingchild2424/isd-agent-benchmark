# ISD-Agent-Bench

A comprehensive benchmark for evaluating LLM-based Instructional Systems Design (ISD) agents.

## Quick Start

### 1. Setup

```bash
# Copy environment file and add your API keys
cp .env.example .env
# Edit .env with your API keys

# Run setup script
chmod +x scripts/*.sh
./scripts/1_setup.sh
```

### 2. Configure Model Provider

Edit `.env` to choose your model:

**Option A: OpenAI**
```bash
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o
OPENAI_API_KEY=your-key-here
```

**Option B: OpenRouter** (Access to Claude, GPT-4, Llama, etc.)
```bash
MODEL_PROVIDER=openrouter
MODEL_NAME=anthropic/claude-3.5-sonnet
OPENROUTER_API_KEY=your-key-here
```

Available OpenRouter models:
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B
- `google/gemini-pro-1.5` - Gemini Pro 1.5

### 3. Run Benchmark

```bash
# Split data into train/test
./scripts/2_split_data.sh

# Run benchmark test
./scripts/3_run_benchmark_test.sh

# Run full benchmark
./scripts/4_run_benchmark.sh
```

## Dataset

| Folder | Count | Description |
|--------|-------|-------------|
| scenarios/idld_aligned | 8,842 | SCOPUS paper-based scenarios |
| scenarios/context_variant/part1 | 9,000 | Context Matrix variations |
| scenarios/context_variant/part2 | 7,953 | Context Matrix variations |
| **Total** | **25,795** | All scenarios |

## Agents

| Agent | Type | Description |
|-------|------|-------------|
| Baseline | General | Single LLM call |
| ReAct-ISD | General | ReAct pattern with ISD tools |
| EduPlanner | ISD-Specialized | Multi-agent collaboration |
| ADDIE-Agent | ISD-Specialized | ADDIE framework |
| Dick-Carey-Agent | ISD-Specialized | Dick & Carey model |
| RPISD-Agent | ISD-Specialized | Rapid Prototyping ISD |

## Directory Structure

```
isd-agent-bench/
├── .env.example          # Environment template
├── README.md             # This file
├── run_benchmark.py      # Main benchmark runner
├── scripts/              # Shell scripts
│   ├── 1_setup.sh        # Install dependencies
│   ├── 2_split_data.sh   # Split train/test
│   ├── 3_run_benchmark_test.sh  # Quick test
│   └── 4_run_benchmark.sh       # Full benchmark
├── scenarios/            # ISD scenarios (25,795)
│   ├── idld_aligned/     # SCOPUS-based (8,842)
│   ├── context_variant/  # Augmented (16,953)
│   ├── train/            # Training set
│   └── test/             # Test set
├── agents/               # 6 ISD agents
├── evaluator/            # ADDIE rubric evaluator
└── shared/               # Common schemas and utilities
```

## License

MIT License
