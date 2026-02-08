# Baseline ISD Agent

Single prompt ADDIE instructional design generator with multi-provider support.

## Overview

Baseline is a reference agent that generates complete ADDIE outputs with a single prompt.
Used as a comparison baseline for evaluating other agents.

Supports multiple LLM providers: Upstage Solar, OpenRouter, OpenAI.

## Features

- Single API call generates complete ADDIE output
- Multi-provider support (Upstage, OpenRouter, OpenAI)
- Bloom's Taxonomy based learning objective design
- Gagné's 9 Events of Instruction application
- Structured JSON output

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Basic execution
baseline run --input scenario.json --output result.json

# Show info
baseline info
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--input, -i` | Input scenario JSON file |
| `--output, -o` | Output ADDIE result file |
| `--trajectory, -t` | Trajectory save file (optional) |
| `--model` | LLM model (default: varies by provider) |
| `--verbose, -v` | Verbose output |

## Environment Variables

```bash
# For Upstage Solar
export UPSTAGE_API_KEY="your-api-key"

# For OpenRouter
export OPENROUTER_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"
```

## Default Models

| Provider | Default Model |
|----------|---------------|
| Upstage | solar-pro-3 |
| OpenRouter | anthropic/claude-3.5-sonnet |
| OpenAI | gpt-4-turbo |

## Implementation Details

This implementation follows **Zero-shot Chain-of-Thought (CoT)** and **Single Prompting** approach as a baseline model.

### Key Characteristics
- **Single-Turn Generation**: Generates complete results with a single system prompt and user input, without complex agent interactions
- **Zero-shot CoT**: Induces step-by-step reasoning through prompt instructions only, without few-shot examples
- **Multi-Provider Support**: Serves as a benchmark baseline for measuring LLM performance across different providers
