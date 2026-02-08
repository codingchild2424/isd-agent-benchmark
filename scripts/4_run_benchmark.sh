#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null
export OPENROUTER_API_KEY
export UPSTAGE_API_KEY
export UPSTAGE_API_KEY2
export UPSTAGE_API_KEY3

# All models via OpenRouter (tmux 병렬 실행)
AGENTS="baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent"

# Rate Limit 설정 - turbo로 변경 (6x16=96 동시)
RATE_LIMIT="turbo"

# 로그 디렉토리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/benchmark_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=============================================="
echo "  3개 모델 tmux 병렬 실행 (GPT + Gemini + Solar)"
echo "  Rate Limit: $RATE_LIMIT (6x16=96 동시)"
echo "  Solar: Upstage API 3개 키 라운드 로빈"
echo "  로그 디렉토리: $LOG_DIR"
echo "=============================================="

# 기존 세션 정리 (있으면)
tmux kill-session -t bench-gpt 2>/dev/null
tmux kill-session -t bench-gemini 2>/dev/null
tmux kill-session -t bench-solar 2>/dev/null

# GPT-5-mini
echo "[1/3] GPT-5-mini (OpenRouter)..."
tmux new-session -d -s bench-gpt \
"cd $(pwd) && source .env && export OPENROUTER_API_KEY && MODEL_PROVIDER=openrouter MODEL_NAME=openai/gpt-5-mini python run_benchmark.py --dataset test --agents $AGENTS --rate-limit $RATE_LIMIT 2>&1 | tee $LOG_DIR/gpt.log; echo '완료!'; read"

# Gemini-3-Flash
echo "[2/3] Gemini-3-Flash (OpenRouter)..."
tmux new-session -d -s bench-gemini \
"cd $(pwd) && source .env && export OPENROUTER_API_KEY && MODEL_PROVIDER=openrouter MODEL_NAME=google/gemini-3-flash-preview python run_benchmark.py --dataset test --agents $AGENTS --rate-limit $RATE_LIMIT 2>&1 | tee $LOG_DIR/gemini.log; echo '완료!'; read"

# Solar Pro3 - Upstage (라운드 로빈: UPSTAGE_API_KEY, UPSTAGE_API_KEY2, UPSTAGE_API_KEY3)
echo "[3/3] Solar-Pro3 (Upstage 라운드 로빈 3키)..."
tmux new-session -d -s bench-solar \
"cd $(pwd) && source .env && export UPSTAGE_API_KEY UPSTAGE_API_KEY2 UPSTAGE_API_KEY3 && MODEL_PROVIDER=upstage MODEL_NAME=solar-pro3 python run_benchmark.py --dataset test --agents $AGENTS --rate-limit $RATE_LIMIT 2>&1 | tee $LOG_DIR/solar.log; echo '완료!'; read"

echo ""
echo "=============================================="
echo "  3개 tmux 세션 실행 중!"
echo "=============================================="
echo ""
echo "📺 세션 목록:"
tmux ls
echo ""
echo "📊 실시간 확인 (attach):"
echo "  tmux attach -t bench-gpt"
echo "  tmux attach -t bench-gemini"
echo "  tmux attach -t bench-solar"
echo ""
echo "📁 로그 파일 (실시간 저장):"
echo "  tail -f $LOG_DIR/gpt.log"
echo "  tail -f $LOG_DIR/gemini.log"
echo "  tail -f $LOG_DIR/solar.log"
echo ""
echo "⌨️  세션에서 나가기: Ctrl+B, D"
echo ""
echo "🛑 전체 중지:"
echo "  tmux kill-server"
