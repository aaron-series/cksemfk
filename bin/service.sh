#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

VENV_PATH="$PROJECT_ROOT/venv"
PYTHON_BIN="$VENV_PATH/bin/python"
UVICORN_BIN="$VENV_PATH/bin/uvicorn"
CONFIG_FILE="$PROJECT_ROOT/conf/config.yaml"

# ===== vLLM / CUDA 핵심 =====
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/libs:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# 필요시 활성
# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_P2P_DISABLE=1

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'


# ----------------------------------------------------------
# config
# ----------------------------------------------------------
get_conf() {
    "$PYTHON_BIN" - <<EOF
import yaml, sys
try:
    c=yaml.safe_load(open('$CONFIG_FILE'))
    print(c['$1']['$2'])
except Exception as e:
    sys.exit(1)
EOF
}

HOST=$(get_conf "server" "host")
PORT=$(get_conf "server" "port")
APP_MODULE=$(get_conf "server" "app_module")

PID_FILE="$PROJECT_ROOT/$(get_conf "paths" "pid_file")"
LOG_FILE="$PROJECT_ROOT/$(get_conf "paths" "log_file")"


# ----------------------------------------------------------
check_venv() {
    if [ ! -f "$PYTHON_BIN" ]; then
        echo -e "${RED}No venv python${NC}"
        exit 1
    fi

    source "$VENV_PATH/bin/activate"

    # 🔥 핵심: chandra + vllm 체크
    "$PYTHON_BIN" - <<EOF
import chandra
from chandra.model.vllm import generate_vllm
print("env ok")
EOF

    if [ $? -ne 0 ]; then
        echo -e "${RED}vLLM environment broken${NC}"
        exit 1
    fi
}


get_pid() {
    [ -f "$PID_FILE" ] && cat "$PID_FILE"
}

is_running() {
    pid=$(get_pid)
    [ -n "$pid" ] && ps -p "$pid" > /dev/null
}


# ----------------------------------------------------------
# 🔥 핵심: 자식 전체 종료
kill_tree() {
    local pid=$1

    for child in $(pgrep -P $pid); do
        kill_tree $child
    done

    kill -TERM $pid 2>/dev/null
}


# ----------------------------------------------------------
start() {
    check_venv

    if is_running; then
        echo -e "${YELLOW}Already running PID=$(get_pid)${NC}"
        exit 0
    fi

    mkdir -p "$(dirname "$LOG_FILE")"

    echo -e "${GREEN}Starting vLLM API $HOST:$PORT${NC}"

    nohup "$UVICORN_BIN" "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    sleep 2

    if is_running; then
        echo -e "${GREEN}Started PID=$(get_pid)${NC}"
    else
        echo -e "${RED}Start failed - check $LOG_FILE${NC}"
        rm -f "$PID_FILE"
    fi
}


# ----------------------------------------------------------
stop() {
    if ! is_running; then
        echo -e "${YELLOW}Not running${NC}"
        rm -f "$PID_FILE"
        return
    fi

    pid=$(get_pid)

    echo -e "Stopping PID=$pid"

    # 🔥 핵심
    kill_tree $pid

    for i in {1..20}; do
        is_running || break
        sleep 0.5
    done

    is_running && kill -9 $pid

    rm -f "$PID_FILE"

    deactivate 2>/dev/null

    echo -e "${GREEN}Stopped${NC}"
}


# ----------------------------------------------------------
console() {
    check_venv
    stop >/dev/null 2>&1

    "$UVICORN_BIN" "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --reload
}


case "$1" in
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 1; start ;;
    status) is_running && echo "running PID=$(get_pid)" || echo "stopped" ;;
    console) console ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|console}"
        ;;
esac
