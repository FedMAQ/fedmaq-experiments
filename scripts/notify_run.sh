#!/usr/bin/env bash

set -euo pipefail

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    . ".env"
    set +a
elif [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$(dirname "$0")/../.env"
    set +a
fi

NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
NTFY_HEARTBEAT_SEC="${NTFY_HEARTBEAT_SEC:-3600}"

show_help() {
    cat << 'EOF'
Usage: bash ./scripts/notify_run.sh [--test] <command> [args...]

Wraps any long-running command, streaming live output to the console while
dispatching start, heartbeat, success, or failure push notifications to an
authenticated, high-entropy ntfy.sh topic.

Requirements:
  NTFY_TOPIC and NTFY_TOKEN must be set in your .env file or environment.
    NTFY_TOPIC="fedmaq-thesis-bunyi-secret123"
    NTFY_TOKEN="tk_..."

Optional Environment Variables:
  NTFY_SERVER         Base URL of the ntfy server (default: https://ntfy.sh).
  NTFY_HEARTBEAT_SEC  Interval for progress heartbeats in seconds (default: 3600; 0 to disable).

Options:
  --test        Send a test notification to verify credentials and connectivity.
  -h, --help    Show this help message.

Examples:
  bash ./scripts/notify_run.sh --test
  bash ./scripts/notify_run.sh ./.venv/bin/python scripts/run_matrix.py --matrix baseline_tuning_wide --run_timeout_seconds 7200 -o ray.temp_dir=/tmp/ray-cjb -o ray.object_store_gb=4
EOF
}

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    [ $# -eq 0 ] && exit 1 || exit 0
fi

if [ -z "${NTFY_TOPIC:-}" ] || [ -z "${NTFY_TOKEN:-}" ]; then
    echo "[notify_run] ERROR: NTFY_TOPIC or NTFY_TOKEN is not set." >&2
    echo "[notify_run] Please set an unguessable high-entropy topic in .env or environment:" >&2
    echo "  NTFY_TOPIC=\"fedmaq-thesis-bunyi-secret123\"" >&2
    echo "  NTFY_TOKEN=\"tk_...\"" >&2
    exit 1
fi

HOST="$(hostname 2>/dev/null || echo "unknown-host")"

send_ntfy() {
    local title="$1"
    local priority="$2"
    local tags="$3"
    local body="$4"

    curl -s -o /dev/null -w "%{http_code}" -X POST "${NTFY_SERVER}/${NTFY_TOPIC}" \
        -H "Authorization: Bearer ${NTFY_TOKEN}" \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${body}" 2>/dev/null || echo "000"
}

if [ "${1:-}" = "--test" ]; then
    echo "[notify_run] Dispatching test notification to ${NTFY_SERVER}/${NTFY_TOPIC}..."
    HTTP_CODE=$(send_ntfy "FedMAQ: Test Notification" "default" "bell,white_check_mark" "Test notification from ${HOST} at $(date)")

    if [ "$HTTP_CODE" = "200" ]; then
        echo "[notify_run] SUCCESS: received HTTP 200 from ${NTFY_SERVER}."
        exit 0
    else
        echo "[notify_run] ERROR: ${NTFY_SERVER} returned HTTP ${HTTP_CODE}. Check NTFY_TOPIC and NTFY_TOKEN." >&2
        exit 1
    fi
fi

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    if [ "$hours" -gt 0 ]; then
        printf "%dh %dm %ds" "$hours" "$minutes" "$seconds"
    elif [ "$minutes" -gt 0 ]; then
        printf "%dm %ds" "$minutes" "$seconds"
    else
        printf "%ds" "$seconds"
    fi
}

START_TIME_HUMAN="$(date 2>/dev/null || echo "now")"
START_SEC="$(date +%s)"
CMD_STR="$*"

send_ntfy "FedMAQ: Run Started" "low" "rocket" "Host: ${HOST}
Started: ${START_TIME_HUMAN}
Command: ${CMD_STR}" >/dev/null

LOG_FILE=$(mktemp /tmp/fedmaq_run_XXXXXX.log 2>/dev/null || echo "/tmp/fedmaq_run_$$.log")
HEARTBEAT_PID=""

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_FILE"
}
trap cleanup EXIT INT TERM

if [ "${NTFY_HEARTBEAT_SEC}" -gt 0 ] 2>/dev/null; then
    # The heartbeat must not outlive the run: its output goes to /dev/null so an
    # orphan cannot hold the caller's pipe open, and on TERM it kills its own
    # in-flight sleep, which `kill "$HEARTBEAT_PID"` alone would leave running.
    (
        SLEEP_PID=""
        trap '[ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 0' TERM INT
        while true; do
            sleep "${NTFY_HEARTBEAT_SEC}" &
            SLEEP_PID=$!
            wait "$SLEEP_PID"
            SLEEP_PID=""
            ELAPSED_NOW=$(( $(date +%s) - START_SEC ))
            DURATION_NOW="$(format_duration "$ELAPSED_NOW")"
            send_ntfy "FedMAQ: Run Heartbeat" "min" "heartbeat,hourglass_flowing_sand" "Host: ${HOST}
Elapsed: ${DURATION_NOW}
Command: ${CMD_STR}" >/dev/null
        done
    ) </dev/null >/dev/null 2>&1 &
    HEARTBEAT_PID=$!
fi

set +e
"$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_SEC="$(date +%s)"
DURATION_SEC=$((END_SEC - START_SEC))
DURATION_STR="$(format_duration "$DURATION_SEC")"

if [ "$EXIT_CODE" -eq 0 ]; then
    send_ntfy "FedMAQ: Run Succeeded" "default" "white_check_mark" "Host: ${HOST}
Duration: ${DURATION_STR}
Command: ${CMD_STR}" >/dev/null
else
    ERROR_TAIL=""
    if [ -f "$LOG_FILE" ]; then
        ERROR_TAIL="$(tail -n 15 "$LOG_FILE" 2>/dev/null || true)"
    fi

    FAIL_MSG="Host: ${HOST}
Duration: ${DURATION_STR}
Exit Code: ${EXIT_CODE}
Command: ${CMD_STR}"

    if [ -n "$ERROR_TAIL" ]; then
        FAIL_MSG="${FAIL_MSG}

Error tail (last 15 lines):
${ERROR_TAIL}"
    fi

    send_ntfy "FedMAQ: Run FAILED (Exit ${EXIT_CODE})" "urgent" "x,warning" "${FAIL_MSG}" >/dev/null
fi

exit "$EXIT_CODE"
