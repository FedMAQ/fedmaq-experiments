#!/usr/bin/env bash
# ==============================================================================
# scripts/notify_run.sh
#
# Execution wrapper for long-running FedMAQ commands and sweeps on JupyterHub.
# Streams output live to the console while sending start, success, or failure
# push notifications to an authenticated ntfy.sh topic.
# ==============================================================================

set -eu

# Load environment variables from .env if present
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

show_help() {
    cat << 'EOF'
Usage: ./scripts/notify_run.sh [--test] <command> [args...]

Wraps any long-running command, streaming live output to the console while
dispatching start, success, or failure push notifications to an authenticated
ntfy.sh topic.

Requirements:
  NTFY_TOPIC and NTFY_TOKEN must be set in your .env file or environment.
    NTFY_TOPIC="fedmaq-thesis-bunyi"
    NTFY_TOKEN="tk_..."

Options:
  --test        Send a test notification to verify credentials and connectivity.
  -h, --help    Show this help message.

Examples:
  ./scripts/notify_run.sh --test
  ./scripts/notify_run.sh ./.venv/bin/python scripts/run_matrix.py --matrix baseline_tuning_wide -o ray.temp_dir=/tmp/ray-cjb -o ray.object_store_gb=4
EOF
}

# Handle help
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
fi

# Validate credentials fail-fast
if [ -z "${NTFY_TOPIC:-}" ] || [ -z "${NTFY_TOKEN:-}" ]; then
    echo "[notify_run] ERROR: NTFY_TOPIC or NTFY_TOKEN is not set." >&2
    echo "[notify_run] Please set them in your .env file or environment:" >&2
    echo "  NTFY_TOPIC=\"fedmaq-thesis-bunyi\"" >&2
    echo "  NTFY_TOKEN=\"tk_...\"" >&2
    exit 1
fi

# Handle self-test mode
if [ "${1:-}" = "--test" ]; then
    HOST="$(hostname 2>/dev/null || echo "unknown-host")"
    echo "[notify_run] Dispatching test notification to https://ntfy.sh/${NTFY_TOPIC}..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://ntfy.sh/${NTFY_TOPIC}" \
        -H "Authorization: Bearer ${NTFY_TOKEN}" \
        -H "Title: FedMAQ: Test Notification" \
        -H "Priority: default" \
        -H "Tags: bell,white_check_mark" \
        -d "Self-test notification from ${HOST} at $(date)")

    if [ "$HTTP_CODE" = "200" ]; then
        echo "[notify_run] SUCCESS: received HTTP 200 from ntfy.sh."
        exit 0
    else
        echo "[notify_run] ERROR: ntfy.sh returned HTTP ${HTTP_CODE}. Check NTFY_TOPIC and NTFY_TOKEN." >&2
        exit 1
    fi
fi

if [ $# -eq 0 ]; then
    echo "[notify_run] ERROR: No command specified." >&2
    show_help >&2
    exit 1
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

HOST="$(hostname 2>/dev/null || echo "unknown-host")"
START_TIME_HUMAN="$(date 2>/dev/null || echo "now")"
START_SEC="$(date +%s)"
CMD_STR="$*"

# Dispatch start notification
curl -s -f -X POST "https://ntfy.sh/${NTFY_TOPIC}" \
    -H "Authorization: Bearer ${NTFY_TOKEN}" \
    -H "Title: FedMAQ: Run Started" \
    -H "Priority: low" \
    -H "Tags: rocket" \
    -d "Host: ${HOST}
Started: ${START_TIME_HUMAN}
Command: ${CMD_STR}" >/dev/null 2>&1 || true

LOG_FILE=$(mktemp /tmp/fedmaq_run_XXXXXX.log 2>/dev/null || echo "/tmp/fedmaq_run_$$.log")

# Stream live output while capturing to temporary log
set +e
"$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_SEC="$(date +%s)"
DURATION_SEC=$((END_SEC - START_SEC))
DURATION_STR="$(format_duration "$DURATION_SEC")"

if [ "$EXIT_CODE" -eq 0 ]; then
    curl -s -f -X POST "https://ntfy.sh/${NTFY_TOPIC}" \
        -H "Authorization: Bearer ${NTFY_TOKEN}" \
        -H "Title: FedMAQ: Run Succeeded" \
        -H "Priority: default" \
        -H "Tags: white_check_mark" \
        -d "Host: ${HOST}
Duration: ${DURATION_STR}
Command: ${CMD_STR}" >/dev/null 2>&1 || true
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

    curl -s -f -X POST "https://ntfy.sh/${NTFY_TOPIC}" \
        -H "Authorization: Bearer ${NTFY_TOKEN}" \
        -H "Title: FedMAQ: Run FAILED (Exit ${EXIT_CODE})" \
        -H "Priority: urgent" \
        -H "Tags: x,warning" \
        -d "${FAIL_MSG}" >/dev/null 2>&1 || true
fi

rm -f "$LOG_FILE"
exit "$EXIT_CODE"
