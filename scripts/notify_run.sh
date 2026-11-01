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
sending status and duration notifications to the configured ntfy topic.

Requirements:
  NTFY_TOPIC and NTFY_TOKEN must be set in your .env file or environment.
    NTFY_TOPIC="replace-with-a-random-topic-name"
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
    echo "[notify_run] Set NTFY_TOPIC and NTFY_TOKEN in .env or the environment." >&2
    echo "[notify_run] Use a random topic name; see docs/agents/ntfy-notifications.md." >&2
    echo "  NTFY_TOPIC=\"replace-with-a-random-topic-name\"" >&2
    echo "  NTFY_TOKEN=\"tk_...\"" >&2
    exit 1
fi

send_ntfy() {
    local title="$1"
    local priority="$2"
    local tags="$3"
    local body="$4"
    local config_file
    local http_code

    if ! config_file="$(umask 077; mktemp)"; then
        printf "000"
        return 0
    fi
    if ! {
        printf 'url = "%s/%s"\n' "$NTFY_SERVER" "$NTFY_TOPIC"
        printf 'request = "POST"\n'
        printf 'max-time = 20\n'
        printf 'silent\n'
        printf 'output = "/dev/null"\n'
        printf 'write-out = "%%{http_code}"\n'
        printf 'header = "Authorization: Bearer %s"\n' "$NTFY_TOKEN"
        printf 'header = "Title: %s"\n' "$title"
        printf 'header = "Priority: %s"\n' "$priority"
        printf 'header = "Tags: %s"\n' "$tags"
        printf 'data = "%s"\n' "$body"
    } > "$config_file"; then
        rm -f "$config_file"
        printf "000"
        return 0
    fi

    if ! http_code="$(curl --config "$config_file" 2>/dev/null)"; then
        [ -n "$http_code" ] || http_code="000"
    fi
    rm -f "$config_file"
    printf "%s" "$http_code"
}

if [ "${1:-}" = "--test" ]; then
    echo "[notify_run] Dispatching test notification to ${NTFY_SERVER}..."
    HTTP_CODE=$(send_ntfy "FedMAQ: Test Notification" "default" "bell,white_check_mark" "Test notification")

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

START_SEC="$(date +%s)"
send_ntfy "FedMAQ: Run Started" "low" "rocket" "Run started" >/dev/null

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
    # curl is capped by --max-time, so a TERM during a send still returns promptly.
    (
        SLEEP_PID=""
        trap '[ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 0' TERM
        while true; do
            sleep "${NTFY_HEARTBEAT_SEC}" &
            SLEEP_PID=$!
            wait "$SLEEP_PID"
            SLEEP_PID=""
            ELAPSED_NOW=$(( $(date +%s) - START_SEC ))
            DURATION_NOW="$(format_duration "$ELAPSED_NOW")"
            send_ntfy "FedMAQ: Run Heartbeat" "min" "heartbeat,hourglass_flowing_sand" "Elapsed: ${DURATION_NOW}" >/dev/null
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
    send_ntfy "FedMAQ: Run Succeeded" "default" "white_check_mark" "Duration: ${DURATION_STR}" >/dev/null
else
    send_ntfy "FedMAQ: Run Failed" "urgent" "x,warning" "Duration: ${DURATION_STR}" >/dev/null
fi

exit "$EXIT_CODE"
