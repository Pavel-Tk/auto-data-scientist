#!/bin/bash
# ==============================================================================
# Auto Data Scientist — Launcher Script
# ==============================================================================
# Outer loop that manages wall-clock time and invokes Claude Code per iteration.
# Each iteration spawns the auto-research skill for one autonomous cycle.
#
# Usage:
#   ./launch_research.sh [BUDGET_HOURS]
#
# Examples:
#   ./launch_research.sh          # Default 8-hour budget
#   ./launch_research.sh 2        # 2-hour budget
#   ./launch_research.sh 24       # 24-hour budget
#   ./launch_research.sh 1        # 1-hour test run
#   ./launch_research.sh --status # Show progress summary
#
# To stop early: Ctrl+C (state is preserved in research_state.md)
# To resume: just run the script again (it reads iteration count and elapsed time from state)
# ==============================================================================

set -euo pipefail

# --- Status mode ---
if [ "${1:-}" = "--status" ]; then
    if [ ! -f "research_state.md" ]; then
        echo "No research_state.md found. Run ./launch_research.sh to initialize."
        exit 0
    fi
    echo "=== Auto Data Scientist — Progress ==="
    grep -E "(Score|Model|Total Experiments|Successful Experiments|Consecutive Failures)" research_state.md 2>/dev/null || echo "No experiments yet."
    echo ""
    echo "=== Hypothesis Queue ==="
    grep -c "^[0-9]\+\." research_state.md 2>/dev/null && echo "hypotheses queued" || echo "Queue empty or not found"
    echo ""
    echo "=== Disk Usage ==="
    du -sh logs/ 2>/dev/null || echo "No logs directory"
    exit 0
fi

# --- Configuration ---
BUDGET_HOURS="${1:-8}"
# Pure bash integer arithmetic (pass whole hours like 2, 4, 8, 12)
BUDGET_SECONDS=$((BUDGET_HOURS * 3600))
COOLDOWN_SECONDS=5  # Delay between iterations to avoid API hammering
ELAPSED_FILE="logs/.elapsed_seconds"  # Persistent elapsed time tracker
ITERATION_TIMEOUT=2700  # Max seconds per iteration (45 minutes)
MIN_DISK_KB=1048576  # Minimum disk space: 1GB
MAX_RETRIES=1  # Retry failed iterations once before skipping
LOG_FILE="logs/loop_$(date +%Y%m%d_%H%M%S).log"

# --- Colors for terminal output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper functions ---
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"; }

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    if [ $hours -gt 0 ]; then
        printf '%dh %dm %ds' $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf '%dm %ds' $minutes $secs
    else
        printf '%ds' $secs
    fi
}

# --- Graceful shutdown ---
cleanup() {
    echo ""
    # Save cumulative elapsed time on shutdown
    NOW=$(date +%s)
    SESSION_ELAPSED=$((NOW - SESSION_START))
    TOTAL_ELAPSED=$((PRIOR_ELAPSED + SESSION_ELAPSED))
    mkdir -p logs
    echo "$TOTAL_ELAPSED" > "$ELAPSED_FILE"
    log_info "Saved cumulative elapsed time: $(format_time $TOTAL_ELAPSED)"
    log_warn "Interrupted by user. State preserved in research_state.md."
    log_info "To resume, run: ./launch_research.sh $BUDGET_HOURS"
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Startup ---
log_header "Auto Data Scientist — Research Loop"
log_info "Budget: ${BUDGET_HOURS}h (${BUDGET_SECONDS}s)"
log_info "Working directory: $(pwd)"
log_info "Cooldown between iterations: ${COOLDOWN_SECONDS}s"
log_info "Per-iteration timeout: ${ITERATION_TIMEOUT}s"

# --- Create logs directory ---
mkdir -p logs

# --- Set up logging (tee to file for overnight review) ---
exec > >(tee -a "$LOG_FILE") 2>&1
log_info "Session log: $LOG_FILE"

# --- Load prior elapsed time (for resume support) ---
PRIOR_ELAPSED=0
if [ -f "$ELAPSED_FILE" ]; then
    PRIOR_ELAPSED=$(cat "$ELAPSED_FILE" 2>/dev/null || echo "0")
    # Validate it's a number
    if ! [[ "$PRIOR_ELAPSED" =~ ^[0-9]+$ ]]; then
        PRIOR_ELAPSED=0
    fi
    if [ "$PRIOR_ELAPSED" -gt 0 ]; then
        log_info "Resuming with $(format_time $PRIOR_ELAPSED) of prior elapsed time"
        REMAINING_BUDGET=$((BUDGET_SECONDS - PRIOR_ELAPSED))
        if [ "$REMAINING_BUDGET" -le 0 ]; then
            log_error "Budget already exhausted ($(format_time $PRIOR_ELAPSED) elapsed of ${BUDGET_HOURS}h budget)."
            log_info "Run with a larger budget or delete $ELAPSED_FILE to reset."
            exit 1
        fi
        log_info "Remaining budget: $(format_time $REMAINING_BUDGET)"
    fi
fi

# --- Require initialization via /auto-research ---
if [ ! -f "research_state.md" ]; then
    log_error "No research_state.md found."
    log_error "Initialize first by running /auto-research from inside Claude Code."
    exit 1
fi

# --- Determine starting iteration ---
# Try to extract the last iteration number from research_state.md
ITERATION=1
if grep -q "Total Experiments" research_state.md 2>/dev/null; then
    LAST_ITER=$(grep -oP 'Total Experiments.*?(\d+)' research_state.md | grep -oP '\d+$' || echo "0")
    if [ -n "$LAST_ITER" ] && [ "$LAST_ITER" -gt 0 ] 2>/dev/null; then
        ITERATION=$((LAST_ITER + 1))
        log_info "Resuming from iteration $ITERATION (found $LAST_ITER completed experiments)"
    fi
fi

# --- Record session start time ---
SESSION_START=$(date +%s)

# --- Main Loop ---
log_header "Starting Autonomous Research Loop"
log_info "Press Ctrl+C to stop at any time (state is always saved)"
echo ""

while true; do
    # --- Check time budget (including prior sessions) ---
    NOW=$(date +%s)
    SESSION_ELAPSED=$((NOW - SESSION_START))
    ELAPSED=$((PRIOR_ELAPSED + SESSION_ELAPSED))

    if [ $ELAPSED -ge $BUDGET_SECONDS ]; then
        log_header "Time Budget Exhausted"
        echo "$ELAPSED" > "$ELAPSED_FILE"
        log_ok "Completed in $(format_time $ELAPSED). Final state in research_state.md."
        break
    fi

    REMAINING=$((BUDGET_SECONDS - ELAPSED))
    BUDGET_PCT=$((ELAPSED * 100 / BUDGET_SECONDS))

    # --- Check disk space ---
    AVAILABLE_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}' || echo "999999999")
    if [ "$AVAILABLE_KB" -lt "$MIN_DISK_KB" ]; then
        log_error "Less than 1GB disk space remaining ($(( AVAILABLE_KB / 1024 ))MB). Stopping to prevent data loss."
        echo "$ELAPSED" > "$ELAPSED_FILE"
        break
    fi

    # --- Iteration header ---
    echo ""
    log_header "Iteration $ITERATION"
    log_info "Elapsed: $(format_time $ELAPSED) / ${BUDGET_HOURS}h (${BUDGET_PCT}% used)"
    log_info "Remaining: $(format_time $REMAINING)"
    log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

    # --- Snapshot experiment count before iteration (for change detection) ---
    EXPERIMENTS_BEFORE=$(grep -oP 'Total Experiments.*?(\d+)' research_state.md 2>/dev/null | grep -oP '\d+$' || echo "0")

    # --- Run iteration with retry logic ---
    ATTEMPT=0
    ITER_SUCCESS=false
    ITER_START=$(date +%s)

    while [ $ATTEMPT -le $MAX_RETRIES ]; do
        if [ $ATTEMPT -gt 0 ]; then
            log_warn "Retrying iteration $ITERATION (attempt $((ATTEMPT + 1))/$((MAX_RETRIES + 1)))..."
            sleep 3
        fi

        # stdin redirected from /dev/null: prevents CLI from ever blocking on user input
        # --allowedTools: restricts Master to non-interactive tools only (no AskUserQuestion)
        timeout $ITERATION_TIMEOUT \
            claude --allowedTools "Agent,Bash,Read,Write,Edit,Glob,Grep" \
            --skill auto-research \
            "Run one autonomous research iteration. Iteration: $ITERATION. Elapsed: ${ELAPSED}s of ${BUDGET_SECONDS}s budget ($(format_time $ELAPSED) of ${BUDGET_HOURS}h, ${BUDGET_PCT}% used). Working directory: $(pwd). This is an autonomous iteration — do NOT ask the user anything. Complete the full cycle: read state, select hypothesis, spawn worker, process results, update state." \
            < /dev/null \
            || {
                ITER_EXIT=$?
                if [ "$ITER_EXIT" -eq 124 ]; then
                    log_warn "Iteration $ITERATION timed out after ${ITERATION_TIMEOUT}s."
                else
                    log_warn "Iteration $ITERATION exited with code $ITER_EXIT."
                fi
            }

        # Check if the iteration actually advanced the experiment count
        EXPERIMENTS_AFTER=$(grep -oP 'Total Experiments.*?(\d+)' research_state.md 2>/dev/null | grep -oP '\d+$' || echo "0")

        if [ "$EXPERIMENTS_AFTER" -gt "$EXPERIMENTS_BEFORE" ]; then
            ITER_SUCCESS=true
            break
        fi

        ATTEMPT=$((ATTEMPT + 1))
    done

    ITER_END=$(date +%s)
    ITER_DURATION=$((ITER_END - ITER_START))

    if [ "$ITER_SUCCESS" = true ]; then
        log_ok "Iteration $ITERATION completed in $(format_time $ITER_DURATION)"
    else
        log_warn "Iteration $ITERATION failed after $((MAX_RETRIES + 1)) attempts ($(format_time $ITER_DURATION)). Skipping."
    fi

    # --- Save cumulative elapsed time after each iteration ---
    SESSION_ELAPSED=$((ITER_END - SESSION_START))
    echo "$((PRIOR_ELAPSED + SESSION_ELAPSED))" > "$ELAPSED_FILE"

    # --- Increment iteration ---
    ITERATION=$((ITERATION + 1))

    # --- Cooldown ---
    log_info "Cooling down for ${COOLDOWN_SECONDS}s..."
    sleep $COOLDOWN_SECONDS
done

# --- Final summary ---
echo ""
log_header "Research Loop Complete"
log_info "Total iterations attempted: $((ITERATION - 1))"
FINAL_ELAPSED=$((PRIOR_ELAPSED + $(date +%s) - SESSION_START))
echo "$FINAL_ELAPSED" > "$ELAPSED_FILE"
log_info "Total wall time (all sessions): $(format_time $FINAL_ELAPSED)"
log_info "Results: research_state.md"
log_info "Logs: logs/"
if [ -f "submission.csv" ]; then
    log_ok "Final submission: submission.csv"
else
    log_warn "No submission.csv found. Run Phase 3 manually if needed."
fi
