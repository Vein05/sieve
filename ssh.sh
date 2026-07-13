#!/usr/bin/env bash
# SSH into lab GPU server
#
# IMPORTANT: ALL work MUST stay inside /mnt/data0/sugam/
# Do NOT write files outside this directory. Ever.
# Do NOT touch /mnt/data0/data or other users' folders.
#
# Your folder:      /mnt/data0/sugam
# Shared datasets:  /mnt/data0/data (read-only)
#
# Rules:
#   - All installs, scripts, outputs go in /mnt/data0/sugam/
#   - Clean up unused images and model weights to reduce memory
#   - Keep datasets, models, and any actively used data

HOST="10.0.42.10"
USER="sugam"
PASS="su@8070"

_ssh() {
    if command -v sshpass &>/dev/null; then
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "${USER}@${HOST}" "$@"
    else
        echo "Password: $PASS"
        echo "(install sshpass for automatic login: brew install sshpass / apt install sshpass)"
        ssh -o StrictHostKeyChecking=no "${USER}@${HOST}" "$@"
    fi
}

# Always show GPU status before running any command
echo "=== GPU Status ==="
_ssh nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r name util used total; do
        echo "  $name | Util:$util | Mem:$used /$total"
    done
_ssh nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r pid pname mem; do
        echo "  PID $pid |$pname |$mem"
    done
echo "=================="

# Run the actual command if one was given
if [ $# -gt 0 ]; then
    _ssh "$@"
fi
