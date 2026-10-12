#!/usr/bin/env bash
# Set up a Forenote development machine with one command.
#
# Usage (in a fresh clone):
#   scripts/bootstrap-dev.sh [--branch NAME] [--models-from HOST] [--claude-from HOST] [--no-tests]
#
#   --branch NAME       check out this branch after a fetch
#   --models-from HOST  copy the models from HOST over ssh (rsync), then check them
#   --claude-from HOST  copy ~/.claude/CLAUDE.md (the user instructions) from HOST
#   --no-tests          do not run the test suite at the end
#
# Without --models-from, the script downloads the models of the machine tier.
# The script asks for the sudo password only to install missing system packages.

set -euo pipefail

BRANCH=""
MODELS_FROM=""
CLAUDE_FROM=""
RUN_TESTS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --branch) BRANCH="$2"; shift 2 ;;
        --models-from) MODELS_FROM="$2"; shift 2 ;;
        --claude-from) CLAUDE_FROM="$2"; shift 2 ;;
        --no-tests) RUN_TESTS=0; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

step "Check the machine"
[ "$(uname -m)" = "x86_64" ] || { echo "Forenote needs Linux x86_64." >&2; exit 1; }
grep -qw avx2 /proc/cpuinfo || { echo "This CPU does not support AVX2. Forenote needs AVX2." >&2; exit 1; }
echo "OK: x86_64 with AVX2, $(nproc) CPU threads, $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo) RAM"

step "Install the system tools (git, rsync, cmake, g++)"
missing=()
for tool in git rsync cmake g++ curl; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    if command -v apt-get >/dev/null; then
        pkgs=()
        for tool in "${missing[@]}"; do
            case "$tool" in g++) pkgs+=(g++) ;; *) pkgs+=("$tool") ;; esac
        done
        echo "Installing: ${pkgs[*]}"
        sudo apt-get update -q
        sudo apt-get install -y -q "${pkgs[@]}"
    else
        echo "Install these tools with your package manager, then run the script again: ${missing[*]}" >&2
        exit 1
    fi
else
    echo "OK: all tools are present"
fi

step "Install uv"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [ -n "$BRANCH" ]; then
    step "Check out $BRANCH"
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only
fi

step "Install the Python dependencies (builds llama-cpp-python, about 1 minute)"
uv sync

MODELS_DIR="$(uv run python -c 'from forenote.paths import Paths; print(Paths.from_env().models_dir)')"
mkdir -p "$MODELS_DIR"

if [ -n "$MODELS_FROM" ]; then
    step "Copy the models from $MODELS_FROM"
    rsync -a --partial --info=progress2 "$MODELS_FROM:.local/share/forenote/models/" "$MODELS_DIR/"
fi

step "Check the models (and download what is missing)"
uv run python - <<'EOF'
from forenote.hw import detect
from forenote.models.download import ModelStore
from forenote.models.manifest import for_tier, load_manifest
from forenote.paths import Paths

store = ModelStore(Paths.from_env().models_dir)
entries = load_manifest()
tier = detect().tier or "low"
wanted = {e.id for e in for_tier(entries, tier)}
wanted.add("gemma-4-e2b-q4_0")  # the model tests use it
# A model that is already present is checked too (the SHA-256 of each file).
wanted |= {e.id for e in entries if store.path(e).is_dir()}
for e in entries:
    if e.id in wanted:
        print(f"{e.id}: checking", flush=True)
        store.install(e)  # verifies the SHA-256, downloads missing or bad files
print(f"Tier {tier}: models ready in {store.models_dir}")
EOF

if [ -n "$CLAUDE_FROM" ]; then
    step "Copy the Claude Code user instructions from $CLAUDE_FROM"
    mkdir -p "$HOME/.claude"
    if [ -f "$HOME/.claude/CLAUDE.md" ]; then
        cp "$HOME/.claude/CLAUDE.md" "$HOME/.claude/CLAUDE.md.bak.$(date +%s)"
        echo "Saved the old file as a backup."
    fi
    rsync -a "$CLAUDE_FROM:.claude/CLAUDE.md" "$HOME/.claude/CLAUDE.md"
fi

if [ "$RUN_TESTS" = 1 ]; then
    step "Run the tests"
    uv run pytest -q
fi

step "Done"
cat <<EOF
Start Claude Code in $ROOT and say:
  "Continue the work from docs/superpowers/notes/2026-10-12-handoff.md."
EOF
