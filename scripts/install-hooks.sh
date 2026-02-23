#!/bin/bash
# Install git hooks for PHI/credential protection
# Run once after cloning: bash scripts/install-hooks.sh
#
# Enforces Prime Directive #0 from .claude/instructions.md

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
    echo "Error: not inside a git repository."
    exit 1
fi

HOOK_SRC="$REPO_ROOT/scripts/hooks/pre-commit"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"

if [ ! -f "$HOOK_SRC" ]; then
    echo "Error: $HOOK_SRC not found."
    exit 1
fi

if [ -f "$HOOK_DST" ] && [ ! -L "$HOOK_DST" ]; then
    echo "Existing pre-commit hook found. Backing up to pre-commit.bak"
    mv "$HOOK_DST" "$HOOK_DST.bak"
fi

ln -sf "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_SRC"

echo "Pre-commit hook installed (symlinked to scripts/hooks/pre-commit)."
echo "PHI and credential protection is active."
