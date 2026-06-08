#!/usr/bin/env bash
# Find AI-ism candidates in branch-changed files: prose tells left by AI agents.
# A helper for the gbdoc workflow (.claude/skills/gbdoc/SKILL.md). Use, adjust,
# copy/paste, etc. as necessary; this finds candidates, judgment does the rest.
#
# Scans text files changed on the current branch (vs a base ref) for:
#   1. The dash family: em-dash, en-dash, and "--" used as a separator.
#   2. Stray unicode: bytes outside printable ASCII (smart quotes, ellipsis,
#      arrows, ...).
#
# Output is `file:line:content` so hits are jump-to-able. It greps the changed
# files directly, so PRE-EXISTING hits in those files show up too. Only touch
# prose the branch actually changed; leave intentional dashes/unicode alone.
# RST underlines and "# ---" banners are already skipped, "--flag" CLI usage is
# not matched, and some files (e.g. README.md) use unicode on purpose.
#
# Uses plain `grep -E` so it works with BSD grep (macOS) and GNU grep alike;
# no PCRE / `-P` needed. The em-dash and en-dash in `dash_re` below are literal
# on purpose; they are the characters being hunted.
#
# Usage:
#   scripts/find_aiisms.sh [base-ref] [path-filter]
#   scripts/find_aiisms.sh                          # vs main, whole repo
#   scripts/find_aiisms.sh main graphblas/core/     # vs main, scoped subtree
#   scripts/find_aiisms.sh --strict                 # exit 1 if anything is found
set -euo pipefail

strict=0
if [ "${1:-}" = "--strict" ]; then
    strict=1
    shift
fi
base="${1:-main}"
path_filter="${2:-}"

if ! git rev-parse --verify --quiet "$base" >/dev/null; then
    echo "Unknown base ref: ${base}" >&2
    exit 2
fi

# Collect text-y files changed on the branch (drop deleted files).
files=()
while IFS= read -r f; do
    files+=("$f")
done < <(
    if [ -n "$path_filter" ]; then
        git diff "${base}...HEAD" --name-only --diff-filter=d -- "$path_filter"
    else
        git diff "${base}...HEAD" --name-only --diff-filter=d
    fi | grep -E '\.(py|rst|md|toml|ya?ml|sh|cfg)$' || true
)

if [ "${#files[@]}" -eq 0 ]; then
    echo "No changed text files vs ${base}."
    exit 0
fi

# Dash family: em-dash, en-dash, " -- " separator, or "word--" jammed form.
# Pure-punctuation lines (RST underlines, "# ---" banners) and "--flag" CLI
# usage are naturally skipped: none have an alnum-adjacent or space-bounded
# "--", nor a real em/en-dash character.
dash_re='—|–| -- |[[:alnum:]]--'
# Stray bytes: anything outside printable ASCII, allowing tab and CR. Run under
# LC_ALL=C so grep matches byte-wise and the check stays locale-independent.
_ws="$(printf '\t\r')"
unicode_re="[^${_ws} -~]"

scan() {  # $1 = label, $2 = regex, $3 = "C" to grep under LC_ALL=C
    local label="$1" re="$2" loc="$3" f m hits=""
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        if [ "$loc" = "C" ]; then
            m="$(LC_ALL=C grep -nE "$re" "$f" 2>/dev/null || true)"
        else
            m="$(grep -nE "$re" "$f" 2>/dev/null || true)"
        fi
        if [ -n "$m" ]; then
            hits="${hits}$(printf '%s\n' "$m" | sed "s|^|${f}:|")"$'\n'
        fi
    done
    echo "=== ${label} ==="
    if [ -n "$hits" ]; then
        printf '%s' "$hits"
        echo "($(printf '%s' "$hits" | grep -c . || true) candidate line(s))"
        return 1
    fi
    echo "(none)"
    return 0
}

rc=0
scan "dash family (em/en-dash, '--' as separator)" "$dash_re" "" || rc=1
echo
scan "stray unicode (non-ASCII bytes)" "$unicode_re" "C" || rc=1
echo
echo "Note: results include any pre-existing hits in changed files. Only touch"
echo "prose the branch changed; leave intentional dashes/unicode alone."
echo "See .claude/skills/gbdoc/SKILL.md for the full workflow."

if [ "$strict" = "1" ]; then
    exit "$rc"
fi
exit 0
