#!/bin/bash
# Run every test script in the tree, once with helio_core_rs (if built) and
# once forced onto the NumPy kernels (HELIO_NO_RUST=1), so the fallback path
# keeps working while the Rust kernels land.
#
#   ./run_tests.sh              both passes
#   ./run_tests.sh numpy        NumPy kernels only
#   ./run_tests.sh rust         helio_core_rs only (fails if it is not built)
#
# Qt tests run offscreen.  Rust unit tests: (cd helio_core_rs && cargo test).
VP=${PYTHON:-python3}
TREE="$(cd "$(dirname "$0")" && pwd)"
cd "$TREE" || exit 1
export PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export HELIO_PREFS="${TMPDIR:-/tmp}/helio_prefs_test_$$.json"
OUT=$(mktemp)
trap 'rm -f "$HELIO_PREFS" "$OUT"' EXIT

MODE=${1:-both}
if [ "$MODE" = rust ] || [ "$MODE" = both ]; then
    if ! $VP -c "from helio_core_rs import lls_normals" 2>/dev/null; then
        [ "$MODE" = rust ] && { echo "helio_core_rs is not built: cd helio_core_rs && maturin develop --release"; exit 1; }
        echo "helio_core_rs not built - running the NumPy pass only"
        MODE=numpy
    fi
fi

FAILED=()
run_pass() {
    local label=$1; shift
    echo "════ $label ════"
    for t in test_*.py tests/test_*.py; do      # beside the code, or in tests/ (the public copy)
        [ -f "$t" ] || continue
        if env "$@" $VP "$t" > "$OUT" 2>&1; then
            printf '  ok    %s\n' "$t"
        else
            printf '  FAIL  %s\n' "$t"
            tail -5 "$OUT" | sed 's/^/        /'
            FAILED+=("$label:$t")
        fi
    done
    :
}

case "$MODE" in
    numpy) run_pass "NumPy kernels" HELIO_NO_RUST=1 ;;
    rust)  run_pass "helio_core_rs" HELIO_NO_RUST=0 ;;
    both)  run_pass "helio_core_rs" HELIO_NO_RUST=0; run_pass "NumPy kernels" HELIO_NO_RUST=1 ;;
    *) echo "usage: $0 [both|numpy|rust]"; exit 2 ;;
esac

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "all passed"; else printf 'failed: %s\n' "${FAILED[@]}"; exit 1; fi
