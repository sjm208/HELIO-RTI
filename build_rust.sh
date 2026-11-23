#!/bin/bash
# Build helio_core_rs, the Rust kernels, into the Python that builds or runs Helio.
#
#   ./build_rust.sh                 build and install (release), then check it
#   PYTHON=/path/to/python3 ./build_rust.sh
#   TEST=1 ./build_rust.sh          also run the Rust unit tests first
#
# Needs a Rust toolchain (https://rustup.rs: `curl https://sh.rustup.rs -sSf | sh`).
# Without one Helio still runs, on its NumPy kernels, slower.  The built module
# is a normal extension module for this Python and this CPU (arm64 on Apple silicon).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
    echo "NOTE: no Rust toolchain (cargo) found - Helio will use its NumPy kernels."
    echo "      Install it with:  curl https://sh.rustup.rs -sSf | sh   and run this again."
    [ "${REQUIRE_RUST:-0}" = "1" ] && exit 1
    exit 0
fi
echo "Rust: $(rustc --version) · Python: $("$PYTHON" -c 'import sys,platform; print(sys.version.split()[0], platform.machine())')"

"$PYTHON" -m pip install --quiet --upgrade "maturin>=1.9,<2"
# NumPy only when it is missing: upgrading it here moved working setups past what
# numba and the tested OpenCV allow (requirements.txt lists the tested pairs).
"$PYTHON" -c "import numpy" 2>/dev/null || "$PYTHON" -m pip install --quiet "numpy>=2.0,<3"
if [ "${TEST:-0}" = "1" ]; then
    (cd helio_core_rs && cargo test --release)
fi
# `maturin develop` installs into the active environment (a virtualenv); PyInstaller then
# finds the module there.  Outside a virtualenv build a wheel and install that.
if "$PYTHON" -c 'import sys; sys.exit(0 if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else 1)'; then
    (cd helio_core_rs && "$PYTHON" -m maturin develop --release)
else
    rm -rf helio_core_rs/wheelhouse
    (cd helio_core_rs && "$PYTHON" -m maturin build --release --out wheelhouse)
    # --no-deps: only the module itself is replaced (a forced reinstall otherwise
    # reinstalls NumPy too, at its newest, past what numba and the tested OpenCV allow)
    "$PYTHON" -m pip install --quiet --force-reinstall --no-deps helio_core_rs/wheelhouse/helio_core_rs-*.whl
fi

# A quick proof it loads and computes: a tiny least-squares problem against NumPy.
"$PYTHON" - <<'PY'
import numpy as np, helio_core_rs as rs
L = np.array([[0.3, 0.1, 0.95], [-0.4, 0.2, 0.89], [0.1, -0.5, 0.86], [0.5, 0.4, 0.77]])
n = np.array([0.2, -0.1, 0.97]); n /= np.linalg.norm(n)
I = np.maximum(L @ n, 0)[:, None].astype(np.float32).repeat(5, axis=1)
got = rs.lls_normals(np.linalg.pinv(L), np.ascontiguousarray(I))
assert np.allclose(got, n, atol=1e-5), got
print(f"helio_core_rs {rs.__version__} loads and agrees with NumPy ({rs.threads()} threads)")
PY
