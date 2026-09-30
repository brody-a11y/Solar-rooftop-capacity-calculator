#!/bin/bash
# Double-click to install or update the Rooftop Solar Calculator.
# The Python environment goes in ~/.rooftop-solar, outside Google Drive, so Drive
# doesn't have to sync thousands of library files.
cd "$(dirname "$0")" || exit 1
HOME_DIR="$HOME/.rooftop-solar"
VENV="$HOME_DIR/venv"

finish() { echo; read -r -p "Press Return to close this window. " _; exit "$1"; }

# macOS ships Python 3.9 as `python3`, so look for a newer install by name and in
# the places the python.org and Homebrew installers use.
ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
PY=""
for c in python3.14 python3.13 python3.12 python3.11 python3.10 \
         /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 \
         /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  p="$(command -v "$c" 2>/dev/null)"
  if [ -n "$p" ] && ok "$p"; then PY="$p"; break; fi
done
if [ -z "$PY" ]; then
  echo "This needs Python 3.10 or newer."
  echo "Install it from https://www.python.org/downloads/ and then double-click Install again."
  command -v python3 >/dev/null && echo "(Found only: $(python3 --version 2>&1))"
  finish 1
fi

echo "Installing with $("$PY" --version)..."
mkdir -p "$HOME_DIR"
"$PY" -m venv "$VENV" || finish 1
"$VENV/bin/python" -m pip install --quiet --upgrade pip || finish 1
"$VENV/bin/python" -m pip install --quiet ".[dev]" || finish 1

echo "Running self-test..."
if "$VENV/bin/python" -m pytest -q -p no:cacheprovider; then
  echo
  echo "Installed. Double-click 'Size Buildings.command' to run it."
  finish 0
fi
echo "Self-test failed. Send the output above to whoever set this up."
finish 1
