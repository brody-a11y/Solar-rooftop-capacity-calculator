#!/bin/bash
# Double-click to download installed rooftop solar systems (30 kW and up) from
# city building-permit open data: San Francisco, Los Angeles, Austin, Seattle,
# Chicago, New York. Writes ~/Downloads/permits_truth.csv, which you then drag
# into 'Accuracy Test.command'. Installed systems are treated as minimums: the
# test reports how much of the tool's MaxFit typically gets built.
# Free (public data); the accuracy test afterwards uses Google/Regrid lookups
# for each site, so at most 40 sites per city are kept by default.
HOME_DIR="$HOME/.rooftop-solar"
CLI="$HOME_DIR/venv/bin/rooftop-solar"

finish() { echo; read -r -p "Press Return to close this window. " _; exit "$1"; }

if [ ! -x "$CLI" ]; then
  echo "Not installed yet. Double-click 'Install.command' first."
  finish 1
fi

# Pulled a newer version in GitHub Desktop? Update the installed tool first.
REPO="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$(find "$REPO/rooftop_solar" "$REPO/pyproject.toml" -newer "$HOME_DIR/installed.stamp" -print -quit 2>/dev/null)" ] \
   || [ ! -f "$HOME_DIR/installed.stamp" ]; then
  echo "Updating to the version you pulled..."
  "$HOME_DIR/venv/bin/python" -m pip install --quiet "$REPO" && touch "$HOME_DIR/installed.stamp"
fi


echo "Cities (sf la austin seattle chicago nyc), separated by spaces, or press Return for all:"
read -r CITIES
OUT="$HOME/Downloads/permits_truth.csv"
ARGS=(permits --out "$OUT")
for c in $CITIES; do ARGS+=(--city "$c"); done

"$CLI" "${ARGS[@]}"
status=$?
if [ $status -eq 0 ]; then
  echo
  echo "Next: double-click 'Accuracy Test.command' and drag in $OUT"
fi
finish $status
