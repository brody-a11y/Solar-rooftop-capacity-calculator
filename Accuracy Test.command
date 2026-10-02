#!/bin/bash
# Double-click, then drag in a truth file of known max-fit designs
# (truth_drive.json from accuracy_test.zip, or a CSV with columns
# name, address, latitude, longitude, true_kw, module_w).
# Uses the Google key saved in ~/.rooftop-solar/google_api_key if present.
# Writes <name>_accuracy.csv and <name>_accuracy.kml next to the truth file.
HOME_DIR="$HOME/.rooftop-solar"
CLI="$HOME_DIR/venv/bin/rooftop-solar"

finish() { echo; read -r -p "Press Return to close this window. " _; exit "$1"; }

if [ ! -x "$CLI" ]; then
  echo "Not installed yet. Double-click 'Install.command' first."
  finish 1
fi

echo "Drag the truth file (e.g. truth_drive.json) into this window, then press Return:"
read -r IN
IN="$(printf '%s' "$IN" | sed -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'$/\1/" -e 's/\\\(.\)/\1/g')"
case "$IN" in "~/"*) IN="$HOME/${IN#\~/}" ;; esac  # typed paths starting with ~
if [ ! -f "$IN" ]; then
  echo "Can't find that file: $IN"
  finish 1
fi

DIR="$(dirname "$IN")"
BASE="$(basename "$IN")"
BASE="${BASE%.*}"
ARGS=(accuracy --truth "$IN" --out "$DIR/${BASE}_accuracy.csv" --layouts "$DIR/${BASE}_accuracy.kml")
if [ -s "$HOME_DIR/google_api_key" ]; then
  echo "Google Solar and Geocoding on (billed per new building or address)."
  export GOOGLE_SOLAR_API_KEY="$(tr -d '[:space:]' < "$HOME_DIR/google_api_key")"
  ARGS+=(--google --google-cache "$HOME_DIR/google_cache" --geocoder google)
else
  echo "No Google key saved; running footprint-only."
fi

if [ -s "$HOME_DIR/google_api_key" ]; then
  echo "Satellite snapshots (optional): type site names separated by ; (e.g. Garrison;Alexander), or press Return to skip:"
  read -r SNAP
  [ -n "$SNAP" ] && ARGS+=(--snapshots "$SNAP")
fi

"$CLI" "${ARGS[@]}"
finish $?
