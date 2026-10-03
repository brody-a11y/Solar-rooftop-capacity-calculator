#!/bin/bash
# Double-click, then drag in a truth file of known max-fit designs
# (truth_drive.json from accuracy_test.zip, or a CSV with columns
# name, address, latitude, longitude, true_kw, module_w).
# Uses the Google key saved in ~/.rooftop-solar/google_api_key if present.
# Writes <name>_accuracy.csv and <name>_accuracy.kml next to the truth file.
HOME_DIR="$HOME/.rooftop-solar"
CLI="$HOME_DIR/venv/bin/rooftop-solar"

finish() { echo; read -r -p "Press Return to close this window. " _; exit "$1"; }

# Find the file the user dragged in or typed. A bare name is looked up in
# Downloads, Desktop and Documents (also when Terminal dropped its first letter).
find_input() {
  local p="$1" d m
  [ -f "$p" ] && { printf '%s' "$p"; return 0; }
  case "$p" in */*) return 1 ;; esac
  for d in "$HOME/Downloads" "$HOME/Desktop" "$HOME/Documents"; do
    [ -f "$d/$p" ] && { printf '%s' "$d/$p"; return 0; }
  done
  for d in "$HOME/Downloads" "$HOME/Desktop" "$HOME/Documents"; do
    m="$(ls -t "$d"/*"$p" 2>/dev/null | head -1)"
    [ -n "$m" ] && [ -f "$m" ] && { printf '%s' "$m"; return 0; }
  done
  return 1
}

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


echo "Drag the truth file (e.g. truth_drive.json) into this window, then press Return:"
read -r IN
IN="$(printf '%s' "$IN" | sed -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'$/\1/" -e 's/\\\(.\)/\1/g')"
case "$IN" in "~/"*) IN="$HOME/${IN#\~/}" ;; esac  # typed paths starting with ~
FOUND="$(find_input "$IN")"
while [ -z "$FOUND" ]; do
  echo "Can't find that file: $IN"
  echo "Drag the file itself from Finder into this window (or type its name if it's in Downloads), then press Return:"
  read -r IN
  IN="$(printf '%s' "$IN" | sed -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'$/\1/" -e 's/\\\(.\)/\1/g')"
  case "$IN" in "~/"*) IN="$HOME/${IN#\~/}" ;; esac
  [ -z "$IN" ] && finish 1
  FOUND="$(find_input "$IN")"
done
IN="$FOUND"
echo "Using $IN"

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
