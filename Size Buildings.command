#!/bin/bash
# Double-click, then drag one of these into the window:
#   a spreadsheet (.csv) of addresses or latitude/longitude - roofs are found automatically
#   a Google Earth .kml/.kmz (or .geojson) of roofs you drew
# Results are written next to the input file:
#   <name>_results.csv            sizes per property (opens in Excel or Numbers)
#   <name>_results_buildings.csv  sizes per building (address-list runs only)
#   <name>_layouts.kml            roof outlines and panels (double-click to open in Google Earth)
# Optional:
#   calibration.json in the same folder as the input is applied automatically.
#   A Google Solar API key saved in ~/.rooftop-solar/google_api_key turns on the Google cross-check.
HOME_DIR="$HOME/.rooftop-solar"
CLI="$HOME_DIR/venv/bin/rooftop-solar"

finish() { echo; read -r -p "Press Return to close this window. " _; exit "$1"; }

if [ ! -x "$CLI" ]; then
  echo "Not installed yet. Double-click 'Install.command' first."
  finish 1
fi

echo "Drag your address spreadsheet (.csv) or Google Earth file (.kml/.kmz) into this window, then press Return:"
read -r IN
# Undo the escaping macOS adds when a file is dragged into Terminal.
IN="$(printf '%s' "$IN" | sed -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'$/\1/" -e 's/\\\(.\)/\1/g')"
case "$IN" in "~/"*) IN="$HOME/${IN#\~/}" ;; esac  # typed paths starting with ~
if [ ! -f "$IN" ]; then
  echo "Can't find that file: $IN"
  finish 1
fi

DIR="$(dirname "$IN")"
BASE="$(basename "$IN")"
BASE="${BASE%.*}"
case "$(printf '%s' "$IN" | tr '[:upper:]' '[:lower:]')" in
  *.csv)
    ARGS=(size-sites --sites "$IN")
    [ -s "$HOME_DIR/google_api_key" ] && ARGS+=(--geocoder google)
    ;;
  *) ARGS=(size --buildings "$IN") ;;
esac
ARGS+=(--out "$DIR/${BASE}_results.csv" --layouts "$DIR/${BASE}_layouts.kml")

if [ -f "$DIR/calibration.json" ]; then
  echo "Using calibration.json from the same folder."
  ARGS+=(--calibration "$DIR/calibration.json")
fi
if [ -s "$HOME_DIR/google_api_key" ]; then
  echo "Google cross-check on (billed per new building)."
  export GOOGLE_SOLAR_API_KEY="$(tr -d '[:space:]' < "$HOME_DIR/google_api_key")"
  ARGS+=(--google --google-cache "$HOME_DIR/google_cache")
fi

if "$CLI" "${ARGS[@]}"; then
  open "$DIR"
  finish 0
fi
finish 1
