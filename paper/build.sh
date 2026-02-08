#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
TEX_DIR="$ROOT_DIR/paper/tex"
OUT_DIR="$ROOT_DIR/paper/final"

mkdir -p "$OUT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -output-directory="$OUT_DIR" -jobname=final "$TEX_DIR/main.tex"
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex not found. Please install a LaTeX distribution." >&2
    exit 1
  fi
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$OUT_DIR" "$TEX_DIR/main.tex"
  (cd "$OUT_DIR" && BIBINPUTS="$TEX_DIR" bibtex final)
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$OUT_DIR" "$TEX_DIR/main.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$OUT_DIR" "$TEX_DIR/main.tex"
fi

echo "Built $OUT_DIR/final.pdf"
