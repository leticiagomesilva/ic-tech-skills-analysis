#!/usr/bin/env bash
# Converte _src/NN_*.py (jupytext, formato percent) em notebooks e executa.
# Uso: bash construir.sh            (todos)   |   bash construir.sh 00 02   (selecionados)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
alvos=("$@")
for src in _src/*.py; do
  nome=$(basename "$src" .py)
  if [ ${#alvos[@]} -gt 0 ]; then
    ok=0; for a in "${alvos[@]}"; do [[ "$nome" == "$a"* ]] && ok=1; done
    [ $ok -eq 1 ] || continue
  fi
  echo ">> $nome"
  $PY -m jupytext --quiet --to ipynb --set-kernel tech-skills "$src" -o "$nome.ipynb"
  $PY executar.py "$nome.ipynb"
done
echo "ok"
