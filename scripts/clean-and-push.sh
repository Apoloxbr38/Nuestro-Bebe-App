#!/usr/bin/env bash
set -euo pipefail

### ================================
###  Clean & Push — Nuestro-Bebé-App
###  Uso:
###    ./scripts/clean-and-push.sh "mensaje de commit"
###  Si no pasas mensaje, se usa uno por defecto.
### ================================

msg="${1:-chore: clean repo (ignore venv/site-packages) + update requirements}"

# Colores
c_green="\033[1;32m"; c_yellow="\033[1;33m"; c_red="\033[1;31m"; c_reset="\033[0m"

say() { echo -e "${c_green}▶${c_reset} $*"; }
warn(){ echo -e "${c_yellow}⚠${c_reset} $*"; }
err() { echo -e "${c_red}✖${c_reset} $*" >&2; }

# 1) Ir a la raíz del repo (carpeta del script/..)
cd "$(dirname "$0")/.." || { err "No pude entrar a la raíz del repo"; exit 1; }

# 2) Comprobaciones básicas
command -v git >/dev/null || { err "git no está instalado"; exit 1; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  err "Esto no parece un repositorio git"; exit 1;
fi

# 3) Asegurar .gitignore (añadir entradas clave sin duplicar)
say "Actualizando .gitignore…"
touch .gitignore
# Bloques que queremos asegurar
mapfile -t GI <<'EOF'
# Python / venv / cachés
.venv/
backend/.venv/
**/__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

# Build/artefactos
dist/
build/
*.log

# Node (por si acaso a futuro)
node_modules/

# SO
.DS_Store
Thumbs.db
EOF

# Añadir líneas que no existan ya
for line in "${GI[@]}"; do
  grep -Fxq "$line" .gitignore || echo "$line" >> .gitignore
done

# 4) Quitar del *índice* (no borra tus archivos locales) lo que no debe versionarse
say "Limpiando el índice (sin borrar tus archivos locales)…"
# Patrones comunes
git rm -r --cached --ignore-unmatch .venv backend/.venv **/__pycache__ 2>/dev/null || true
git rm -r --cached --ignore-unmatch *.pyc *.pyo *.pyd 2>/dev/null || true
git rm -r --cached --ignore-unmatch dist build node_modules 2>/dev/null || true

# Casos concretos que te dieron error en GitHub (binarios enormes dentro de venv)
git rm -r --cached --ignore-unmatch backend/.venv/lib/python*/site-packages/xgboost/lib/libxgboost.so 2>/dev/null || true
git rm -r --cached --ignore-unmatch backend/.venv/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null || true
git rm -r --cached --ignore-unmatch backend/.venv/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so 2>/dev/null || true

# 5) (Opcional) Actualizar requirements si el venv está activo
if [[ "${VIRTUAL_ENV:-}" != "" ]]; then
  say "Entorno virtual detectado (${VIRTUAL_ENV}). Actualizando backend/requirements.txt…"
  python -m pip freeze > backend/requirements.txt || warn "No pude regenerar requirements.txt, continúo…"
else
  warn "No hay entorno virtual activo; dejo backend/requirements.txt como está."
fi

# 6) Añadir todo y hacer commit
say "git add -A"
git add -A

# Ver si hay cambios
if git diff --cached --quiet; then
  warn "No hay cambios para commitear. Paso al push."
else
  say "git commit -m \"$msg\""
  git commit -m "$msg" || warn "Nada que commitear."
fi

# 7) Asegurar remoto y rama
origin_url="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$origin_url" ]]; then
  warn "No hay remoto 'origin' configurado."
  read -rp "URL de tu repo (ej: https://github.com/Apoloxbr38/Nuestro-Bebe-App.git): " newurl
  git remote add origin "$newurl"
  origin_url="$newurl"
  say "Remoto 'origin' configurado a $origin_url"
else
  say "Remoto origin: $origin_url"
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
say "Rama actual: ${current_branch}"

# 8) Push con upstream si hace falta
say "Empujando a GitHub…"
if git rev-parse --symbolic-full-name --quiet --verify "refs/remotes/origin/${current_branch}" >/dev/null; then
  git push origin "${current_branch}"
else
  git push -u origin "${current_branch}"
fi

echo
say "¡Listo! ✅"
echo "Repo limpio (sin .venv ni binarios pesados) y cambios empujados a 'origin/${current_branch}'."
echo -e "Si clonas en otra máquina:\n  1) python3 -m venv backend/.venv\n  2) source backend/.venv/bin/activate\n  3) pip install -r backend/requirements.txt\n  4) uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000"
