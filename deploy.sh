#!/bin/bash
# =============================================================================
# Script de Despliegue - F1 Race Finish Predictor
# =============================================================================
# Este script configura el entorno y ejecuta la aplicación Streamlit.
# Uso: ./deploy.sh [port]
# =============================================================================

set -e

# Configuración
PORT="${1:-8501}"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$APP_DIR")"

echo "=========================================="
echo "F1 Race Finish Predictor - Deployment"
echo "=========================================="
echo ""

# Verificar que existe el modelo
if [ ! -f "$PROJECT_DIR/models/best_model_pipe.pkl" ]; then
    echo "ERROR: No se encontró el modelo en $PROJECT_DIR/models/best_model_pipe.pkl"
    echo "Ejecuta primero los notebooks de preparación y modelado."
    exit 1
fi

echo "[OK] Modelo encontrado: $PROJECT_DIR/models/best_model_pipe.pkl"
echo ""

# Verificar dependencias
echo "Verificando dependencias..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "Streamlit no está instalado. Instalando..."
    pip install streamlit pandas joblib numpy
}
echo "[OK] Dependencias verificadas."
echo ""

# Crear directorio para logs si no existe
mkdir -p "$PROJECT_DIR/logs"

# Ejecutar Streamlit
echo "=========================================="
echo "Iniciando aplicación Streamlit..."
echo "Puerto: $PORT"
echo "URL: http://localhost:$PORT"
echo "=========================================="
echo ""
echo "Presiona Ctrl+C para detener."
echo ""

cd "$PROJECT_DIR"
streamlit run "$APP_DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
