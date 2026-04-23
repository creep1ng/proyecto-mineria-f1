#!/bin/bash
# =============================================================================
# Script de Instalación de Dependencias - F1 Race Finish Predictor
# =============================================================================
# Uso: ./install_deps.sh
# =============================================================================

set -e

echo "=========================================="
echo "Instalando dependencias..."
echo "=========================================="

pip install streamlit pandas numpy scikit-learn matplotlib seaborn \
    imbalanced-learn xgboost optuna scipy joblib ipykernel \
    ydata-profiling phik

echo ""
echo "[OK] Dependencias instaladas correctamente."
echo ""
echo "Para ejecutar la aplicación:"
echo "  1. Genera los notebooks ejecutando:"
echo "       jupyter notebook notebooks/01_preparacion_datos.ipynb"
echo "       jupyter notebook notebooks/02_modelado_predictivo.ipynb"
echo ""
echo "  2. Ejecuta el despliegue:"
echo "       ./deploy.sh"
