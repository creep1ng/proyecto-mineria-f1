# Proyecto Final de Minería de Datos

## Predicción de Finalización de Carrera en Fórmula 1

**Metodología:** CRISP-DM  
**Dataset:** `f1_final_dataset.csv` (23,777 registros, 30 variables)  
**Variable objetivo:** `finished` (1 = terminó, 0 = no terminó)  
**Línea base:** P = 60%

---

## 📁 Estructura del Proyecto

```
├── app/
│   └── app.py                          # Aplicación Streamlit para despliegue
├── models/
│   └── best_model_pipe.pkl             # Pipeline completo del mejor modelo
├── notebooks/
│   ├── 01_preparacion_datos.ipynb      # CRISP-DM Fases 1-3: Preparación de datos
│   ├── 02_modelado_predictivo.ipynb    # CRISP-DM Fase 4: Modelado y evaluación
│   └── 03_despliegue.ipynb             # CRISP-DM Fase 5: Despliegue y monitoreo
├── output/
│   ├── informe_crisp_dm.md             # Informe completo CRISP-DM
│   ├── pandas_profiling_report.html    # Reporte de profiling
│   ├── train_original.csv              # 70% original para CV leakage-safe
│   ├── train_balanced.csv              # Evidencia del balanceo del 70%
│   ├── test_unbalanced_raw.csv         # Test crudo reservado
│   ├── test_unbalanced.csv             # Test transformado para auditoría
│   ├── preprocessing_pipe.pkl          # Pipeline de preprocesamiento
│   └── metricas_finales.csv            # Métricas del modelo final
├── f1_final_dataset.csv                # Dataset original (Git LFS)
└── README.md
```

---

## 🎯 Resumen de Resultados

### Preparación de Datos
- **Variables iniciales:** 30
- **Variable eliminada por leakage:** `laps` (conocida después de la carrera)
- **Variables creadas:** `experience_ratio`, `grid_above_avg`, `avg_finish_rate`
- **Balanceo:** SMOTENC aplicado solo al 70% de entrenamiento y dentro de cada fold de CV

### Modelado Predictivo
- **Modelos entrenados:** 7 (4 supervisados + 3 ensambles)
- **Validación cruzada:** Estratificada k=5 sobre el 70% original, no sobre datos ya balanceados
- **Selección:** ANOVA + Tukey HSD para identificar los 3 mejores
- **Hiperparametrización:** GridSearchCV + Optuna (optimización bayesiana)
- **Control de sobreajuste:** `abs(F1_train - F1_test) <= 0.05`

### Mejor Modelo: RandomForest (GridSearchCV)
| Métrica | Valor |
|---------|-------|
| Train F1 | 0.6666 |
| Test F1 | 0.6353 |
| Gap Train-Test F1 | 0.0313 |
| Accuracy | 0.7819 |
| Precision | 0.5501 |
| Recall | 0.7515 |
| ROC-AUC | 0.8584 |
| Estado gap <= 5 pp | APROBADO |

---

## 🚀 Cómo Ejecutar

### 1. Preparación de Datos
```bash
jupyter notebook notebooks/01_preparacion_datos.ipynb
```

### 2. Modelado Predictivo
```bash
jupyter notebook notebooks/02_modelado_predictivo.ipynb
```

### 3. Despliegue con Streamlit
```bash
streamlit run app/app.py
```

---

## 📦 Dependencias Principales

- pandas
- numpy
- scikit-learn
- matplotlib / seaborn
- imbalanced-learn
- xgboost
- optuna
- streamlit
- joblib

---

## 📝 Entregables

1. **Informe CRISP-DM** (`output/informe_crisp_dm.md`)
2. **Notebook de preparación de datos** (`notebooks/01_preparacion_datos.ipynb`)
3. **Notebook de modelado predictivo** (`notebooks/02_modelado_predictivo.ipynb`)
4. **Notebook de despliegue** (`notebooks/03_despliegue.ipynb`)
5. **Aplicación Streamlit** (`app/app.py`)

---

## 👥 Autores

- **Equipo de Minería de Datos**

---

*Proyecto académico desarrollado siguiendo la metodología CRISP-DM.*
