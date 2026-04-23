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
│   └── best_model_pipe.pkl             # Pipeline del mejor modelo (SVM + Scaler)
├── notebooks/
│   ├── 01_preparacion_datos.ipynb      # CRISP-DM Fases 1-3: Preparación de datos
│   ├── 02_modelado_predictivo.ipynb    # CRISP-DM Fase 4: Modelado y evaluación
│   └── 03_despliegue.ipynb             # CRISP-DM Fase 5: Despliegue y monitoreo
├── output/
│   ├── informe_crisp_dm.md             # Informe completo CRISP-DM
│   ├── pandas_profiling_report.html    # Reporte de profiling
│   ├── train_balanced.csv              # Datos de entrenamiento balanceados
│   ├── test_unbalanced.csv             # Datos de prueba (sin balancear)
│   ├── preprocessing_pipe.pkl          # Pipeline de preprocesamiento
│   └── metricas_finales.csv            # Métricas del modelo final
├── f1_final_dataset.csv                # Dataset original (Git LFS)
└── README.md
```

---

## 🎯 Resumen de Resultados

### Preparación de Datos
- **Variables iniciales:** 30
- **Variables eliminadas:** 10 (redundancia por correlación > 0.8 o irrelevancia por MI < 0.01)
- **Variables creadas:** `experience_ratio`, `grid_above_avg`, `avg_finish_rate`
- **Balanceo:** SMOTE aplicado solo al 70% de entrenamiento (24,874 registros 50/50)

### Modelado Predictivo
- **Modelos entrenados:** 7 (4 supervisados + 3 ensambles)
- **Validación cruzada:** Estratificada k=5 sobre el 70% train
- **Selección:** ANOVA + Tukey HSD para identificar los 3 mejores
- **Hiperparametrización:** GridSearchCV + Optuna (optimización bayesiana)

### Mejor Modelo: SVM (Optuna)
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.6153 |
| Precision | 0.5750 |
| Recall | 0.8798 |
| **F1-Score** | **0.6955** |
| ROC-AUC | 0.6968 |

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
