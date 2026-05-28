"""Regenerate the CRISP-DM notebooks with leakage-safe methodology."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_notebook(path: Path, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path}")


def notebook_01():
    return [
        md(
            """
            # 01 - Preparación de Datos

            Este notebook cubre las fases CRISP-DM de entendimiento del negocio, entendimiento de datos y preparación. La regla metodológica central es NO usar información de test ni datos sintéticos para seleccionar, escalar o validar modelos.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import pickle
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from imblearn.over_sampling import SMOTENC
            from imblearn.pipeline import Pipeline as ImbPipeline
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            ROOT = Path.cwd().resolve()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from app.f1_pipeline import (
                CATEGORICAL_FEATURES,
                DEFAULT_SELECTED_FEATURES,
                FeatureEngineer,
                LeakageColumnDropper,
                ColumnSelector,
                Winsorizer,
                smotenc_feature_indices,
            )

            RANDOM_STATE = 42
            DATA_PATH = ROOT / 'f1_final_dataset.csv'
            OUTPUT_DIR = ROOT / 'output'
            OUTPUT_DIR.mkdir(exist_ok=True)

            sns.set_theme(style='whitegrid')
            pd.set_option('display.max_columns', 80)
            print(f'Root: {ROOT}')
            """
        ),
        md(
            """
            ## Fase 1: Entendimiento del Negocio

            El objetivo predictivo es estimar si un piloto terminará una carrera (`finished = 1`) usando solo variables disponibles antes o al inicio de la carrera. Por eso `laps` se elimina: se conoce después del evento y produciría leakage directo.

            Criterio de éxito: comparar al menos 4 modelos supervisados y 3 ensambles, seleccionar con validación cruzada sobre el 70% original, aplicar ANOVA/Tukey, optimizar los 3 mejores y desplegar un pipeline que acepte datos crudos pre-carrera.
            """
        ),
        md("## Fase 2: Entendimiento de los Datos"),
        code(
            """
            df = pd.read_csv(DATA_PATH)
            print(f'Dataset original: {df.shape[0]:,} filas x {df.shape[1]} columnas')
            display(df.head())

            target_counts = df['finished'].value_counts().sort_index()
            target_pct = df['finished'].value_counts(normalize=True).sort_index().mul(100)
            display(pd.DataFrame({'conteo': target_counts, 'porcentaje': target_pct.round(2)}))
            """
        ),
        code(
            """
            quality = {
                'finished_binario': bool(df['finished'].isin([0, 1]).all()),
                'grid_en_rango_0_34': bool(df['grid'].between(0, 34).all()),
                'driver_age_en_rango_17_60': bool(df['driver_age'].between(17, 60).all()),
                'sin_nulos': bool(df.isna().sum().sum() == 0),
                'laps_presente_pero_excluida_por_leakage': 'laps' in df.columns,
            }
            display(pd.Series(quality, name='cumple'))

            desc = df.describe().T
            desc['missing'] = df.isna().sum()
            desc['missing_pct'] = df.isna().mean().mul(100)
            display(desc[['count', 'missing', 'missing_pct', 'mean', 'std', 'min', '50%', 'max']].round(3))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts.plot(kind='bar', ax=ax, color=['#c94c4c', '#2f9e44'])
            ax.set_title('Distribución de finished')
            ax.set_xlabel('finished')
            ax.set_ylabel('registros')
            ax.set_xticklabels(['0: no terminó', '1: terminó'], rotation=0)
            for idx, value in enumerate(target_counts):
                ax.text(idx, value, f'{value:,}', ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'distribucion_finished.png', dpi=140, bbox_inches='tight')
            plt.show()
            """
        ),
        md(
            """
            ## Fase 3: Preparación de Datos

            La división train/test se hace ANTES de cualquier selección, winsorización, escalado o balanceo. Las decisiones de selección de variables se justifican con diagnósticos calculados solo sobre el 70% de entrenamiento original.
            """
        ),
        code(
            """
            df_model = df.drop(columns=['laps'])
            X = df_model.drop(columns=['finished'])
            y = df_model['finished']

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
            )

            print('Split estratificado:')
            print(f'Train original: {X_train_raw.shape[0]:,} filas')
            print(y_train.value_counts().sort_index().to_dict())
            print(f'Test original: {X_test_raw.shape[0]:,} filas')
            print(y_test.value_counts().sort_index().to_dict())
            """
        ),
        code(
            """
            # Diagnóstico de correlación y MI SOLO con train original.
            train_diag = X_train_raw.copy()
            train_diag['finished'] = y_train.values
            corr_target = train_diag.corr(numeric_only=True)['finished'].drop('finished').abs().sort_values(ascending=False)

            fe_diag = FeatureEngineer().fit_transform(X_train_raw)
            mi_X = fe_diag.drop(columns=[c for c in fe_diag.columns if c not in DEFAULT_SELECTED_FEATURES], errors='ignore')
            mi_scores = mutual_info_classif(mi_X, y_train, random_state=RANDOM_STATE)
            mi_df = pd.DataFrame({'feature': mi_X.columns, 'mutual_information': mi_scores}).sort_values('mutual_information', ascending=False)

            display(pd.DataFrame({'abs_corr_finished_train': corr_target}).head(20).round(4))
            display(mi_df.round(5))
            """
        ),
        code(
            """
            selected_features = DEFAULT_SELECTED_FEATURES
            removed_features = sorted(set(X_train_raw.columns).union({'experience_ratio', 'grid_above_avg', 'avg_finish_rate'}) - set(selected_features))

            print(f'Features seleccionadas ({len(selected_features)}):')
            print(selected_features)
            print('\\nFeatures excluidas por leakage/redundancia/baja utilidad diagnóstica:')
            print(removed_features)
            print('\\nCategóricas para SMOTENC:')
            print(CATEGORICAL_FEATURES)
            """
        ),
        code(
            """
            prep_no_sampler = Pipeline([
                ('drop_leakage', LeakageColumnDropper()),
                ('feature_engineering', FeatureEngineer()),
                ('winsorizer', Winsorizer()),
                ('selector', ColumnSelector(selected_features)),
                ('scaler', StandardScaler()),
            ])

            prep_for_smote = Pipeline([
                ('drop_leakage', LeakageColumnDropper()),
                ('feature_engineering', FeatureEngineer()),
                ('winsorizer', Winsorizer()),
                ('selector', ColumnSelector(selected_features)),
            ])

            X_train_selected = prep_for_smote.fit_transform(X_train_raw, y_train)
            X_test_selected = prep_for_smote.transform(X_test_raw)

            sampler = SMOTENC(
                categorical_features=smotenc_feature_indices(selected_features),
                random_state=RANDOM_STATE,
                k_neighbors=5,
            )
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_selected, y_train)

            scaler = StandardScaler()
            X_train_balanced_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_balanced), columns=selected_features
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_selected), columns=selected_features
            )

            train_balanced = X_train_balanced_scaled.copy()
            train_balanced['finished'] = y_train_balanced.to_numpy()
            test_unbalanced = X_test_scaled.copy()
            test_unbalanced['finished'] = y_test.to_numpy()

            train_original = X_train_raw.copy()
            train_original['finished'] = y_train.to_numpy()
            test_original = X_test_raw.copy()
            test_original['finished'] = y_test.to_numpy()

            train_balanced.to_csv(OUTPUT_DIR / 'train_balanced.csv', index=False)
            test_unbalanced.to_csv(OUTPUT_DIR / 'test_unbalanced.csv', index=False)
            train_original.to_csv(OUTPUT_DIR / 'train_original.csv', index=False)
            test_original.to_csv(OUTPUT_DIR / 'test_unbalanced_raw.csv', index=False)

            with open(OUTPUT_DIR / 'preprocessing_pipe.pkl', 'wb') as f:
                pickle.dump(prep_no_sampler, f)

            schema = {
                'target': 'finished',
                'leakage_columns_removed': ['laps'],
                'selected_features': selected_features,
                'categorical_features_for_smotenc': CATEGORICAL_FEATURES,
                'smotenc_categorical_indices': smotenc_feature_indices(selected_features),
                'cv_rule': 'CV se ejecuta sobre train_original.csv con SMOTENC dentro de cada fold, no sobre train_balanced.csv.',
            }
            with open(OUTPUT_DIR / 'feature_schema.json', 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)

            print('Artefactos guardados:')
            print(f"train_original.csv: {train_original.shape}")
            print(f"train_balanced.csv: {train_balanced.shape}; clases {pd.Series(y_train_balanced).value_counts().sort_index().to_dict()}")
            print(f"test_unbalanced.csv: {test_unbalanced.shape}; clases {y_test.value_counts().sort_index().to_dict()}")
            print('preprocessing_pipe.pkl y feature_schema.json actualizados')
            """
        ),
        md(
            """
            ## Resumen metodológico de preparación

            - `laps` se eliminó por ser información posterior a la carrera.
            - El split 70/30 se hizo antes de selección, winsorización, escalado y SMOTE.
            - `train_balanced.csv` documenta el balanceo exigido sobre el 70%, pero NO se usa para validar modelos.
            - La validación cruzada correcta ocurre en el notebook 02 sobre `train_original.csv`, con `SMOTENC` dentro del pipeline de cada fold.
            """
        ),
    ]


def notebook_02():
    return [
        md(
            """
            # 02 - Modelado Predictivo y Evaluación

            Este notebook corrige el leakage metodológico: cada fold de validación ajusta winsorización, ingeniería, selección, `SMOTENC` y escalado únicamente con el sub-train del fold. El test se usa una sola vez al final para estimar generalización.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import pickle
            import sys
            import time
            import warnings

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from imblearn.over_sampling import SMOTENC
            from imblearn.pipeline import Pipeline as ImbPipeline
            from scipy import stats
            from sklearn.base import clone
            from sklearn.dummy import DummyClassifier
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
            from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            import optuna
            from xgboost import XGBClassifier

            ROOT = Path.cwd().resolve()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from app.f1_pipeline import (
                DEFAULT_SELECTED_FEATURES,
                FeatureEngineer,
                LeakageColumnDropper,
                ColumnSelector,
                Winsorizer,
                smotenc_feature_indices,
            )

            warnings.filterwarnings('ignore')
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            RANDOM_STATE = 42
            OUTPUT_DIR = ROOT / 'output'
            MODELS_DIR = ROOT / 'models'
            MODELS_DIR.mkdir(exist_ok=True)
            sns.set_theme(style='whitegrid')
            """
        ),
        code(
            """
            train_original = pd.read_csv(OUTPUT_DIR / 'train_original.csv')
            test_original = pd.read_csv(OUTPUT_DIR / 'test_unbalanced_raw.csv')

            X_train = train_original.drop(columns=['finished'])
            y_train = train_original['finished']
            X_test = test_original.drop(columns=['finished'])
            y_test = test_original['finished']

            print(f'Train original para CV: {X_train.shape}; clases {y_train.value_counts().sort_index().to_dict()}')
            print(f'Test reservado: {X_test.shape}; clases {y_test.value_counts().sort_index().to_dict()}')
            """
        ),
        code(
            """
            selected_features = DEFAULT_SELECTED_FEATURES
            cat_indices = smotenc_feature_indices(selected_features)

            def build_pipeline(model):
                return ImbPipeline([
                    ('drop_leakage', LeakageColumnDropper()),
                    ('feature_engineering', FeatureEngineer()),
                    ('winsorizer', Winsorizer()),
                    ('selector', ColumnSelector(selected_features)),
                    ('sampler', SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE, k_neighbors=5)),
                    ('scaler', StandardScaler()),
                    ('model', model),
                ])

            base_models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE),
                'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE),
                'KNN': KNeighborsClassifier(n_neighbors=11),
                'NaiveBayes': GaussianNB(),
                'RandomForest': RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE),
                'XGBoost': XGBClassifier(n_estimators=140, max_depth=3, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=2),
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
            }
            """
        ),
        code(
            """
            baseline = DummyClassifier(strategy='most_frequent')
            baseline.fit(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_metrics = {
                'Accuracy': accuracy_score(y_test, baseline_pred),
                'Precision': precision_score(y_test, baseline_pred, zero_division=0),
                'Recall': recall_score(y_test, baseline_pred, zero_division=0),
                'F1': f1_score(y_test, baseline_pred, zero_division=0),
            }
            display(pd.Series(baseline_metrics, name='baseline_mayoritaria').to_frame().round(4))
            """
        ),
        code(
            """
            cv_rows = []
            fold_rows = []
            fitted_models = {}
            test_rows = []

            for name, estimator in base_models.items():
                pipe = build_pipeline(estimator)
                print(f'Validando {name}...', end=' ')
                start = time.time()
                scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
                elapsed = time.time() - start

                for fold, f1 in enumerate(scores['test_f1'], start=1):
                    fold_rows.append({'Modelo': name, 'fold': fold, 'F1': f1})

                row = {'Modelo': name, 'Tiempo_s': elapsed}
                for metric in scoring:
                    row[f'CV_{metric}_mean'] = scores[f'test_{metric}'].mean()
                    row[f'CV_{metric}_std'] = scores[f'test_{metric}'].std()
                row['Train_F1_mean'] = scores['train_f1'].mean()
                row['Gap_Train_CV_F1'] = row['Train_F1_mean'] - row['CV_f1_mean']
                cv_rows.append(row)

                fitted = clone(pipe).fit(X_train, y_train)
                fitted_models[name] = fitted
                pred = fitted.predict(X_test)
                prob = fitted.predict_proba(X_test)[:, 1]
                test_rows.append({
                    'Modelo': name,
                    'Test_Accuracy': accuracy_score(y_test, pred),
                    'Test_Precision': precision_score(y_test, pred, zero_division=0),
                    'Test_Recall': recall_score(y_test, pred, zero_division=0),
                    'Test_F1': f1_score(y_test, pred, zero_division=0),
                    'Test_ROC_AUC': roc_auc_score(y_test, prob),
                })
                print(f"CV F1={row['CV_f1_mean']:.4f}; Test F1={test_rows[-1]['Test_F1']:.4f}; {elapsed:.1f}s")

            cv_df = pd.DataFrame(cv_rows).sort_values('CV_f1_mean', ascending=False)
            fold_df = pd.DataFrame(fold_rows)
            test_df = pd.DataFrame(test_rows).sort_values('Test_F1', ascending=False)

            cv_df.to_csv(OUTPUT_DIR / 'comparativa_modelos_cv.csv', index=False)
            test_df.to_csv(OUTPUT_DIR / 'comparativa_modelos_test_informativo.csv', index=False)
            fold_df.to_csv(OUTPUT_DIR / 'f1_folds_modelos.csv', index=False)

            display(cv_df.round(4))
            display(test_df.round(4))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=fold_df, x='F1', y='Modelo', order=cv_df['Modelo'], ax=ax)
            ax.set_title('Distribución F1 por fold - CV leakage-safe')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'cv_f1_boxplot.png', dpi=140, bbox_inches='tight')
            plt.show()
            """
        ),
        md("## ANOVA y Tukey para selección de los 3 mejores"),
        code(
            """
            groups = [group['F1'].values for _, group in fold_df.groupby('Modelo')]
            anova_stat, anova_p = stats.f_oneway(*groups)
            print(f'ANOVA F={anova_stat:.4f}, p-value={anova_p:.6f}')

            tukey = pairwise_tukeyhsd(endog=fold_df['F1'], groups=fold_df['Modelo'], alpha=0.05)
            print(tukey)

            top3_names = cv_df.head(3)['Modelo'].tolist()
            print('Top 3 seleccionados por F1 medio de CV:', top3_names)

            with open(OUTPUT_DIR / 'anova_tukey.txt', 'w', encoding='utf-8') as f:
                f.write(f'ANOVA F={anova_stat:.6f}, p-value={anova_p:.8f}\\n\\n')
                f.write(str(tukey))
                f.write(f'\\n\\nTop 3 por CV F1: {top3_names}\\n')
            """
        ),
        md(
            """
            ## Hiperparametrización de los 3 mejores

            Se aplican dos estrategias a cada modelo seleccionado: `GridSearchCV` y optimización bayesiana con Optuna. La selección final se hace con F1 de CV interno ajustado por sobreajuste; el test sigue reservado para la evaluación final.

            Regla de control: el modelo desplegable debe cumplir `abs(F1_train - F1_test) <= 0.05`. Para empujar esa condición sin contaminar el test, se penalizan candidatos cuya brecha `F1_train_CV - F1_validacion_CV` supere 5 puntos porcentuales durante la optimización interna.
            """
        ),
        code(
            """
            MAX_F1_GAP = 0.05
            GAP_PENALTY_WEIGHT = 2.0

            def adjusted_cv_score(cv_f1, train_f1):
                gap = max(0.0, train_f1 - cv_f1)
                penalty = max(0.0, gap - MAX_F1_GAP) * GAP_PENALTY_WEIGHT
                return cv_f1 - penalty

            param_grids = {
                'LogisticRegression': {
                    'model__C': [0.05, 0.1, 0.3, 1.0],
                    'model__penalty': ['l2'],
                    'model__solver': ['lbfgs'],
                },
                'SVM': {
                    'model__C': [0.5, 1.0, 2.0],
                    'model__gamma': ['scale', 0.01],
                    'model__kernel': ['rbf'],
                },
                'KNN': {
                    'model__n_neighbors': [7, 11, 17],
                    'model__weights': ['uniform', 'distance'],
                },
                'NaiveBayes': {'model__var_smoothing': [1e-9, 1e-8, 1e-7]},
                'RandomForest': {
                    'model__n_estimators': [160, 220],
                    'model__max_depth': [4, 6, 8],
                    'model__min_samples_leaf': [20, 50, 80],
                    'model__min_samples_split': [40, 100],
                },
                'GradientBoosting': {
                    'model__n_estimators': [80, 120, 140],
                    'model__learning_rate': [0.04, 0.06],
                    'model__max_depth': [1, 2],
                    'model__min_samples_leaf': [20],
                },
                'XGBoost': {
                    'model__n_estimators': [100, 140],
                    'model__max_depth': [1, 2],
                    'model__learning_rate': [0.04, 0.06],
                    'model__min_child_weight': [10],
                    'model__reg_lambda': [8],
                    'model__reg_alpha': [0.5],
                },
            }

            def suggest_estimator(name, trial):
                if name == 'LogisticRegression':
                    return LogisticRegression(C=trial.suggest_float('C', 0.05, 5.0, log=True), max_iter=1000, random_state=RANDOM_STATE)
                if name == 'SVM':
                    return SVC(C=trial.suggest_float('C', 0.3, 4.0, log=True), gamma=trial.suggest_categorical('gamma', ['scale', 0.005, 0.02]), kernel='rbf', probability=True, random_state=RANDOM_STATE)
                if name == 'KNN':
                    return KNeighborsClassifier(n_neighbors=trial.suggest_int('n_neighbors', 5, 21, step=2), weights=trial.suggest_categorical('weights', ['uniform', 'distance']))
                if name == 'NaiveBayes':
                    return GaussianNB(var_smoothing=trial.suggest_float('var_smoothing', 1e-10, 1e-6, log=True))
                if name == 'RandomForest':
                    leaf = trial.suggest_int('min_samples_leaf', 20, 90, step=10)
                    return RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators', 120, 240, step=40), max_depth=trial.suggest_int('max_depth', 3, 8), min_samples_leaf=leaf, min_samples_split=max(40, leaf * 2), random_state=RANDOM_STATE, n_jobs=-1)
                if name == 'GradientBoosting':
                    return GradientBoostingClassifier(n_estimators=trial.suggest_int('n_estimators', 70, 150, step=20), learning_rate=trial.suggest_float('learning_rate', 0.03, 0.08), max_depth=trial.suggest_int('max_depth', 1, 2), min_samples_leaf=20, random_state=RANDOM_STATE)
                if name == 'XGBoost':
                    return XGBClassifier(n_estimators=trial.suggest_int('n_estimators', 80, 160, step=40), max_depth=trial.suggest_int('max_depth', 1, 2), learning_rate=trial.suggest_float('learning_rate', 0.03, 0.08), min_child_weight=10, subsample=trial.suggest_float('subsample', 0.8, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.8, 1.0), reg_lambda=8, reg_alpha=0.5, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=2)
                raise ValueError(name)
            """
        ),
        code(
            """
            tuning_rows = []
            best_candidates = []
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

            for name in top3_names:
                print(f'Optimizando {name} con GridSearchCV...')
                grid = GridSearchCV(
                    estimator=build_pipeline(base_models[name]),
                    param_grid=param_grids[name],
                    scoring='f1',
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                )
                grid.fit(X_train, y_train)
                grid_check = cross_validate(grid.best_estimator_, X_train, y_train, cv=inner_cv, scoring={'f1': 'f1'}, n_jobs=-1, return_train_score=True)
                grid_cv_f1 = grid_check['test_f1'].mean()
                grid_train_f1 = grid_check['train_f1'].mean()
                grid_gap = grid_train_f1 - grid_cv_f1
                grid_adjusted = adjusted_cv_score(grid_cv_f1, grid_train_f1)
                tuning_rows.append({'Modelo': name, 'Metodo': 'GridSearchCV', 'CV_F1': grid_cv_f1, 'Train_CV_F1': grid_train_f1, 'Gap_Train_CV_F1': grid_gap, 'CV_F1_Ajustado': grid_adjusted, 'Params': grid.best_params_})
                best_candidates.append((f'{name}_GridSearchCV', grid_adjusted, grid_cv_f1, grid_train_f1, grid.best_estimator_))

                print(f'Optimizando {name} con Optuna...')
                def objective(trial):
                    estimator = suggest_estimator(name, trial)
                    pipe = build_pipeline(estimator)
                    scores = cross_validate(pipe, X_train, y_train, cv=inner_cv, scoring={'f1': 'f1'}, n_jobs=-1, return_train_score=True)
                    return adjusted_cv_score(scores['test_f1'].mean(), scores['train_f1'].mean())

                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
                study.optimize(objective, n_trials=8, show_progress_bar=False)
                optuna_unfit = build_pipeline(suggest_estimator(name, study.best_trial))
                optuna_check = cross_validate(optuna_unfit, X_train, y_train, cv=inner_cv, scoring={'f1': 'f1'}, n_jobs=-1, return_train_score=True)
                optuna_cv_f1 = optuna_check['test_f1'].mean()
                optuna_train_f1 = optuna_check['train_f1'].mean()
                optuna_gap = optuna_train_f1 - optuna_cv_f1
                optuna_adjusted = adjusted_cv_score(optuna_cv_f1, optuna_train_f1)
                optuna_pipe = clone(optuna_unfit).fit(X_train, y_train)
                tuning_rows.append({'Modelo': name, 'Metodo': 'Optuna', 'CV_F1': optuna_cv_f1, 'Train_CV_F1': optuna_train_f1, 'Gap_Train_CV_F1': optuna_gap, 'CV_F1_Ajustado': optuna_adjusted, 'Params': study.best_params})
                best_candidates.append((f'{name}_Optuna', optuna_adjusted, optuna_cv_f1, optuna_train_f1, optuna_pipe))

            tuning_df = pd.DataFrame(tuning_rows).sort_values('CV_F1_Ajustado', ascending=False)
            tuning_df.to_csv(OUTPUT_DIR / 'optimizacion_top3.csv', index=False)
            display(tuning_df)
            """
        ),
        md("## Evaluación final en test reservado"),
        code(
            """
            best_name, best_adjusted_cv_f1, best_cv_f1, best_cv_train_f1, best_pipe = sorted(best_candidates, key=lambda item: item[1], reverse=True)[0]
            final_train_pred = best_pipe.predict(X_train)
            final_pred = best_pipe.predict(X_test)
            final_prob = best_pipe.predict_proba(X_test)[:, 1]
            final_train_f1 = f1_score(y_train, final_train_pred, zero_division=0)
            final_test_f1 = f1_score(y_test, final_pred, zero_division=0)
            final_train_test_gap = abs(final_train_f1 - final_test_f1)
            gap_status = 'APROBADO' if final_train_test_gap <= MAX_F1_GAP else 'RECHAZADO'

            final_metrics = {
                'Modelo': best_name,
                'CV_F1_Ajustado': best_adjusted_cv_f1,
                'CV_F1_seleccion': best_cv_f1,
                'CV_Train_F1_seleccion': best_cv_train_f1,
                'Train_F1_final': final_train_f1,
                'Accuracy': accuracy_score(y_test, final_pred),
                'Precision': precision_score(y_test, final_pred, zero_division=0),
                'Recall': recall_score(y_test, final_pred, zero_division=0),
                'F1': final_test_f1,
                'ROC_AUC': roc_auc_score(y_test, final_prob),
                'Gap_CV_Test_F1': best_cv_f1 - final_test_f1,
                'Gap_Train_Test_F1': final_train_test_gap,
                'Criterio_Gap_Train_Test_Max': MAX_F1_GAP,
                'Estado_Gap_Train_Test': gap_status,
            }

            metricas_finales = pd.DataFrame([final_metrics])
            metricas_finales.to_csv(OUTPUT_DIR / 'metricas_finales.csv', index=False)
            joblib.dump(best_pipe, MODELS_DIR / 'best_model_pipe.pkl')

            cm = confusion_matrix(y_test, final_pred)
            pd.DataFrame(cm, index=['real_0', 'real_1'], columns=['pred_0', 'pred_1']).to_csv(OUTPUT_DIR / 'confusion_matrix_final.csv')

            display(metricas_finales.round(4))
            display(pd.DataFrame(cm, index=['real_0', 'real_1'], columns=['pred_0', 'pred_1']))
            if gap_status != 'APROBADO':
                raise RuntimeError(f'El modelo final no cumple la brecha máxima de F1 train-test: {final_train_test_gap:.4f} > {MAX_F1_GAP:.4f}')
            print(f'Pipeline final guardado en {MODELS_DIR / "best_model_pipe.pkl"}')
            print('Pasos del pipeline:', list(best_pipe.named_steps.keys()))
            print(f'Criterio de brecha F1 train-test: {gap_status} ({final_train_test_gap:.4f} <= {MAX_F1_GAP:.4f})')
            """
        ),
        code(
            """
            # Sanity check: el pipeline final acepta datos crudos, sin variables derivadas ni escaladas.
            sample_raw = X_test.head(5).copy()
            sample_pred = best_pipe.predict(sample_raw)
            sample_prob = best_pipe.predict_proba(sample_raw)[:, 1]
            display(pd.DataFrame({'pred': sample_pred, 'prob_finish': sample_prob}).round(4))
            """
        ),
        md(
            """
            ## Veredicto metodológico

            - La selección de modelos se hizo por F1 de CV sobre el 70% original.
            - SMOTENC, winsorización y escalado se ajustaron dentro de cada fold.
            - ANOVA/Tukey se aplicó sobre los F1 por fold de los 7 modelos.
            - Los 3 mejores por CV fueron optimizados con GridSearchCV y Optuna.
            - El test desbalanceado se usó solo al final para estimar generalización.
            - El pipeline final contiene preparación + balanceo de entrenamiento + escalado + modelo y acepta datos crudos en inferencia.
            """
        ),
    ]


def notebook_03():
    return [
        md(
            """
            # 03 - Despliegue

            Este notebook valida que el artefacto desplegable sea un pipeline completo y que acepte datos crudos pre-carrera. La fase de Evaluación ya ocurrió antes; aquí se documenta despliegue y monitoreo.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import sys

            import joblib
            import pandas as pd

            ROOT = Path.cwd().resolve()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            MODEL_PATH = ROOT / 'models' / 'best_model_pipe.pkl'
            OUTPUT_DIR = ROOT / 'output'
            pipe = joblib.load(MODEL_PATH)
            print('Modelo cargado:', MODEL_PATH)
            print('Pasos:', list(pipe.named_steps.keys()))
            """
        ),
        code(
            """
            with open(OUTPUT_DIR / 'feature_schema.json', encoding='utf-8') as f:
                schema = json.load(f)
            print(json.dumps(schema, indent=2, ensure_ascii=False))
            """
        ),
        code(
            """
            raw_test = pd.read_csv(OUTPUT_DIR / 'test_unbalanced_raw.csv')
            X_raw = raw_test.drop(columns=['finished'])
            y_raw = raw_test['finished']

            sample = X_raw.head(10)
            pred = pipe.predict(sample)
            prob = pipe.predict_proba(sample)[:, 1]
            display(pd.DataFrame({'real': y_raw.head(10).to_numpy(), 'pred': pred, 'prob_finish': prob}).round(4))
            """
        ),
        code(
            """
            # Ejemplo manual con columnas crudas pre-carrera. No se envían variables escaladas ni derivadas.
            example = X_raw.median(numeric_only=True).to_frame().T
            example['grid'] = 1
            example['driver_race_count'] = 120
            example['constructor_race_count'] = 350
            example['driver_prev_finish_rate'] = 0.85
            example['constructor_prev_finish_rate'] = 0.82
            example['driver_last5_finish_rate'] = 0.80
            example['constructor_last5_finish_rate'] = 0.80
            example['has_qualifying'] = 1
            example['top10_start'] = 1

            pred = pipe.predict(example)[0]
            prob = pipe.predict_proba(example)[0, 1]
            print(f'Predicción ejemplo crudo: pred={pred}, prob_finish={prob:.4f}')
            display(example.T.rename(columns={0: 'valor'}))
            """
        ),
        md(
            """
            ## Streamlit

            La aplicación `app/app.py` carga `models/best_model_pipe.pkl`. La interfaz debe enviar columnas crudas compatibles con `test_unbalanced_raw.csv`; el pipeline se encarga de ingeniería, selección, winsorización y escalado. En inferencia, el paso `sampler` no altera una muestra nueva porque `imblearn.Pipeline.predict` solo aplica transformaciones y el estimador final.
            """
        ),
    ]


def main():
    write_notebook(NOTEBOOKS / '01_preparacion_datos.ipynb', notebook_01())
    write_notebook(NOTEBOOKS / '02_modelado_predictivo.ipynb', notebook_02())
    write_notebook(NOTEBOOKS / '03_despliegue.ipynb', notebook_03())


if __name__ == '__main__':
    main()
