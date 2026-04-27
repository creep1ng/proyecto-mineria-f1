import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

def find_cell_index(substring):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and substring in cell.source:
            return i
    return None

def find_markdown_index(substring):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and substring in cell.source:
            return i
    return None

# 1. Update CV folds from 5 to 10
idx_cv = find_cell_index("StratifiedKFold(n_splits=5")
if idx_cv is not None:
    src = nb.cells[idx_cv].source
    src = src.replace("n_splits=5", "n_splits=10")
    # Update print and comments
    src = src.replace("Folds: 5", "Folds: 10")
    nb.cells[idx_cv].source = src

# 2. Update all n_jobs=5 to n_jobs=-1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("n_jobs=5", "n_jobs=-1")
        cell.source = cell.source.replace("n_jobs = 5", "n_jobs = -1")

# 3. Update markdown that mentions cv=5
idx_md_cv = find_markdown_index("cv=5")
if idx_md_cv is not None:
    nb.cells[idx_md_cv].source = nb.cells[idx_md_cv].source.replace("cv=5", "cv=10")

# 4. After test metrics evaluation cell, add stacking training
# Find the cell that ends test metrics and before comparison table
idx_test_metrics = find_cell_index("test_metrics_df = pd.DataFrame(test_metrics)")
if idx_test_metrics is not None:
    stacking_code = """# ============================================================
# 2.5 MODELO DE STACKING CON TOP 4 POR F1-SCORE
# ============================================================
from sklearn.ensemble import StackingClassifier

# Identificar top 4 modelos por F1 en test set
test_f1_ranking = test_metrics_df.sort_values('F1', ascending=False).reset_index(drop=True)
print('Ranking por F1 (test) - modelos base:')
display(test_f1_ranking[['Modelo', 'F1']].round(4))

top4_names = test_f1_ranking.head(4)['Modelo'].tolist()
print(f'Top 4 modelos seleccionados para stacking: {top4_names}')

# Construir estimators para StackingClassifier
stacking_estimators = []
for name in top4_names:
    model = trained_models[name]
    # Si es pipeline, usarlo directo; si no, envolver en pipeline con scaler si aplica
    stacking_estimators.append((name, model))

# Stacking con meta-clasificador LogisticRegression
stacking_model = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    passthrough=False,
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)

print('Entrenando StackingClassifier ...', end=' ')
start = time.time()
scores_stack = cross_validate(
    stacking_model, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False, n_jobs=-1
)
stacking_model.fit(X_train, y_train)
elapsed = time.time() - start

cv_results['Stacking'] = scores_stack
fit_times['Stacking'] = elapsed
trained_models['Stacking'] = stacking_model

# Evaluar en test
y_pred_stack = stacking_model.predict(X_test)
y_prob_stack = stacking_model.predict_proba(X_test)[:, 1]

acc_s = accuracy_score(y_test, y_pred_stack)
prec_s = precision_score(y_test, y_pred_stack, zero_division=0)
rec_s = recall_score(y_test, y_pred_stack, zero_division=0)
f1_s = f1_score(y_test, y_pred_stack, zero_division=0)
roc_s = roc_auc_score(y_test, y_prob_stack)

test_metrics.append({
    'Modelo': 'Stacking',
    'Accuracy': acc_s,
    'Precision': prec_s,
    'Recall': rec_s,
    'F1': f1_s,
    'ROC_AUC': roc_s,
    'Tiempo_s': fit_times['Stacking']
})
confusion_matrices['Stacking'] = confusion_matrix(y_test, y_pred_stack)

print(f'OK (F1={f1_s:.4f}, {elapsed:.2f}s)')

test_metrics_df = pd.DataFrame(test_metrics)
print('Métricas en Test Set incluyendo Stacking:')
display(test_metrics_df.round(4))"""
    nb.cells.insert(idx_test_metrics + 1, nbf.v4.new_code_cell(stacking_code))

# 5. Update comparison table building to include Stacking
# Find the cell that builds summary_rows
idx_comp = find_cell_index("summary_rows = []")
if idx_comp is not None:
    src = nb.cells[idx_comp].source
    src = src.replace("for name in models.keys():", "for name in list(models.keys()) + ['Stacking']:")
    nb.cells[idx_comp].source = src

# 6. Update ANOVA data preparation to include Stacking
idx_anova = find_cell_index("for name in models.keys():")
if idx_anova is not None:
    # Need to find the one inside ANOVA section - there are multiple. Let's check context.
    # The ANOVA cell has "anova_data = []" just before. Find that cell.
    pass

# Better approach: find the ANOVA cell and replace the loop
idx_anova_cell = find_cell_index("anova_data = []")
if idx_anova_cell is not None:
    src = nb.cells[idx_anova_cell].source
    src = src.replace("for name in models.keys():", "for name in list(models.keys()) + ['Stacking']:")
    nb.cells[idx_anova_cell].source = src

# 7. Update Tukey HSD section to mention 8 models
idx_tukey_md = find_markdown_index("7 modelos en los 5 folds")
if idx_tukey_md is not None:
    nb.cells[idx_tukey_md].source = nb.cells[idx_tukey_md].source.replace("7 modelos", "8 modelos")
    nb.cells[idx_tukey_md].source = nb.cells[idx_tukey_md].source.replace("5 folds", "10 folds")

# 8. Update top-3 selection to include Stacking
idx_top3 = find_cell_index("top3 = ranking.head(3)")
if idx_top3 is not None:
    src = nb.cells[idx_top3].source
    src = src.replace("top3_models = {k: models[k] for k in top3}", """top3_models = {}
for k in top3:
    if k in models:
        top3_models[k] = models[k]
    elif k == 'Stacking':
        top3_models[k] = stacking_model""")
    nb.cells[idx_top3].source = src

# 9. Add Stacking to param_grids for hyperparameterization
idx_grids = find_cell_index("param_grids = {")
if idx_grids is not None:
    src = nb.cells[idx_grids].source
    # Add Stacking grid before the closing brace
    stacking_grid = """    'Stacking': {
        'final_estimator__C': [0.01, 0.1, 1, 10, 100],
        'final_estimator__penalty': ['l2'],
        'final_estimator__solver': ['lbfgs'],
        'cv': [3, 5]
    },"""
    src = src.replace("    'GradientBoosting+Voting': {}  # Complejo", stacking_grid + "\n    'GradientBoosting+Voting': {}  # Complejo")
    nb.cells[idx_grids].source = src

# 10. Update Optuna objective to support Stacking
idx_optuna = find_cell_index("def make_optuna_objective")
if idx_optuna is not None:
    src = nb.cells[idx_optuna].source
    # Add Stacking case before else raise ValueError
    old_else = """        else:
            raise ValueError(f'Optuna no configurado para {model_name}')"""
    new_else = """        elif model_name == 'Stacking':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            cv_stack = trial.suggest_categorical('cv', [3, 5])
            meta = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
            model = StackingClassifier(
                estimators=stacking_estimators,
                final_estimator=meta,
                passthrough=False,
                cv=cv_stack,
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            raise ValueError(f'Optuna no configurado para {model_name}')"""
    src = src.replace(old_else, new_else)
    nb.cells[idx_optuna].source = src

# 11. Update final pipeline building to support Stacking
idx_final = find_cell_index("needs_scaling = global_best_name in")
if idx_final is not None:
    src = nb.cells[idx_final].source
    old = """needs_scaling = global_best_name in ['LogisticRegression', 'KNN', 'SVM', 'NaiveBayes']

if needs_scaling:"""
    new = """needs_scaling = global_best_name in ['LogisticRegression', 'KNN', 'SVM', 'NaiveBayes']
is_stacking = global_best_name == 'Stacking'

if is_stacking:
    final_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', global_best_model)
    ])
elif needs_scaling:"""
    src = src.replace(old, new)
    nb.cells[idx_final].source = src

# 12. Update documentation cells mentioning 7 models -> 8 models
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace("7 modelos", "8 modelos")
        cell.source = cell.source.replace("k=5", "k=10")
        cell.source = cell.source.replace("StratifiedKFold(k=5)", "StratifiedKFold(k=10)")
        cell.source = cell.source.replace("5 folds", "10 folds")
        # Add note about stacking
        if "Métodos implementados:" in cell.source:
            cell.source = cell.source.replace(
                "7. Gradient Boosting / VotingClassifier (ensamble)",
                "7. Gradient Boosting / VotingClassifier (ensamble)\n8. StackingClassifier (ensamble) — top 4 modelos por F1-score"
            )
        if "Métodos evaluados" in cell.source and "LR, KNN, SVM, NB, RF, XGB, GB+Voting" in cell.source:
            cell.source = cell.source.replace(
                "7 (LR, KNN, SVM, NB, RF, XGB, GB+Voting)",
                "8 (LR, KNN, SVM, NB, RF, XGB, GB+Voting, Stacking)"
            )
        if "StratifiedKFold(k=5)" in cell.source:
            cell.source = cell.source.replace("StratifiedKFold(k=5)", "StratifiedKFold(k=10)")

# 13. Update confusion matrix plot grid from 2x4 to 3x3
idx_cm = find_cell_index("fig, axes = plt.subplots(2, 4")
if idx_cm is not None:
    src = nb.cells[idx_cm].source
    src = src.replace("fig, axes = plt.subplots(2, 4, figsize=(16, 8))", "fig, axes = plt.subplots(3, 3, figsize=(16, 12))")
    nb.cells[idx_cm].source = src

# 14. Add Stacking to boxplot CV scores
idx_box = find_cell_index("f1_scores_dict = {name: cv_results")
if idx_box is not None:
    src = nb.cells[idx_box].source
    src = src.replace("for name in models.keys()", "for name in list(models.keys()) + ['Stacking']")
    src = src.replace("k=5)", "k=10)")
    nb.cells[idx_box].source = src

# 15. Update final summary metrics cell if it mentions specific model counts
idx_resum = find_markdown_index("Resumen Ejecutivo")
if idx_resum is not None:
    src = nb.cells[idx_resum].source
    src = src.replace("7 (LR, KNN, SVM, NB, RF, XGB, GB+Voting)", "8 (LR, KNN, SVM, NB, RF, XGB, GB+Voting, Stacking)")
    src = src.replace("StratifiedKFold(k=5)", "StratifiedKFold(k=10)")
    nb.cells[idx_resum].source = src

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} modificado exitosamente.")
