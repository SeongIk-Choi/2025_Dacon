import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from lazypredict.Supervised import LazyRegressor

RANDOM_STATE = 42
N_SPLITS = 5
OUT_DIR = os.path.abspath("./RESULT_1_ENCODER_COMPARISON")

# embeddings loc
EMBEDDINGS_DIR = "./data_tox/Embeddings"

os.makedirs(OUT_DIR, exist_ok=True)
print("EMBEDDINGS_DIR:", EMBEDDINGS_DIR)
print("OUT_DIR:", OUT_DIR)

RFR_ONLY = [RandomForestRegressor]

def list_embedding_files():
    files = sorted(
        f
        for f in os.listdir(EMBEDDINGS_DIR)
        if f.endswith(".npy") and not f.lower().startswith("labels")
    )
    return [os.path.join(EMBEDDINGS_DIR, f) for f in files]

def run_kfold_lazy_rfr(
    x_path: str, y_path: Optional[str] = None,
    n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE,
    out_dir: str = OUT_DIR,
):
    X = np.asarray(np.load(x_path), dtype=np.float32)
    y = np.asarray(np.load(y_path), dtype=np.float32).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch: X {X.shape}, y {y.shape} for {x_path} / {y_path}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stem = os.path.splitext(os.path.basename(x_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_rows = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        reg = LazyRegressor(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            random_state=random_state,
            regressors=RFR_ONLY,
        )
        models, _ = reg.fit(X_train, X_valid, y_train, y_valid)
        models_df = models.reset_index().rename(columns={"index": "Model"})

        keep = ["Model", "R-Squared", "RMSE"]
        keep = [c for c in keep if c in models_df.columns]
        models_df = models_df[keep]
        if "R-Squared" in models_df.columns:
            models_df = models_df.rename(columns={"R-Squared": "R2"})

        models_df.insert(0, "Embedding", stem)
        models_df.insert(1, "Fold", fold_idx)
        models_df.insert(2, "y_path", os.path.basename(y_path))
        fold_rows.append(models_df)

        fold_csv = os.path.join(out_dir, f"rfr_lazy_{stem}_fold{fold_idx}_{ts}.csv")
        models_df.to_csv(fold_csv, index=False)
        print(f"  fold {fold_idx} -> {fold_csv}")

    all_df = pd.concat(fold_rows, ignore_index=True)
    fold_stack_path = os.path.join(out_dir, f"rfr_lazy_{stem}_all_folds_{ts}.csv")
    all_df.to_csv(fold_stack_path, index=False)

    summary = all_df.groupby("Model", as_index=False).agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
    )
    summary.insert(0, "Embedding", stem)
    summary_path = os.path.join(out_dir, f"rfr_lazy_{stem}_cv_summary_{ts}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  [stack] {fold_stack_path}\n  [summary] {summary_path} (R2/RMSE mean±std)")
    return all_df, summary, summary_path, fold_stack_path

from glob import glob
emb_paths = glob('../data_tox/Embeddings/train_*.npy')
emb_paths = [x for x in emb_paths if 'labels' not in x]
print(emb_paths)

y_path = '../data_tox/Embeddings/train_labels.npy'
master_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
all_summaries = []

print("Running ")
for x_path in emb_paths:
    print("\n===", os.path.basename(x_path), "===")
    _, summary, _, _ = run_kfold_lazy_rfr(x_path, y_path=y_path, out_dir=OUT_DIR)
    all_summaries.append(summary)

if all_summaries:
    master = pd.concat(all_summaries, ignore_index=True)
    master_path = os.path.join(OUT_DIR, f"rfr_lazy_all_embeddings_cv_summary_{master_ts}.csv")
    master.to_csv(master_path, index=False)
    print("\n[ALL] Master summary:", master_path)

