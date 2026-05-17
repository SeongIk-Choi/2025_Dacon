"""
MLM (or arbitrary encoder) embeddings + LazyRegressor: fit on full train,
then record performance only on Leaderboard / Blind sets (no 5-fold CV).

Assumes the same npy naming as `evaluate_testset.ipynb`:
  {Embeddings}/train_{stem}.npy + train_labels.npy
  {Embeddings}/leaderboard_{stem}.npy + leaderboard_labels.npy
  {Embeddings}/blind_{stem}.npy + blind_labels.npy

Example:
  cd src_tox
  python lazy_regression_all_models_LD_BL.py --stem morgan
  python lazy_regression_all_models_LD_BL.py --x-train ... --y-train ... (explicit paths)
"""

from __future__ import annotations

import argparse
import os
import warnings
import sys
from datetime import datetime
from pathlib import Path
from termios import TAB2

import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.utils import all_estimators

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42

MODEL_LIST = [
    "LGBMRegressor",
    "ExtraTreesRegressor",
    "SVR",
    "NuSVR",
    "KNeighborsRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "GradientBoostingRegressor",
    "BaggingRegressor",
    "BayesianRidge",
    "LassoLarsIC",
    "PoissonRegressor",
    "Ridge",
    "HuberRegressor",
    "TransformedTargetRegressor",
    "LinearRegression",
    "MLPRegressor",
    "GammaRegressor",
    "LinearSVR",
    "TweedieRegressor",
    "OrthogonalMatchingPursuit",
    "AdaBoostRegressor",
    "SGDRegressor",
    "DecisionTreeRegressor",
    "ExtraTreeRegressor",
    "PassiveAggressiveRegressor",
    "ElasticNet",
    "Lasso",
    "DummyRegressor",
    "LassoLars",
    "QuantileRegressor",
    "RANSACRegressor",
    "GaussianProcessRegressor",
    "KernelRidge",
]

def build_model_list() -> list:
    name_to_cls = dict(all_estimators(type_filter="regressor"))
    models = [name_to_cls[n] for n in MODEL_LIST if n in name_to_cls]

    try:
        from lightgbm import LGBMRegressor

        if "LGBMRegressor" in MODEL_LIST:
            models.insert(0, LGBMRegressor)
    except Exception as e:
        print("[WARN] LGBMRegressor:", e, file=sys.stderr)

    try:
        from xgboost import XGBRegressor

        if "XGBRegressor" in MODEL_LIST:
            insert_pos = 1 if models and models[0].__name__ == "LGBMRegressor" else 0
            models.insert(insert_pos, XGBRegressor)
    except Exception as e:
        print("[WARN] XGBRegressor:", e, file=sys.stderr)

    return models


def load_npy(path: str) -> np.ndarray:
    p = str(Path(path).resolve())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return np.load(p)


def run_lazy_one_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: list,
    random_state: int,
    out_dir: Path,
    data_type: str,
    embedding_tag: str,
    split_name: str,
    ts: str,
) -> pd.DataFrame:
    """Fit on full train; LazyRegressor computes RMSE/R² on `(X_test, y_test)`."""
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        random_state=random_state,
        regressors=models,
    )
    models_df_out, _ = reg.fit(
        X_train, X_test, y_train, y_test
    )
    models_df = models_df_out.reset_index().rename(columns={"index": "Model"})

    keep_cols = ["Model"]
    for col in ["Adjusted R-Squared", "R-Squared", "RMSE", "Time Taken"]:
        if col in models_df.columns:
            keep_cols.append(col)
    models_df = models_df[keep_cols]

    models_df.insert(0, "Data_Type", data_type)
    models_df.insert(1, "Embedding_Type", embedding_tag)
    models_df.insert(2, "Eval_Split", split_name)

    out_path = out_dir / f"lazy_regressor_{data_type}_{embedding_tag}_{split_name}_{ts}.csv"
    models_df.to_csv(out_path, index=False)
    print(f"[{split_name}] -> {out_path} (n_test={X_test.shape[0]})")
    if "R-Squared" in models_df.columns:
        print(models_df.sort_values("R-Squared", ascending=False).head(5).to_string())
    return models_df


def run_ld_and_blind(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_ld: np.ndarray,
    y_ld: np.ndarray,
    X_bl: np.ndarray,
    y_bl: np.ndarray,
    models: list,
    *,
    out_dir: Path,
    data_type: str = "MIN",
    embedding_tag: str = "MORGAN",
    random_state: int = RANDOM_STATE,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    y_ld = np.asarray(y_ld, dtype=np.float32).ravel()
    y_bl = np.asarray(y_bl, dtype=np.float32).ravel()
    for name, a, b in [("train", X_train, y_train), ("leaderboard", X_ld, y_ld), ("blind", X_bl, y_bl)]:
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"{name}: X {a.shape} vs y {b.shape}")

    print(f"[TS] {ts}  train={X_train.shape[0]}  leaderboard={X_ld.shape[0]}  blind={X_bl.shape[0]}")

    run_lazy_one_split(
        X_train, y_train, X_ld, y_ld, models, random_state,
        out_dir, data_type, embedding_tag, "leaderboard", ts,
    )
    run_lazy_one_split(
        X_train, y_train, X_bl, y_bl, models, random_state,
        out_dir, data_type, embedding_tag, "blind", ts,
    )


def _paths_from_stem(emb_dir: Path, stem: str) -> dict:
    return {
        "x_train": emb_dir / f"train_{stem}.npy",
        "y_train": emb_dir / "train_labels.npy",
        "x_ld": emb_dir / f"leaderboard_{stem}.npy",
        "y_ld": emb_dir / "leaderboard_labels.npy",
        "x_bl": emb_dir / f"blind_{stem}.npy",
        "y_bl": emb_dir / "blind_labels.npy",
    }


def main():
    TAG = 'MLM'
    x_train_path = f'./data_tox/Embeddings/train_{TAG.lower()}.npy'
    y_train_path = './data_tox/Embeddings/train_labels.npy'
    x_ld_path = f'./data_tox/Embeddings/leaderboard_{TAG.lower()}.npy'
    y_ld_path = './data_tox/Embeddings/leaderboard_labels.npy'
    x_bl_path = f'./data_tox/Embeddings/blind_{TAG.lower()}.npy'
    y_bl_path = './data_tox/Embeddings/blind_labels.npy'

    X_tr = load_npy(x_train_path)
    y_tr = load_npy(y_train_path)
    X_ld = load_npy(x_ld_path)
    y_ld = load_npy(y_ld_path)
    X_bl = load_npy(x_bl_path)
    y_bl = load_npy(y_bl_path)

    models = build_model_list()
    print("[INFO] Model count:", len(models))

    OUT_DIR = f"./RESULT_2_{TAG}_REGRESSOR_COMPARISON"
    RANDOM_STATE=42
    os.makedirs(OUT_DIR, exist_ok=True)

    run_ld_and_blind(
        X_tr,
        y_tr,
        X_ld,
        y_ld,
        X_bl,
        y_bl,
        models,
        out_dir=OUT_DIR,
        data_type='MIN',
        embedding_tag=TAG,
        random_state=RANDOM_STATE,
    )

if __name__ == "__main__":
    main()
