"""
train_baseline_cv.py

5-fold cross-validated version of the ORIGINAL LightFusionNet baseline
(visual MobileNetV3-Small branch + rPPG green-channel branch, fused with
a fixed weight), matching exactly the CV methodology used in
train_cv_safe.py for the newer MoE model - so the two results are a fair,
apples-to-apples comparison.

WHY THIS SCRIPT EXISTS
-----------------------
Your original baseline (run_fusion.py, reproducing the paper's approach)
was only ever evaluated on ONE train/dev/test split, giving MAE 8.10.
Your improved models (MoE gate, attention fusion) were evaluated with
5-fold cross-validation ensembling, giving MAE 7.73 / 7.33. Comparing
"8.10 (1 split)" against "7.73 (5-fold ensemble)" isn't fully apples to
apples, because 5-fold ensembling alone tends to lower error a bit,
regardless of architecture. This script closes that gap by applying the
SAME 5-fold ensembling procedure to the ORIGINAL baseline, so you get an
honest "baseline with 5-fold CV" number to compare against.

METHOD (identical in spirit to train_cv_safe.py)
--------------------------------------------------
1. Pool = Training + Development videos ONLY (Testing is never touched
   until final evaluation).
2. Split the pool into 5 folds with GroupKFold, grouped by SUBJECT ID
   (not video ID) - so e.g. subject 203's Freeform AND Northwind videos
   always stay together, either both in train or both in validation.
   This prevents "leakage" where the model secretly memorizes a person
   instead of learning general patterns.
3. For each of the 5 folds:
     a. Train the ORIGINAL visual pipeline (SelectKBest -> SVR+RandomForest
        voting ensemble, exactly as in LightFusionNet/video_mobilenet.py)
        on that fold's training videos.
     b. Train the ORIGINAL rPPG pipeline (feature selection -> SVR /
        stacking ensemble, exactly as in LightFusionNet/rppg_pipeline.py)
        on that fold's training videos.
     c. Both models predict on the untouched Testing split.
     d. Fuse visual + rPPG predictions using the paper's weighting scheme
        (tests 90/10, 95/5, 98/2, 99/1 visual/rPPG, same as run_fusion.py).
4. Average the 5 folds' Testing predictions together (this "ensembling"
   step is what "5-fold cross-validation" means here - not 5 separate
   scores, but 5 independently-trained models voting together on the
   same untouched test set).
5. Report final MAE/RMSE/PCC/CCC, plus per-fold MAE so you can see how
   stable the result is across folds.

INPUTS (already extracted - nothing needs to be re-run):
    - config.VISUAL_RAW_BRANCH_DIR/visual_embeddings.npy   (N, 2304)
    - config.VISUAL_RAW_BRANCH_DIR/visual_sample_ids.npy   (N,)
    - config.RPPG_FEATURES_CSV                             (~297 rows)
    - config.LABELS_PATH                                   (video, split, BDI-II)

OUTPUTS:
    - config.BASELINE_CV_DIR/baseline_cv_predictions.csv   per-video final predictions
    - config.BASELINE_CV_DIR/results_baseline_cv.txt       summary metrics

A NOTE ON THE rPPG "BEST STRATEGY" SELECTION (read this):
Your original rppg_pipeline.py evaluates all 3 candidate strategies
(fixed SVR / stacking ensemble / weighted combo) directly on the Testing
set and keeps whichever scores lowest. That means the original pipeline
already picks its strategy using a peek at the test labels - a
methodological soft spot that predates this script and is already baked
into your existing 8.10/8.15 numbers. This script reproduces that exact
behavior per fold, for faithful comparison - it is not something newly
introduced here. If you ever want a stricter, leakage-free version later,
the fix is to pick the strategy using validation-fold performance instead
of Testing performance - just ask for that as a follow-up.

Run:
    python train_baseline_cv.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel, VarianceThreshold,
)
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.base import clone

import config

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


N_FOLDS = 5
VISUAL_WEIGHTS_TO_TEST = [0.90, 0.95, 0.98, 0.99]  # same as run_fusion.py


# ---------------------------------------------------------------------------
# Metrics (identical formulas to your other scripts, for direct comparability)
# ---------------------------------------------------------------------------

def concordance_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except Exception:
        pcc = 0.0
    ccc = concordance_ccc(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "CCC": ccc}


# ---------------------------------------------------------------------------
# rPPG feature engineering + selection (copied faithfully from
# LightFusionNet/rppg_pipeline.py so fold-level results are trustworthy)
# ---------------------------------------------------------------------------

def create_enhanced_features(df, feature_cols):
    available = [c for c in feature_cols if c in df.columns]
    out = df[available].copy().fillna(df[available].median())

    if {"RMSSD", "SDNN"}.issubset(out.columns):
        out["rmssd_sdnn_ratio"] = out["RMSSD"] / (out["SDNN"] + 1e-8)
    if {"LF", "HF"}.issubset(out.columns):
        out["lf_hf_ratio"] = out["LF"] / (out["HF"] + 1e-8)
    if "dfa_alpha" in out.columns:
        out["dfa_alpha_centered"] = out["dfa_alpha"] - 1.0

    return out.replace([np.inf, -np.inf], np.nan).fillna(out.median())


class AdvancedFeatureSelector:
    def __init__(self, k_best=25):
        self.k_best = k_best
        self.selected_features = None

    def fit(self, X, y, feature_names):
        feature_names = np.array(feature_names[: X.shape[1]])

        vt = VarianceThreshold(threshold=0.01)
        X_vt = vt.fit_transform(X)
        vt_feats = feature_names[vt.get_support()]

        k = min(self.k_best, X_vt.shape[1])
        kbest_feats = vt_feats[SelectKBest(f_regression, k=k).fit(X_vt, y).get_support()]

        mi_idx = np.argsort(mutual_info_regression(X_vt, y, random_state=42))[-self.k_best:]
        mi_feats = vt_feats[mi_idx]

        rf_sel = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            max_features=min(self.k_best, X_vt.shape[1]),
        )
        rf_feats = vt_feats[rf_sel.fit(X_vt, y).get_support()]

        combined = np.concatenate([vt_feats, kbest_feats, mi_feats, rf_feats])
        counts = pd.Series(combined).value_counts()
        self.selected_features = counts.head(self.k_best).index.tolist()
        return self

    def transform(self, X, feature_names):
        feature_names = np.array(feature_names)
        keep = [i for i, f in enumerate(feature_names) if f in self.selected_features]
        return X[:, keep]

    def fit_transform(self, X, y, feature_names):
        return self.fit(X, y, feature_names).transform(X, feature_names)


def fit_stacking(base_estimators, meta_learner, X, y, groups, n_splits=5):
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros((len(X), len(base_estimators)))
    fitted_bases = {}

    for i, (name, est) in enumerate(base_estimators):
        fold_preds = np.zeros(len(X))
        for tr, va in gkf.split(X, y, groups):
            m = clone(est)
            m.fit(X[tr], y[tr])
            fold_preds[va] = m.predict(X[va])
        oof[:, i] = fold_preds

        full = clone(est)
        full.fit(X, y)
        fitted_bases[name] = full

    meta = clone(meta_learner)
    meta.fit(oof, y)
    return fitted_bases, meta


def predict_stacked(fitted_bases, meta, X):
    meta_feats = np.column_stack([m.predict(X) for m in fitted_bases.values()])
    return meta.predict(meta_feats)


def build_base_models():
    base = [
        ("svr_rbf",  SVR(kernel="rbf",  C=5.0, epsilon=0.05, gamma="scale")),
        ("svr_poly", SVR(kernel="poly", degree=2, C=2.0, epsilon=0.1, gamma="scale")),
        ("elastic",  ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=2000)),
        ("bayesian", BayesianRidge()),
    ]
    if lgb is not None:
        base.append(("lgb", lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)))
    if xgb is not None:
        base.append(("xgb", xgb.XGBRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42,
            objective="reg:squarederror", verbosity=0)))
    meta = ElasticNet(alpha=0.1, l1_ratio=0.5)
    return base, meta


def train_rppg_fold(train_df, test_df, feature_cols):
    """
    Full original rPPG pipeline (3 candidate strategies), trained on this
    fold's training videos only, evaluated on the untouched Testing videos.
    Returns: predictions aligned to test_df row order, name of strategy used.
    """
    train_feat = create_enhanced_features(train_df, feature_cols)
    test_feat = create_enhanced_features(test_df, feature_cols)
    common_cols = sorted(set(train_feat.columns) & set(test_feat.columns))

    X_train = train_feat[common_cols].values.astype(float)
    y_train = train_df["BDI_II"].values.astype(float)
    X_test = test_feat[common_cols].values.astype(float)
    groups_train = train_df["video_id"].astype(str).str.split("_").str[0].values

    imputer = KNNImputer(n_neighbors=3)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    selector = AdvancedFeatureSelector(k_best=25)
    X_train_sel = selector.fit_transform(X_train, y_train, common_cols)
    X_test_sel = selector.transform(X_test, common_cols)

    pt = PowerTransformer(method="yeo-johnson")
    X_train_tf = pt.fit_transform(X_train_sel)
    X_test_tf = pt.transform(X_test_sel)

    y_test = test_df["BDI_II"].values.astype(float)

    # Strategy 1: fixed SVR
    svr = SVR(kernel="rbf", C=5.0, epsilon=0.05, gamma="scale")
    svr.fit(X_train_tf, y_train)
    pred_svr = svr.predict(X_test_tf)
    mae_svr = np.mean(np.abs(y_test - pred_svr))

    # Strategy 2: stacking ensemble
    base_models, meta_learner = build_base_models()
    fitted_bases, fitted_meta = fit_stacking(base_models, meta_learner, X_train_tf, y_train, groups_train)
    pred_ens = predict_stacked(fitted_bases, fitted_meta, X_test_tf)
    mae_ens = np.mean(np.abs(y_test - pred_ens))

    # Strategy 3: weighted combination
    alpha = 0.6
    pred_combo = alpha * pred_svr + (1 - alpha) * pred_ens
    mae_combo = np.mean(np.abs(y_test - pred_combo))

    strategies = {
        "Fixed SVR": (mae_svr, pred_svr),
        "Stacking Ensemble": (mae_ens, pred_ens),
        "Weighted Combination": (mae_combo, pred_combo),
    }
    best_name, (best_mae, best_preds) = min(strategies.items(), key=lambda kv: kv[1][0])
    return best_preds, best_name


# ---------------------------------------------------------------------------
# Data loading + alignment (with explicit leakage/mismatch checks)
# ---------------------------------------------------------------------------

def load_aligned_data():
    labels = pd.read_csv(config.LABELS_PATH)
    labels = labels.rename(columns={"video": "video_id"})

    visual_embeddings = np.load(config.VISUAL_RAW_BRANCH_DIR / "visual_embeddings.npy")
    visual_ids = np.load(config.VISUAL_RAW_BRANCH_DIR / "visual_sample_ids.npy", allow_pickle=True)
    visual_ids = [str(v) for v in visual_ids]

    rppg_df = pd.read_csv(config.RPPG_FEATURES_CSV)
    rppg_df["video_id"] = rppg_df["video_id"].astype(str)

    visual_id_set = set(visual_ids)
    rppg_id_set = set(rppg_df["video_id"])
    label_id_set = set(labels["video_id"].astype(str))

    common_ids = sorted(visual_id_set & rppg_id_set & label_id_set)

    print(f"Visual embeddings available : {len(visual_id_set)}")
    print(f"rPPG features available     : {len(rppg_id_set)}")
    print(f"Labels available            : {len(label_id_set)}")
    print(f"Common to all three         : {len(common_ids)}")

    missing_from_visual = (rppg_id_set & label_id_set) - visual_id_set
    missing_from_rppg = (visual_id_set & label_id_set) - rppg_id_set
    if missing_from_visual:
        print(f"WARNING: {len(missing_from_visual)} videos have rPPG+labels but no visual embedding - dropped.")
    if missing_from_rppg:
        print(f"WARNING: {len(missing_from_rppg)} videos have visual+labels but no rPPG features - dropped.")

    id_to_visual_idx = {vid: i for i, vid in enumerate(visual_ids)}

    labels_idx = labels.set_index("video_id")
    rppg_idx = rppg_df.set_index("video_id")

    rows = []
    for vid in common_ids:
        label_split = labels_idx.loc[vid, "split"]
        rppg_split = rppg_idx.loc[vid, "split"]
        if str(label_split) != str(rppg_split):
            raise RuntimeError(
                f"ALIGNMENT MISMATCH for {vid}: labels.csv says split='{label_split}' "
                f"but rppg_features.csv says split='{rppg_split}'. Stopping - fix this "
                f"before trusting any results."
            )
        label_bdi = float(labels_idx.loc[vid, "BDI-II"])
        rppg_bdi = float(rppg_idx.loc[vid, "BDI_II"])
        if abs(label_bdi - rppg_bdi) > 1e-6:
            raise RuntimeError(
                f"LABEL MISMATCH for {vid}: labels.csv BDI-II={label_bdi} but "
                f"rppg_features.csv BDI_II={rppg_bdi}. Stopping - this means samples "
                f"may be paired with the wrong label somewhere upstream."
            )
        rows.append({
            "video_id": vid,
            "split": label_split,
            "BDI_II": label_bdi,
            "visual_idx": id_to_visual_idx[vid],
        })

    aligned = pd.DataFrame(rows)
    rppg_feature_cols = [c for c in rppg_df.columns if c not in ("video_id", "split", "BDI_II")]

    return aligned, visual_embeddings, rppg_df.set_index("video_id"), rppg_feature_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config.BASELINE_CV_DIR.mkdir(parents=True, exist_ok=True)

    aligned, visual_embeddings, rppg_by_id, rppg_feature_cols = load_aligned_data()
    print(f"\nTotal aligned samples: {len(aligned)}")
    print(aligned["split"].value_counts())

    pool_df = aligned[aligned["split"].isin(["Training", "Development"])].reset_index(drop=True)
    test_df_ids = aligned[aligned["split"] == "Testing"].reset_index(drop=True)
    print(f"\nTrain+Dev pool: {len(pool_df)} videos  |  Testing (untouched): {len(test_df_ids)} videos")

    group_ids = pool_df["video_id"].str.split("_").str[0].values
    print(f"Unique subjects in pool: {len(set(group_ids))}")

    y_test = test_df_ids["BDI_II"].values.astype(float)
    X_visual_test = visual_embeddings[test_df_ids["visual_idx"].values]

    rppg_test_df = rppg_by_id.loc[test_df_ids["video_id"]].reset_index()
    assert (rppg_test_df["video_id"].values == test_df_ids["video_id"].values).all(), \
        "Testing set rPPG rows are out of order relative to labels - stopping before computing wrong metrics."
    rppg_test_df["BDI_II"] = test_df_ids["BDI_II"].values  # use canonical label

    kfold = GroupKFold(n_splits=N_FOLDS)
    indices = np.arange(len(pool_df))

    fold_visual_preds = []
    fold_rppg_preds = []
    fold_strategy_used = []

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(indices, groups=group_ids)):
        train_people = set(group_ids[i] for i in train_idx)
        val_people = set(group_ids[i] for i in val_idx)
        overlap = train_people & val_people
        if overlap:
            raise RuntimeError(f"LEAKAGE in fold {fold_index + 1}: subjects {overlap} in both train and val")

        print(f"\n=== Fold {fold_index + 1}/{N_FOLDS} ===")
        fold_train_df = pool_df.iloc[train_idx].reset_index(drop=True)
        print(f"Train: {len(fold_train_df)} videos ({len(train_people)} subjects)")

        # ---- Visual branch ----
        X_visual_train = visual_embeddings[fold_train_df["visual_idx"].values]
        y_visual_train = fold_train_df["BDI_II"].values.astype(float)

        k = min(1000, X_visual_train.shape[1])
        selector = SelectKBest(f_regression, k=k)
        X_visual_train_sel = selector.fit_transform(X_visual_train, y_visual_train)
        X_visual_test_sel = selector.transform(X_visual_test)

        svr = SVR(kernel="linear", C=0.1)
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
        ensemble = VotingRegressor([("svr", svr), ("rf", rf)])
        ensemble.fit(X_visual_train_sel, y_visual_train)
        visual_pred = ensemble.predict(X_visual_test_sel)
        fold_visual_preds.append(visual_pred)

        visual_fold_mae = np.mean(np.abs(y_test - visual_pred))
        print(f"  Visual branch alone -> MAE={visual_fold_mae:.4f}")

        # ---- rPPG branch ----
        fold_train_rppg_df = rppg_by_id.loc[fold_train_df["video_id"]].reset_index()
        assert (fold_train_rppg_df["video_id"].values == fold_train_df["video_id"].values).all(), \
            f"Fold {fold_index + 1} rPPG rows are out of order relative to labels - stopping."
        fold_train_rppg_df["BDI_II"] = fold_train_df["BDI_II"].values  # canonical label

        rppg_pred, strategy_used = train_rppg_fold(fold_train_rppg_df, rppg_test_df, rppg_feature_cols)
        fold_rppg_preds.append(rppg_pred)
        fold_strategy_used.append(strategy_used)

        rppg_fold_mae = np.mean(np.abs(y_test - rppg_pred))
        print(f"  rPPG branch alone   -> MAE={rppg_fold_mae:.4f}  (strategy: {strategy_used})")

    # ---- Ensemble: average the 5 folds' Testing predictions ----
    visual_ensemble_pred = np.mean(np.array(fold_visual_preds), axis=0)
    rppg_ensemble_pred = np.mean(np.array(fold_rppg_preds), axis=0)

    print(f"\n{'=' * 60}")
    print("5-FOLD ENSEMBLE RESULTS (averaged across all 5 fold-models)")
    print(f"{'=' * 60}")

    visual_metrics = compute_metrics(y_test, visual_ensemble_pred)
    rppg_metrics = compute_metrics(y_test, rppg_ensemble_pred)
    print(f"Visual only  -> MAE={visual_metrics['MAE']:.4f}  RMSE={visual_metrics['RMSE']:.4f}  "
          f"PCC={visual_metrics['PCC']:.4f}  CCC={visual_metrics['CCC']:.4f}")
    print(f"rPPG only    -> MAE={rppg_metrics['MAE']:.4f}  RMSE={rppg_metrics['RMSE']:.4f}  "
          f"PCC={rppg_metrics['PCC']:.4f}  CCC={rppg_metrics['CCC']:.4f}")

    results_rows = []
    best_weight_mae = None
    best_weight_name = None

    for visual_weight in VISUAL_WEIGHTS_TO_TEST:
        rppg_weight = 1.0 - visual_weight
        fused_pred = visual_weight * visual_ensemble_pred + rppg_weight * rppg_ensemble_pred
        m = compute_metrics(y_test, fused_pred)
        col_name = f"fusion_{int(visual_weight*100)}_{int(rppg_weight*100)}"
        print(f"{col_name:16s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  PCC={m['PCC']:.4f}  CCC={m['CCC']:.4f}")
        results_rows.append({"weight_config": col_name, **m})
        if best_weight_mae is None or m["MAE"] < best_weight_mae:
            best_weight_mae = m["MAE"]
            best_weight_name = col_name

    print(f"\nBest fusion weight: {best_weight_name}  (MAE={best_weight_mae:.4f})")
    print(f"rPPG strategy chosen per fold: {fold_strategy_used}")

    # ---- Per-fold stability check (same style as train_cv_safe.py) ----
    visual_fold_maes = np.array([np.mean(np.abs(y_test - p)) for p in fold_visual_preds])
    rppg_fold_maes = np.array([np.mean(np.abs(y_test - p)) for p in fold_rppg_preds])
    print(f"\nPer-fold visual MAE: mean={visual_fold_maes.mean():.4f}  std={visual_fold_maes.std():.4f}")
    print(f"Per-fold rPPG MAE:   mean={rppg_fold_maes.mean():.4f}  std={rppg_fold_maes.std():.4f}")

    # ---- Save outputs ----
    pred_out = test_df_ids[["video_id", "BDI_II"]].copy()
    pred_out["visual_pred"] = visual_ensemble_pred
    pred_out["rppg_pred"] = rppg_ensemble_pred
    for visual_weight in VISUAL_WEIGHTS_TO_TEST:
        rppg_weight = 1.0 - visual_weight
        col_name = f"fusion_{int(visual_weight*100)}_{int(rppg_weight*100)}"
        pred_out[col_name] = visual_weight * visual_ensemble_pred + rppg_weight * rppg_ensemble_pred
    pred_out.to_csv(config.BASELINE_CV_DIR / "baseline_cv_predictions.csv", index=False)

    results_path = config.BASELINE_CV_DIR / "results_baseline_cv.txt"
    with open(results_path, "w") as f:
        f.write("Original LightFusionNet baseline, 5-fold cross-validated ensemble\n")
        f.write(f"Testing samples: {len(test_df_ids)}\n")
        f.write(f"Per-fold visual MAE: mean={visual_fold_maes.mean():.4f}  std={visual_fold_maes.std():.4f}\n")
        f.write(f"Per-fold rPPG MAE:   mean={rppg_fold_maes.mean():.4f}  std={rppg_fold_maes.std():.4f}\n")
        f.write(f"rPPG strategy chosen per fold: {fold_strategy_used}\n\n")
        f.write(f"{'Config':<16}{'MAE':<10}{'RMSE':<10}{'PCC':<10}{'CCC':<10}\n")
        for row in results_rows:
            f.write(f"{row['weight_config']:<16}{row['MAE']:<10.4f}{row['RMSE']:<10.4f}"
                    f"{row['PCC']:<10.4f}{row['CCC']:<10.4f}\n")
        f.write(f"\nBest fusion weight: {best_weight_name}  (MAE={best_weight_mae:.4f})\n")
        f.write("\nCompare against: single-split baseline MAE 8.10, MoE 5-fold 7.73, attention 5-fold 7.33\n")

    print(f"\nSaved predictions to: {config.BASELINE_CV_DIR / 'baseline_cv_predictions.csv'}")
    print(f"Saved summary to:     {results_path}")


if __name__ == "__main__":
    main()
