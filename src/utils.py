"""Reusable functions for the anomaly-detection project.

The notebook `notebooks/anomaly_detection.ipynb` only calls these functions and comments the results.

File layout
-----------
1. Common low-level tools: seed, timer, loading, score orientation (higher = more suspicious),
   ranking metrics (AP, ROC-AUC) and thresholded metrics, IQR rule, 60/20/20 split, preprocessing,
   threshold selection (max recall under a minimum precision), model building and evaluation, plots.
2. High-level steps called by the notebook, by part:
   - Part 1: Mouse (IF, LOF, IQR thresholds, novelty detection)            -> `mouse_*`
   - Part 2: Credit Card (preparation, grids, selection, test)            -> `creditcard_*`, `run_*_grid`, ...
   - Part 3: researched method EasyEnsemble + ECOD / GMM complement         -> `run_easyensemble`, `run_new_methods`
   - Part 4: AI4I 2020 (predictive maintenance, precision >= 90 %)          -> `ai4i_*`
   - Part 5: robustness (contaminated Mouse) and SHAP (Isolation Forest)   -> `mouse_robustness`, `shap_*`
   - Test-set plots (thresholds frozen on validation)                      -> `report_test_graphs`

Conventions: positive class = 1; higher score = more suspicious observation; transformations and
resampling fitted on the training set only; thresholds chosen on validation; the test set is used
once, with frozen models and thresholds.
"""
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier, IsolationForest
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_curve, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# =============================================================================
# 1. Common tools
# =============================================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # mouse.txt, creditcard.csv, ai4i2020.csv
SEED = 42                 # single seed for the whole project (reproducibility)
N_MOUSE_OUTLIERS = 10     # the last 10 rows of mouse.txt are the reference anomalies
CC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]   # Time excluded from the predictors


class Timer:
    """Context-manager stopwatch: `with Timer() as t: ...` then `t.elapsed` in seconds.
    Used to measure training and scoring times."""

    def __enter__(self):
        """Start the stopwatch when entering the `with` block."""
        self._t0 = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, *exc):
        """Stop the stopwatch and store the duration in `elapsed` (exceptions are not swallowed)."""
        self.elapsed = time.perf_counter() - self._t0
        return False


def set_seed(seed=SEED):
    """Fix the random seeds (random, numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def load_mouse(path, n_outliers=N_MOUSE_OUTLIERS):
    """Load mouse.txt and build the evaluation label vector.

    The last `n_outliers` rows of the file are the reference anomalies (y=1).
    Labels are built in file order, before any shuffling or split, so that they
    stay aligned with the observations. They are only used for evaluation,
    never to fit the unsupervised methods.

    Returns X (n, 2) and y (n,) with 0 = normal, 1 = anomaly.
    """
    # the file uses bare CR line endings; sep=r"\s+" handles them
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["x1", "x2"])
    if df.shape != (500, 2):
        raise ValueError(f"Unexpected Mouse format: {df.shape}, expected (500, 2)")
    if df.isna().any().any():
        raise ValueError("Missing values in Mouse")

    y = np.zeros(len(df), dtype=int)
    y[-n_outliers:] = 1
    return df.to_numpy(dtype=float), y


def orient_scores(raw_scores, higher_is_anomaly):
    """Harmonise scores so that a higher value = more suspicious observation.

    Isolation Forest (`score_samples`) and LOF (`negative_outlier_factor_`,
    `score_samples`) are oriented the other way: pass
    higher_is_anomaly=False to flip the sign.
    """
    raw_scores = np.asarray(raw_scores, dtype=float)
    return raw_scores if higher_is_anomaly else -raw_scores


def ranking_metrics(y_true, scores):
    """Threshold-free ranking evaluation: AP and ROC-AUC."""
    return {
        "AP": average_precision_score(y_true, scores),
        "AUC-ROC": roc_auc_score(y_true, scores),
    }


def threshold_metrics(y_true, scores, threshold):
    """Evaluation after thresholding: alert if score >= threshold.

    Positive class = 1 (anomaly). Returns confusion matrix, precision, recall,
    F1, balanced accuracy, number of alerts, false alarms (FP) and missed
    anomalies (FN).
    """
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "n_alerts": int(y_pred.sum()),
        "FP": int(fp),
        "FN": int(fn),
    }


def iqr_threshold(scores, k=1.5):
    """Label-free threshold (Tukey heuristic): Q3 + k * IQR.

    Computed on the scores only (higher = more suspicious), without labels.
    """
    q1, q3 = np.percentile(scores, [25, 75])
    return q3 + k * (q3 - q1)


def run_isolation_forest(X, n_estimators=100, seed=SEED):
    """Isolation Forest on X (outlier detection, no labels).

    Returns the model, harmonised scores (higher = more suspicious), the
    default-threshold predictions (1 = anomaly) and timings (s).
    """
    model = IsolationForest(n_estimators=n_estimators, random_state=seed)
    with Timer() as t_fit:
        model.fit(X)
    with Timer() as t_score:
        # score_samples: lower = more abnormal -> flip the sign
        scores = orient_scores(model.score_samples(X), higher_is_anomaly=False)
        pred = (model.predict(X) == -1).astype(int)
    return {"model": model, "scores": scores, "pred": pred,
            "t_fit": t_fit.elapsed, "t_score": t_score.elapsed}


def run_lof_outlier(X, n_neighbors=20):
    """LOF in outlier-detection mode (novelty=False): scores of the training points.

    Scores come from `negative_outlier_factor_` (negated LOF), flipped here so
    that higher = more suspicious (score = LOF, ~1 for a normal point).
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    with Timer() as t_fit:
        pred = (model.fit_predict(X) == -1).astype(int)
    scores = orient_scores(model.negative_outlier_factor_, higher_is_anomaly=False)
    return {"model": model, "scores": scores, "pred": pred, "t_fit": t_fit.elapsed}


def plot_detection(X, scores, pred, name, y=None, axes=None):
    """Three views of a method: detected points, map coloured by score, score histogram."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.scatter(*X[pred == 0].T, s=12, c="tab:blue", alpha=.5, label="not flagged")
    ax.scatter(*X[pred == 1].T, s=45, c="tab:red", marker="x", label=f"flagged ({pred.sum()})")
    if y is not None:  # reference labels: colouring only
        ax.scatter(*X[y == 1].T, s=110, facecolors="none", edgecolors="k",
                   label="reference anomaly")
    ax.set(title=f"{name}: flagged points (default threshold)", xlabel="x1", ylabel="x2", aspect="equal")
    ax.legend(fontsize=8)
    ax = axes[1]
    sc = ax.scatter(*X.T, c=scores, s=18, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="score (higher = more suspicious)")
    ax.set(title=f"{name}: coloured by score", xlabel="x1", ylabel="x2", aspect="equal")
    ax = axes[2]
    ax.hist(scores, bins=40, color="tab:gray", edgecolor="white")
    ax.set(title=f"{name}: score histogram", xlabel="score", ylabel="count")
    return axes


def compare_thresholds(y_true, scores, thresholds):
    """Comparison table of threshold rules {name: threshold} on the same scores.

    One row per rule: threshold, number of alerts, TN/FP/FN/TP, precision,
    recall, F1 (anomaly class), balanced accuracy.
    """
    rows = []
    for name, thr in thresholds.items():
        m = threshold_metrics(y_true, scores, thr)
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        rows.append({"rule": name, "threshold": thr, "alerts": m["n_alerts"],
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "precision": m["precision"], "recall": m["recall"],
                     "F1": m["F1"], "bal. acc.": m["balanced_accuracy"]})
    return pd.DataFrame(rows).set_index("rule")


def split_novelty(X, y, normal_test_frac=0.2, seed=SEED):
    """Split for novelty detection on Mouse.

    The test set gathers ALL anomalies plus a fraction `normal_test_frac` of the
    normal points; the training set only contains the remaining normals.
    Returns the indices (idx_train, idx_test) in the original order of X,
    which keeps the alignment with y.
    """
    rng = np.random.RandomState(seed)
    idx_normal = np.flatnonzero(y == 0)
    idx_anom = np.flatnonzero(y == 1)
    n_test_normal = int(round(normal_test_frac * len(idx_normal)))
    test_normal = rng.choice(idx_normal, size=n_test_normal, replace=False)
    idx_test = np.sort(np.r_[test_normal, idx_anom])
    idx_train = np.setdiff1d(idx_normal, test_normal)
    return idx_train, idx_test


def run_lof_novelty(X_train, X_test, n_neighbors=20):
    """LOF in novelty-detection mode (novelty=True).

    Fitted on X_train only. Returns:
    - training scores via `negative_outlier_factor_` (the only valid score on
      training points, used to set a label-free threshold);
    - test scores via `score_samples` (unseen points);
    - default-threshold predictions on the test set, and timings (s).
    All scores are harmonised: higher = more suspicious.
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    with Timer() as t_fit:
        model.fit(X_train)
    with Timer() as t_score:
        scores_test = orient_scores(model.score_samples(X_test), higher_is_anomaly=False)
    return {"model": model,
            "scores_train": orient_scores(model.negative_outlier_factor_, higher_is_anomaly=False),
            "scores_test": scores_test,
            "pred_test": (model.predict(X_test) == -1).astype(int),
            "t_fit": t_fit.elapsed, "t_score": t_score.elapsed}


def load_creditcard(path):
    """Load creditcard.csv as is (284,807 rows, columns Time, V1..V28, Amount, Class)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: download creditcard.csv from "
                                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into the data/ folder.")
    return pd.read_csv(path)


def deduplicate(df):
    """Remove strictly identical rows (all columns, including Time and Class).

    Guarantees that the same transaction cannot end up in two partitions
    (train -> validation/test leakage). Returns the cleaned DataFrame
    (index reset) and the number of removed rows.
    """
    out = df.drop_duplicates().reset_index(drop=True)
    return out, len(df) - len(out)


def split_60_20_20(y, seed=SEED):
    """Reproducible stratified split 60 % / 20 % / 20 % (train / validation / test).

    Returns three index arrays (positions in y); the exact same partitions are
    reused for every method on a given dataset.
    """
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(idx, train_size=0.6, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(idx_rest, train_size=0.5, stratify=y[idx_rest], random_state=seed)
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def make_preprocessor(scaled=True, amount_col="Amount", features=CC_FEATURES):
    """Credit Card preprocessing (to be fitted on the training set only).

    scaled=True  : log1p on Amount (highly skewed) then StandardScaler;
                   StandardScaler on V1..V28. For scale-sensitive methods
                   (LOF, logistic regression, SMOTE).
    scaled=False : no transformation (trees, Isolation Forest: insensitive
                   to feature scaling).
    """
    if not scaled:
        return ColumnTransformer([("id", "passthrough", list(features))])
    v_cols = [c for c in features if c != amount_col]
    amount = Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                       ("scale", StandardScaler())])
    return ColumnTransformer([("amount", amount, [amount_col]), ("v", StandardScaler(), v_cols)])


def choose_threshold(y_true, scores, min_precision):
    """Operating threshold chosen on validation: max recall subject to precision >= min_precision.

    Rule: among thresholds that satisfy the constraint and produce at least
    one alert (alert if score >= threshold), take the one with the highest
    recall; ties are broken by precision.
    If no threshold satisfies the constraint, `feasible=False` and the best
    observed trade-off (max-F1 threshold) is returned WITHOUT lowering the
    requirement: the caller must then report "target not met".
    """
    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]          # aligned with `thresholds`
    n_alerts = len(scores) - np.searchsorted(np.sort(scores), thresholds, side="left")   # number of scores >= threshold
    ok = (precision >= min_precision) & (n_alerts >= 1)
    if ok.any():
        cand = np.flatnonzero(ok)
        best = cand[np.lexsort((precision[cand], recall[cand]))[-1]]   # sort: recall then precision
        feasible = True
    else:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best, feasible = int(np.argmax(f1)), False
    return {"threshold": float(thresholds[best]), "precision": float(precision[best]),
            "recall": float(recall[best]), "n_alerts": int(n_alerts[best]), "feasible": feasible}


def build_supervised(model, variant, params, seed=SEED, smote_ratio=0.1):
    """Supervised pipeline for Credit Card.

    model   : 'tree' (decision tree) or 'logreg' (logistic regression)
    variant : 'none' (no correction), 'class_weight' (class weighting)
              or 'smote' (SMOTE on the training set only)
    params  : classifier hyperparameters (e.g. {'max_depth': 5} or {'C': 1.0})

    Transformations and resampling are Pipeline steps: they are fitted /
    applied only during `fit` on the training set, never during
    `predict_proba` on validation / test.
    """
    cw = "balanced" if variant == "class_weight" else None
    if model == "tree":
        clf = DecisionTreeClassifier(class_weight=cw, random_state=seed, **params)
    elif model == "logreg":
        clf = LogisticRegression(class_weight=cw, max_iter=2000, random_state=seed, **params)
    else:
        raise ValueError(model)
    # tree: no scaling, except before SMOTE (nearest-neighbour distances)
    scaled = model == "logreg" or variant == "smote"
    steps = [("prep", make_preprocessor(scaled=scaled))]
    if variant == "smote":
        steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, random_state=seed)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def _config_row(name, scores, y_val, min_precision, default_threshold, t_fit, t_score, extra=None):
    """Result row shared by supervised models and detectors: ranking (AP, ROC-AUC),
    operating threshold (project rule), metrics at that threshold, and metrics at
    the method's default threshold (for comparison)."""
    rank = ranking_metrics(y_val, scores)
    op = choose_threshold(y_val, scores, min_precision)
    at = threshold_metrics(y_val, scores, op["threshold"])
    default = threshold_metrics(y_val, scores, default_threshold)
    return {"config": name, **(extra or {}), **rank, **op,
            "F1": at["F1"], "bal. acc.": at["balanced_accuracy"], "FP": at["FP"], "FN": at["FN"],
            "default threshold": default_threshold, "alerts@default": default["n_alerts"],
            "precision@default": default["precision"], "recall@default": default["recall"],
            "F1@default": default["F1"],
            "t_fit (s)": t_fit, "t_score (s)": t_score}


def evaluate_config(name, estimator, X_train, y_train, X_val, y_val, min_precision, extra=None):
    """Supervised model: fit the Pipeline on the training set, score validation
    (class-1 probability), choose the threshold. Default threshold: 0.5.
    Returns the result row, the fitted model and the validation scores."""
    with Timer() as t_fit:
        estimator.fit(X_train, y_train)
    with Timer() as t_score:
        scores = estimator.predict_proba(X_val)[:, 1]
    row = _config_row(name, scores, y_val, min_precision, 0.5, t_fit.elapsed, t_score.elapsed, extra)
    return {"row": row, "model": estimator, "scores_val": scores}


def evaluate_detector(name, detector, X_fit, X_val, y_val, min_precision, extra=None):
    """Unsupervised detector (Isolation Forest, LOF novelty=True, ...).

    `fit(X_fit)` uses NO labels; y_val is only used to evaluate and to choose
    the threshold (a supervised selection step, on validation).
    Validation scores from `score_samples`, flipped (higher = more suspicious).
    Default threshold: the one used by `predict` (-offset_).
    """
    with Timer() as t_fit:
        detector.fit(X_fit)
    with Timer() as t_score:
        scores = orient_scores(detector.score_samples(X_val), higher_is_anomaly=False)
    row = _config_row(name, scores, y_val, min_precision, float(-detector.offset_), t_fit.elapsed, t_score.elapsed, extra)
    return {"row": row, "model": detector, "scores_val": scores}


def subsample_stratified(y, n_max, seed=SEED):
    """Indices of a fixed stratified subsample (<= n_max) of the training set,
    for expensive methods. y is only used for stratification."""
    idx = np.arange(len(y))
    if len(y) <= n_max:
        return idx
    keep, _ = train_test_split(idx, train_size=n_max, stratify=y, random_state=seed)
    return np.sort(keep)


def select_best(rows):
    """Best configuration on validation.

    Target met first; then, if met, highest recall; if NOT met (each row is then a
    "best trade-off" computed at its own threshold, so recalls are not comparable),
    highest trade-off F1; ties broken by precision then AP."""
    df = pd.DataFrame(rows)
    df["_crit"] = np.where(df["feasible"].astype(bool), df["recall"], df["F1"])
    return df.sort_values(["feasible", "_crit", "precision", "AP"], ascending=False).iloc[0].drop("_crit")


FAMILY_COLORS = {"tree": "tab:green", "logreg": "tab:blue", "IF": "tab:orange", "LOF": "tab:red",
                 "ECOD": "tab:purple", "GMM": "tab:brown",
                 "EE": "tab:green"}


def _n_alerts(scores, thresholds):
    """Number of alerts (score >= threshold) for each threshold."""
    return len(scores) - np.searchsorted(np.sort(scores), thresholds, side="left")


def plot_config_overview(res, min_precision=0.8):
    """Overview of configurations: AP, operating point and training time.

    `res`: DataFrame indexed by configuration with columns 'model', 'AP',
    'precision', 'recall', 'feasible', 't_fit (s)' (precision/recall at the operating threshold).
    """
    res = res.sort_values("AP")
    colors = [FAMILY_COLORS[m] for m in res["model"]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 0.42 * len(res) + 2.5),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.barh(res.index, res["AP"], color=colors)
    ax.set(title="Average Precision (validation)", xlabel="AP")
    ax = axes[1]
    for m, c in FAMILY_COLORS.items():
        d = res[res["model"] == m]
        ok = d["feasible"].astype(bool)
        ax.scatter(d.loc[ok, "recall"], d.loc[ok, "precision"], c=c, s=70, label=f"{m} (target met)")
        ax.scatter(d.loc[~ok, "recall"], d.loc[~ok, "precision"], c=c, s=70, marker="x", label=f"{m} (not met)")
    ax.axhline(min_precision, color="k", ls="--", lw=1, label=f"precision constraint = {min_precision}")
    ax.set(title="Operating point chosen on validation", xlabel="recall", ylabel="precision",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    ax.barh(res.index, res["t_fit (s)"], color=colors)
    ax.set(title="Training time (s, log scale)", xlabel="seconds", xscale="log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([])
    plt.tight_layout()
    return fig


def plot_pr_roc(scores_val, y_val, min_precision, op_rows, split="validation"):
    """ROC and precision-recall curves of several methods.

    scores_val: {name: scores}; op_rows: {name: dict with precision, recall, feasible}
    at the operating threshold (marker: circle = target met, cross = not met).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    prevalence = np.mean(y_val)
    for (name, s), c in zip(scores_val.items(), plt.cm.tab10.colors):
        fpr, tpr, _ = roc_curve(y_val, s)
        p, r, _ = precision_recall_curve(y_val, s)
        m = ranking_metrics(y_val, s)
        axes[0].plot(fpr, tpr, color=c, label=f"{name} (AUC={m['AUC-ROC']:.3f})")
        axes[1].plot(r, p, color=c, label=f"{name} (AP={m['AP']:.3f})")
        op = op_rows[name]
        axes[1].scatter(op["recall"], op["precision"], color=c, s=90, zorder=5,
                        marker="o" if op["feasible"] else "x", linewidths=2)
    axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="random")
    axes[0].set(title=f"ROC curve ({split})", xlabel="false positive rate", ylabel="true positive rate (recall)")
    axes[1].axhline(min_precision, color="k", ls="--", lw=1, label=f"precision constraint = {min_precision}")
    axes[1].axhline(prevalence, color="gray", ls=":", lw=1, label=f"random (= prevalence {prevalence:.2%})")
    axes[1].set(title=f"Precision-recall curve ({split}); markers = frozen threshold",
                xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    for a in axes:
        a.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_threshold_sweep(scores_val, y_val, op_rows, pos="fraud"):
    """F1 and balanced accuracy as a function of the threshold, one panel per method.

    The x axis is the number of alerts (log scale), equivalent to the threshold
    but comparable across methods. Vertical lines: operating threshold
    (project rule), F1 maximum, balanced-accuracy maximum.
    """
    n = len(scores_val)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, scores_val.items()):
        p, r, t = precision_recall_curve(y_val, s)
        p, r = p[:-1], r[:-1]
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        fpr, tpr, tt = roc_curve(y_val, s)
        bal = (tpr + 1 - fpr) / 2
        na_pr, na_roc = _n_alerts(s, t), _n_alerts(s, tt[1:])
        ax.plot(na_pr, f1, color="tab:purple", label=f"F1 ({pos} class)")
        ax.plot(na_roc, bal[1:], color="tab:cyan", label="balanced accuracy")
        i_f1, i_bal = int(np.argmax(f1)), int(np.argmax(bal[1:]))
        ax.axvline(na_pr[i_f1], color="tab:purple", ls=":", lw=1.2, label=f"max F1 ({na_pr[i_f1]} alerts)")
        ax.axvline(na_roc[i_bal], color="tab:cyan", ls=":", lw=1.2, label=f"max bal. acc. ({na_roc[i_bal]} alerts)")
        op = op_rows[name]
        ax.axvline(op["n_alerts"], color="k", ls="--", lw=1.2,
                   label=f"operating threshold ({op['n_alerts']} alerts" + ("" if op["feasible"] else ", target not met") + ")")
        ax.set(xscale="log", title=name, xlabel="number of alerts (threshold decreasing to the right)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axes[0].set_ylabel("value")
    plt.tight_layout()
    return fig


def plot_score_hist(scores_val, y_val, op_rows, pos="fraud"):
    """Histogram of validation scores per class (log scale), with the operating threshold."""
    n = len(scores_val)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, scores_val.items()):
        bins = np.histogram_bin_edges(s, bins=60)
        ax.hist(s[y_val == 0], bins=bins, alpha=.6, label="normal", color="tab:blue")
        ax.hist(s[y_val == 1], bins=bins, alpha=.8, label=pos, color="tab:red")
        ax.axvline(op_rows[name]["threshold"], color="k", ls="--", label="operating threshold")
        ax.set(yscale="log", title=name, xlabel="score (higher = more suspicious)", ylabel="count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def wilson_interval(k, n, z=1.96):
    """~95 % Wilson interval for a proportion k/n (precision or recall)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centre - half, centre + half)


def evaluate_on_test(model, kind, X_test, y_test, threshold):
    """Evaluate an already fitted model and a FROZEN threshold on the test set (no tuning here).

    kind: 'proba' (supervised, predict_proba) or 'score_samples' (detector: flipped score).
    Returns AP, ROC-AUC, metrics at the fixed threshold, number of alerts and scoring time.
    """
    with Timer() as t_score:
        if kind == "proba":
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = orient_scores(model.score_samples(X_test), higher_is_anomaly=False)
    out = {**ranking_metrics(y_test, scores), **threshold_metrics(y_test, scores, threshold)}
    out["t_score_test (s)"] = t_score.elapsed
    out["scores"] = scores
    out["n_fraud"] = int(np.sum(y_test))
    out["n_total"] = len(y_test)
    return out


def plot_default_vs_operational(rows, min_precision=0.8):
    """Default vs operating threshold (validation): number of alerts, precision, recall.

    rows: {method: result row with 'n_alerts', 'precision', 'recall',
    'alerts@default', 'precision@default', 'recall@default'}.
    """
    names = list(rows)
    x = np.arange(len(names))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (d, o, title, log) in zip(axes, [
            ("alerts@default", "n_alerts", "Number of alerts (log)", True),
            ("precision@default", "precision", "Precision", False),
            ("recall@default", "recall", "Recall", False)]):
        ax.bar(x - w / 2, [rows[n][d] for n in names], w, label="default threshold", color="tab:gray")
        ax.bar(x + w / 2, [rows[n][o] for n in names], w, label="operating threshold / trade-off", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)
        if title == "Precision":
            ax.axhline(min_precision, color="k", ls="--", lw=1, label=f"constraint {min_precision}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# =============================================================================
# 2. High-level steps: the notebook only calls these functions
# =============================================================================
MIN_PRECISION_FRAUD = 0.80          # Credit Card operating constraint
MIN_PRECISION_MAINTENANCE = 0.90    # AI4I operating constraint

SUPERVISED_GRID = {            # 6 configurations per model = 3 imbalance treatments x 2 values
    "tree": {v: [{"max_depth": 5}, {"max_depth": 10}] for v in ("none", "class_weight", "smote")},
    "logreg": {v: [{"C": 0.01}, {"C": 1.0}] for v in ("none", "class_weight", "smote")},
}
UNSUP_LABELS = {   # display name of each family in tables and plots
    "tree": "Decision tree", "logreg": "Logistic regression", "IF": "Isolation Forest", "LOF": "LOF"}
METHOD_DESCRIPTIONS = {   # (preprocessing, imbalance treatment / tuning) for the final table
    "Decision tree": ("log1p(Amount)+StandardScaler (before SMOTE)", "SMOTE (ratio 0.1), depth 5"),
    "Logistic regression": ("log1p(Amount)+StandardScaler", "class weighting, C=1"),
    "Isolation Forest": ("none", "none (unsupervised), 100 trees, max_samples=4096"),
    "LOF": ("log1p(Amount)+StandardScaler", "none (unsupervised), k=100, 30,000-row subsample"),
    "EasyEnsemble": ("none (trees)", "10 balanced 1:1 subsamples + AdaBoost (50 stumps)"),
}


def print_versions():
    """Library versions and seed (reproducibility)."""
    import platform
    import sys
    import imblearn
    import scipy
    set_seed()
    print("python", sys.version.split()[0], f"({platform.system()} {platform.release()})")
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                "imbalanced-learn": imblearn.__version__}
    try:   # shap is only used in part 5.2; show its version without hiding a missing install
        import shap
        versions["shap"] = shap.__version__
    except ImportError:
        versions["shap"] = "not installed"
    print(" | ".join(f"{k} {v}" for k, v in versions.items()))
    print("seed:", SEED)


# ------------------------------ Part 1: Mouse -----------------------------
def mouse_load_and_check(path):
    """Load Mouse, check the format and label alignment, print a summary."""
    X, y = load_mouse(path)
    print("X:", X.shape, X.dtype, "| y:", y.shape, y.dtype)
    print("NaN:", np.isnan(X).sum(), "| counts (normal, anomalies):", np.bincount(y))
    assert y[-N_MOUSE_OUTLIERS:].all() and not y[:-N_MOUSE_OUTLIERS].any()   # anomalies = last 10 rows
    df = pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y)
    display(df.describe().T)
    display(df.head(3))
    display(df.tail(12))
    return X, y


def plot_mouse(X, y):
    """Mouse scatter plot: normal points and reference anomalies."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=14, c="tab:blue", alpha=.6, label=f"normal ({(y == 0).sum()})")
    ax.scatter(*X[y == 1].T, s=60, c="tab:red", marker="x", label=f"reference anomaly ({y.sum()})")
    ax.set(xlabel="x1", ylabel="x2", title="Mouse: points and reference anomalies", aspect="equal")
    ax.legend()
    return fig


def fit_mouse_methods(X, y, n_neighbors=20):
    """IF and LOF on the whole Mouse dataset (no labels); numeric summary at the default threshold."""
    res_if = run_isolation_forest(X)
    res_lof = run_lof_outlier(X, n_neighbors=n_neighbors)
    for name, r in [("IF", res_if), ("LOF", res_lof)]:   # sign check: anomalies should score higher
        print(f"{name}: mean score normal = {r['scores'][y == 0].mean():.3f} | anomalies = {r['scores'][y == 1].mean():.3f}"
              f" | flagged by default = {r['pred'].sum()} | fit time = {r['t_fit'] * 1000:.0f} ms")
    return res_if, res_lof


def plot_mouse_methods(X, y, res_if, res_lof, n_neighbors=20):
    """2x3 grid: flagged points, map coloured by score, histogram (IF then LOF)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    plot_detection(X, res_if["scores"], res_if["pred"], "Isolation Forest", y=y, axes=axes[0])
    plot_detection(X, res_lof["scores"], res_lof["pred"], f"LOF (k={n_neighbors})", y=y, axes=axes[1])
    plt.tight_layout()
    return fig


def compare_lof_scores(X, y, res_lof, n_neighbors=20):
    """LOF training score (novelty=False) vs novelty score (novelty=True) on the same points."""
    lof_nov = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(X)
    s_train = res_lof["scores"]
    s_nov = orient_scores(lof_nov.score_samples(X), False)
    print("rank correlation (Spearman):", round(pd.Series(s_train).corr(pd.Series(s_nov), method="spearman"), 3))
    print("mean |train - novelty| gap:", round(np.abs(s_train - s_nov).mean(), 3))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s_train, s_nov, s=12, c=np.where(y == 1, "tab:red", "tab:blue"))
    ax.plot([s_train.min(), s_train.max()], [s_train.min(), s_train.max()], "k--", lw=1)
    ax.set(xlabel="LOF training score (novelty=False)", ylabel="score_samples (novelty=True) on X",
           title="Same LOF, two different scores")
    return fig


def mouse_threshold_study(X, y, res_if, res_lof):
    """Default threshold vs IQR rules (k=1.5 and 3, fixed a priori) for IF and LOF.

    Displays the tables (confusion matrix, precision, recall, F1, ...) and
    returns (rules, tables) for the figure."""
    rules, tables = {}, {}
    for name, r in [("IF", res_if), ("LOF", res_lof)]:
        s = r["scores"]
        default_thr = -r["model"].offset_       # default threshold, "higher = more suspicious" orientation
        assert ((s >= default_thr) == r["pred"].astype(bool)).all()   # consistent with predict / fit_predict
        rules[name] = {"default": default_thr, "IQR k=1.5": iqr_threshold(s, 1.5), "IQR k=3": iqr_threshold(s, 3.0)}
        tables[name] = compare_thresholds(y, s, rules[name])
        print(f"=== {name} ===")
        display(tables[name].round(3))
    return rules, tables


def plot_mouse_thresholds(X, y, res_if, res_lof, rules, tables):
    """Score histogram with the 3 thresholds + points flagged by each rule."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for row, (name, r) in enumerate([("IF", res_if), ("LOF", res_lof)]):
        s = r["scores"]
        ax = axes[row, 0]
        ax.hist(s, bins=40, color="tab:gray", edgecolor="white")
        for (rule, thr), col in zip(rules[name].items(), ["k", "tab:orange", "tab:red"]):
            ax.axvline(thr, color=col, ls="--", label=f"{rule} = {thr:.2f}")
        ax.set(title=f"{name}: scores and thresholds", xlabel="score")
        ax.legend(fontsize=8)
        for ax, (rule, thr) in zip(axes[row, 1:], rules[name].items()):
            flag = s >= thr
            ax.scatter(*X[~flag].T, s=12, c="tab:blue", alpha=.5)
            ax.scatter(*X[flag].T, s=45, c="tab:red", marker="x", label=f"flagged ({flag.sum()})")
            ax.scatter(*X[y == 1].T, s=110, facecolors="none", edgecolors="k", label="reference anomaly")
            ax.set(title=f'{name} – {rule} (F1 = {tables[name].loc[rule, "F1"]:.2f})', aspect="equal")
            ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def mouse_ranking_table(y, res_if, res_lof):
    """AP and ROC-AUC of each method's scores (threshold-independent)."""
    rank = pd.DataFrame({"IF": ranking_metrics(y, res_if["scores"]), "LOF": ranking_metrics(y, res_lof["scores"])}).T
    display(rank.round(3))
    return rank


def plot_mouse_disagreement(X, y, res_if, res_lof, ks=(1.5, 3.0)):
    """IF / LOF disagreements under the SAME threshold rule (IQR, k fixed a priori), on the 500 points.

    For each k: points flagged by both methods, by IF only, by LOF only (with the number of
    true anomalies in each group). Labels are only used for counting and circling."""
    fig, axes = plt.subplots(1, len(ks), figsize=(6.5 * len(ks), 6))
    rows = []
    for ax, k in zip(np.atleast_1d(axes), ks):
        f_if = res_if["scores"] >= iqr_threshold(res_if["scores"], k)
        f_lof = res_lof["scores"] >= iqr_threshold(res_lof["scores"], k)
        groups = {"both": f_if & f_lof, "IF only": f_if & ~f_lof, "LOF only": ~f_if & f_lof}
        ax.scatter(*X[~(f_if | f_lof)].T, s=10, c="lightgray", label="none")
        for (lab, m), (col, mk) in zip(groups.items(), [("tab:purple", "o"), ("tab:orange", "s"), ("tab:red", "^")]):
            ax.scatter(*X[m].T, s=40, c=col, marker=mk, label=f"{lab} ({m.sum()}, incl. {int(y[m].sum())} anomalies)")
            rows.append({"rule": f"IQR k={k}", "group": lab, "points": int(m.sum()), "of which anomalies": int(y[m].sum())})
        ax.scatter(*X[y == 1].T, s=130, facecolors="none", edgecolors="k", label="reference anomaly")
        ax.set(title=f"IF / LOF disagreements – IQR rule k={k}", xlabel="x1", ylabel="x2", aspect="equal")
        ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    display(pd.DataFrame(rows).set_index(["rule", "group"]))
    return fig


def mouse_novelty_fit(X, y, n_neighbors=20, normal_test_frac=0.2):
    """Novelty detection: LOF novelty=True fitted on the remaining normals, evaluated on held-out points.

    Thresholds without test labels: default (`predict`) and IQR computed on the training scores
    (`negative_outlier_factor_`). Prints counts, scores, timings and table; returns objects for the figure."""
    idx_train, idx_test = split_novelty(X, y, normal_test_frac=normal_test_frac, seed=SEED)
    assert not set(idx_train) & set(idx_test) and y[idx_train].sum() == 0   # no leakage, 100 % normal training set
    X_tr, X_te, y_te = X[idx_train], X[idx_test], y[idx_test]
    print(f"train: {len(X_tr)} normal points | test: {len(X_te)} points = {(y_te == 0).sum()} normal + {y_te.sum()} anomalies")
    nov = run_lof_novelty(X_tr, X_te, n_neighbors=n_neighbors)
    s_tr, s_te = nov["scores_train"], nov["scores_test"]
    print(f"mean score: train {s_tr.mean():.3f} | test normal {s_te[y_te == 0].mean():.3f} | test anomalies {s_te[y_te == 1].mean():.3f}")
    print(f"time: fit {nov['t_fit'] * 1000:.1f} ms | test scoring {nov['t_score'] * 1000:.1f} ms")
    thr = {"default (predict)": -nov["model"].offset_, "IQR k=1.5": iqr_threshold(s_tr, 1.5), "IQR k=3": iqr_threshold(s_tr, 3.0)}
    assert ((s_te >= thr["default (predict)"]) == nov["pred_test"].astype(bool)).all()
    table = compare_thresholds(y_te, s_te, thr)
    display(table.round(3))
    display(pd.DataFrame({"LOF novelty (test)": ranking_metrics(y_te, s_te)}).T.round(3))
    return {"X_tr": X_tr, "X_te": X_te, "y_te": y_te, "nov": nov, "table": table}


def plot_mouse_novelty(res):
    """Figure: training set only, test set (ground truth), test coloured by score with detections."""
    X_tr, X_te, y_te, nov = res["X_tr"], res["X_te"], res["y_te"], res["nov"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    ax.scatter(*X_tr.T, s=14, c="tab:blue", alpha=.6)
    ax.set(title=f"Training set: {len(X_tr)} normal points only", xlabel="x1", ylabel="x2", aspect="equal")
    ax = axes[1]
    ax.scatter(*X_tr.T, s=10, c="lightgray", label="training (background)")
    ax.scatter(*X_te[y_te == 0].T, s=28, c="tab:blue", label=f"test normal ({(y_te == 0).sum()})")
    ax.scatter(*X_te[y_te == 1].T, s=70, c="tab:red", marker="x", label=f"test anomaly ({y_te.sum()})")
    ax.set(title="Test set: held-out points (ground truth)", xlabel="x1", aspect="equal")
    ax.legend(fontsize=8)
    ax = axes[2]
    sc = ax.scatter(*X_te.T, c=nov["scores_test"], s=40, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="LOF novelty score (higher = more suspicious)")
    det = nov["pred_test"] == 1
    ax.scatter(*X_te[det].T, s=140, facecolors="none", edgecolors="red", linewidths=1.5, label=f"flagged by default ({det.sum()})")
    ax.set(title="Test set: score and detections (default threshold)", xlabel="x1", aspect="equal")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ------------------------- Part 2: Credit Card ----------------------------
def creditcard_overview(path):
    """Load Credit Card and print shape, dtypes, missing values, fraud rate."""
    df_raw = load_creditcard(path)
    print("shape:", df_raw.shape)
    print("dtypes:", df_raw.dtypes.value_counts().to_dict())
    print("missing values:", int(df_raw.isna().sum().sum()))
    cnt = df_raw["Class"].value_counts()
    print(f"frauds: {cnt[1]} / {len(df_raw)} = {cnt[1] / len(df_raw):.4%}  (1 in {cnt[0] / cnt[1]:.0f})")
    return df_raw


def creditcard_deduplicate(df_raw):
    """Inspect then remove exact duplicates (before splitting, to avoid any leakage)."""
    dup = df_raw.duplicated()
    feat_cols = [c for c in df_raw.columns if c != "Class"]
    grp = df_raw[df_raw.duplicated(feat_cols, keep=False)].groupby(feat_cols)["Class"].nunique()
    print("duplicated rows (all columns):", dup.sum(), "| of which frauds:", df_raw[dup]["Class"].sum())
    print("duplicate groups with conflicting labels:", int((grp > 1).sum()))
    df, n_removed = deduplicate(df_raw)
    print(f"after removal: {len(df)} rows ({n_removed} removed) | frauds: {df.Class.sum()} ({df.Class.mean():.4%})")
    return df


def creditcard_split(df, seed=SEED):
    """Stratified 60/20/20 split; anti-leakage checks; Time excluded from the predictors.

    Returns a dict {df, y, Xtr, ytr, Xva, yva, Xte, yte, idx_*}."""
    y = df["Class"].to_numpy()
    idx_tr, idx_va, idx_te = split_60_20_20(y, seed=seed)
    parts = {"train": idx_tr, "validation": idx_va, "test": idx_te}
    assert len(set(idx_tr) | set(idx_va) | set(idx_te)) == len(df)            # complete partition
    assert not (set(idx_tr) & set(idx_va) or set(idx_tr) & set(idx_te) or set(idx_va) & set(idx_te))
    keys = {k: set(map(tuple, df.iloc[i][CC_FEATURES + ["Time"]].to_numpy())) for k, i in parts.items()}
    assert not (keys["train"] & keys["validation"] or keys["train"] & keys["test"]
                or keys["validation"] & keys["test"])                           # no identical row across partitions
    X = df[CC_FEATURES]
    out = {"df": df, "y": y, "idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te}
    for tag, i in zip(("tr", "va", "te"), (idx_tr, idx_va, idx_te)):
        out["X" + tag], out["y" + tag] = X.iloc[i], y[i]
    display(pd.DataFrame({k: {"transactions": len(i), "frauds": int(y[i].sum()), "fraud rate": y[i].mean()}
                          for k, i in parts.items()}).T)
    return out


def run_supervised_grid(cc, min_precision=MIN_PRECISION_FRAUD, grid=SUPERVISED_GRID):
    """Decision tree and logistic regression x 3 imbalance treatments x 2 hyperparameter values.

    Each configuration's threshold is chosen on validation. Returns (sup, res_sup);
    displays the validation results table."""
    assert all(sum(len(v) for v in g.values()) <= 6 for g in grid.values())   # <= 6 configurations per model
    sup = {}
    for model, variants in grid.items():
        for variant, plist in variants.items():
            for params in plist:
                name = f"{model} | {variant} | {', '.join(f'{k}={v}' for k, v in params.items())}"
                est = build_supervised(model, variant, params, seed=SEED)
                sup[name] = evaluate_config(name, est, cc["Xtr"], cc["ytr"], cc["Xva"], cc["yva"], min_precision,
                                            extra={"model": model, "imbalance": variant})
    res = pd.DataFrame([v["row"] for v in sup.values()]).set_index("config")
    cols = ["AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "F1", "FP", "FN", "t_fit (s)"]
    display(res[cols].round(3))
    return sup, res


def print_resampling_sizes(sup, cc):
    """Sample sizes actually seen by the classifier with SMOTE (training only); validation/test unchanged."""
    pipe = sup["logreg | smote | C=1.0"]["model"]
    Z = pipe.named_steps["prep"].transform(cc["Xtr"])
    _, y_sm = pipe.named_steps["smote"].fit_resample(Z, cc["ytr"])
    print(f"original training set: {len(cc['ytr'])} (frauds {cc['ytr'].sum()}) | after SMOTE: {len(y_sm)} "
          f"(frauds {y_sm.sum()}, normal {(y_sm == 0).sum()})")
    print(f"validation unchanged: {len(cc['yva'])} (frauds {cc['yva'].sum()}) | test unchanged: {len(cc['yte'])} (frauds {cc['yte'].sum()})")


def run_unsupervised_grid(cc, min_precision=MIN_PRECISION_FRAUD, n_sub=30000):
    """Isolation Forest (full training set) and LOF novelty=True (fixed stratified subsample).

    Fitted without labels; threshold and configuration chosen on validation.
    Returns (unsup, res_unsup, prep_scaled); displays the validation table."""
    Xtr, ytr, Xva, yva = cc["Xtr"], cc["ytr"], cc["Xva"], cc["yva"]
    idx_sub = subsample_stratified(ytr, n_max=n_sub, seed=SEED)
    prep_scaled = make_preprocessor(scaled=True).fit(Xtr)             # fitted on the whole training set
    Ztr_sub, Zva = prep_scaled.transform(Xtr.iloc[idx_sub]), prep_scaled.transform(Xva)
    print(f"LOF: subsample of {len(idx_sub)} rows incl. {ytr[idx_sub].sum()} frauds ({ytr[idx_sub].mean():.3%})"
          f" | IF: full training set {len(Xtr)} rows")
    unsup = {}
    for ne in (100, 300):
        for ms in (256, 1024, 4096):
            name = f"IF | n_estimators={ne}, max_samples={ms}"
            det = IsolationForest(n_estimators=ne, max_samples=ms, random_state=SEED, n_jobs=-1)
            unsup[name] = evaluate_detector(name, det, Xtr.to_numpy(), Xva.to_numpy(), yva, min_precision,
                                            extra={"model": "IF", "n_fit": len(Xtr)})
    for k in (10, 20, 50, 100, 200, 500):
        name = f"LOF | n_neighbors={k}"
        det = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        unsup[name] = evaluate_detector(name, det, Ztr_sub, Zva, yva, min_precision, extra={"model": "LOF", "n_fit": len(Ztr_sub)})
    res = pd.DataFrame([v["row"] for v in unsup.values()]).set_index("config")
    cols = ["AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "F1", "FP", "FN",
            "F1@default", "alerts@default", "t_fit (s)", "t_score (s)"]
    display(res[cols].round(3))
    return unsup, res, prep_scaled


def best_per_family(res, families, pool=None, y_val=None):
    """Best configuration of each family (selected on validation); displays the table.

    If `pool` and `y_val` are given, also prints the maximum precision reachable with >= 10 alerts
    (to qualify a "target not met")."""
    best = {m: select_best(res[res["model"] == m].reset_index().to_dict("records")) for m in families}
    display(pd.DataFrame(best).T[["config", "AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "FP", "FN"]])
    if pool is not None:
        for m, b in best.items():
            s = pool[b["config"]]["scores_val"]
            p, r, t = precision_recall_curve(y_val, s)
            n = _n_alerts(s, t)
            ok = n >= 10
            j = p[:-1][ok].argmax()
            print(f"{m} ({b['config']}): max precision with >= 10 alerts = {p[:-1][ok].max():.3f} "
                  f"(recall {r[:-1][ok][j]:.3f}, {n[ok][j]} alerts)")
    return best


def collect_best_methods(best_sup, best_unsup, sup, unsup):
    """Gather the 4 retained configurations (tree, logistic regression, IF, LOF).

    Returns (best_cfg, pool, scores_best, op_best): names, fitted models, validation scores
    and result rows at the operating threshold."""
    best_cfg = {"Decision tree": best_sup["tree"]["config"], "Logistic regression": best_sup["logreg"]["config"],
                "Isolation Forest": best_unsup["IF"]["config"], "LOF": best_unsup["LOF"]["config"]}
    pool = {**sup, **unsup}
    scores_best = {lab: pool[cfg]["scores_val"] for lab, cfg in best_cfg.items()}
    op_best = {lab: pool[cfg]["row"] for lab, cfg in best_cfg.items()}
    for lab, cfg in best_cfg.items():
        print(f"{lab:22s} -> {cfg}")
    return best_cfg, pool, scores_best, op_best


def report_ap_auc(rows_val):
    """AP / ROC-AUC table and bar chart (validation) of the retained methods."""
    cmp_val = pd.DataFrame({lab: {"AP": r["AP"], "AUC-ROC": r["AUC-ROC"]} for lab, r in rows_val.items()}).T
    display(cmp_val.round(3))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cmp_val.plot.bar(ax=ax, color=["tab:blue", "tab:orange"], rot=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    ax.set(title="AP and ROC-AUC on validation (best configuration of each method)", ylim=(0, 1.1))
    plt.tight_layout()
    return cmp_val


def report_default_vs_operational(rows_val, min_precision=MIN_PRECISION_FRAUD):
    """Default vs retained threshold (validation): table + figure."""
    cmp_thr = pd.DataFrame({lab: {
        "default threshold": r["default threshold"], "alerts (default)": r["alerts@default"],
        "precision (default)": r["precision@default"], "recall (default)": r["recall@default"],
        "F1 (default)": r["F1@default"], "retained threshold": r["threshold"],
        "alerts (retained)": r["n_alerts"], "precision (retained)": r["precision"], "recall (retained)": r["recall"],
        "F1 (retained)": r["F1"], "target met": r["feasible"]} for lab, r in rows_val.items()}).T
    display(cmp_thr.round(3))
    return plot_default_vs_operational(rows_val, min_precision)


def select_and_freeze(rows_val, best_cfg):
    """Select the method on VALIDATION (target met, recall, precision, AP) and freeze model + threshold."""
    selected = select_best([{"method": lab, **r} for lab, r in rows_val.items()])
    frozen = {lab: {"config": best_cfg[lab], "threshold": rows_val[lab]["threshold"]} for lab in best_cfg}
    print(f"Method selected on validation: {selected['method']}  ({selected['config']})")
    print(f"  frozen threshold = {selected['threshold']:.6f} | validation: precision {selected['precision']:.3f}, "
          f"recall {selected['recall']:.3f}, {selected['n_alerts']} alerts, target met = {selected['feasible']}")
    display(pd.DataFrame(frozen).T)
    return selected["method"], frozen


def evaluate_frozen_on_test(pool, frozen, selected, rows_val, cc, prep_scaled, min_precision):
    """Evaluate the 4 frozen configurations ONCE on the test set; displays the final table (★ = selected)."""
    Xte = cc["Xte"]
    inputs = {"Decision tree": ("proba", Xte), "Logistic regression": ("proba", Xte),
              "Isolation Forest": ("score_samples", Xte.to_numpy()), "LOF": ("score_samples", prep_scaled.transform(Xte))}
    test_res = {lab: evaluate_on_test(pool[frozen[lab]["config"]]["model"], inputs[lab][0], inputs[lab][1], cc["yte"],
                                      frozen[lab]["threshold"]) for lab in frozen}
    final = pd.DataFrame([{
        "method": ("★ " if lab == selected else "") + lab, "preprocessing": METHOD_DESCRIPTIONS[lab][0],
        "imbalance treatment / tuning": METHOD_DESCRIPTIONS[lab][1],
        "AP": r["AP"], "AUC-ROC": r["AUC-ROC"], "frozen threshold": frozen[lab]["threshold"],
        "precision": r["precision"], "recall": r["recall"], "F1": r["F1"], "bal. acc.": r["balanced_accuracy"],
        "alerts": r["n_alerts"], "false alarms (FP)": r["FP"], "missed frauds (FN)": r["FN"],
        "fit time (s)": rows_val[lab]["t_fit (s)"], "test scoring time (s)": r["t_score_test (s)"],
        f"constraint ≥ {min_precision} met on test": r["precision"] >= min_precision}
        for lab, r in test_res.items()]).set_index("method")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        display(final.round(3))
    return test_res, final


def report_selected_on_test(test_res, selected, frozen, rows_val, min_precision, unit="fraud"):
    """Interpretation in numbers of observations for the selected method (frozen threshold).
    unit='fraud' (Credit Card) or 'failure' (AI4I maintenance): adapts the wording."""
    words = {"fraud": ("alerts to review", "true frauds", "false alarms", "missed frauds", "true fraud"),
             "failure": ("interventions triggered", "failures prevented", "unnecessary interventions", "missed failures",
                         "failure prevented")}[unit]
    r = test_res[selected]
    (tn, fp), (fn, tp) = r["confusion_matrix"]
    lo_p, hi_p = wilson_interval(tp, tp + fp)
    lo_r, hi_r = wilson_interval(tp, tp + fn)
    print(f"Test: {r['n_total']} observations incl. {r['n_fraud']} positives ({r['n_fraud'] / r['n_total']:.3%}) "
          f"| method: {selected}, frozen threshold {frozen[selected]['threshold']:.6f}")
    print(f"- {words[0]}: {r['n_alerts']} ({r['n_alerts'] / r['n_total']:.3%} of observations) incl. {tp} {words[1]} and {fp} {words[2]}")
    print(f"- detected: {tp} / {r['n_fraud']}; {words[3]}: {fn}")
    print(f"- precision {r['precision']:.3f} (95 % Wilson CI: {lo_p:.3f} - {hi_p:.3f}); recall {r['recall']:.3f} (95 % CI: {lo_r:.3f} - {hi_r:.3f})")
    print(f"- precision constraint >= {min_precision} met on test: {r['precision'] >= min_precision}"
          f" | val -> test: precision {rows_val[selected]['precision']:.3f} -> {r['precision']:.3f}, "
          f"recall {rows_val[selected]['recall']:.3f} -> {r['recall']:.3f}")
    print(f"- on average {r['n_alerts'] / max(tp, 1):.2f} alerts per {words[4]}")


# ------------------------- Part 4: predictive maintenance (AI4I 2020) -------------------------
AI4I_NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
AI4I_CATEGORICAL = ["Type"]
AI4I_FEATURES = AI4I_CATEGORICAL + AI4I_NUMERIC
AI4I_DROPPED = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]   # identifiers and failure types (target leakage)
AI4I_DESCRIPTIONS = {
    "Logistic regression": ("one-hot(Type) + StandardScaler (fitted on train)", "class weighting, C=1 (Credit Card hyperparameters)"),
    "Isolation Forest": ("one-hot(Type), raw numeric features", "100 trees, max_samples=4096 (Credit Card hyperparameters)"),
}


def ai4i_overview(path):
    """Load AI4I 2020; inspect format, missing values, duplicates, failure frequency and types.

    Columns TWF, HDF, PWF, OSF, RNF are shown for DESCRIPTION only; they are not used as inputs."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    print("shape:", df.shape, "| missing values:", int(df.isna().sum().sum()), "| duplicated rows:", int(df.duplicated().sum()))
    print(f"failures (Machine failure = 1): {df['Machine failure'].sum()} / {len(df)} = {df['Machine failure'].mean():.2%}")
    display(df.groupby("Type")["Machine failure"].agg(n="count", failures="sum", rate="mean").round(4))
    types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    print("failure types (descriptive, not used):", df[types].sum().to_dict(),
          "| failures without type:", int(((df["Machine failure"] == 1) & (df[types].sum(1) == 0)).sum()))
    return df


def ai4i_split(df, seed=SEED):
    """Inputs = Type + 5 numeric features (identifiers and failure types removed); target = Machine failure.
    Stratified 60/20/20 split (same function as Credit Card). Returns {Xtr, ytr, Xva, yva, Xte, yte, ...}."""
    X, y = df[AI4I_FEATURES], df["Machine failure"].to_numpy()
    idx_tr, idx_va, idx_te = split_60_20_20(y, seed=seed)
    out = {"df": df, "y": y}
    for tag, i in zip(("tr", "va", "te"), (idx_tr, idx_va, idx_te)):
        out["X" + tag], out["y" + tag] = X.iloc[i], y[i]
    parts = {"train": idx_tr, "validation": idx_va, "test": idx_te}
    assert not (set(idx_tr) & set(idx_va) or set(idx_tr) & set(idx_te) or set(idx_va) & set(idx_te))
    assert not set(AI4I_FEATURES) & set(AI4I_DROPPED)            # no leaking column among the inputs
    display(pd.DataFrame({k: {"observations": len(i), "failures": int(y[i].sum()), "rate": y[i].mean()} for k, i in parts.items()}).T)
    return out


def plot_ai4i_frequency(df):
    """Failure frequency: overall and by product type (full dataset description, not the training set)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(["normal", "failure"], df["Machine failure"].value_counts().sort_index().values, color=["tab:blue", "tab:red"])
    ax.set(title=f"Machine failure: {df['Machine failure'].mean():.2%} failures", ylabel="observations")
    for i, v in enumerate(df["Machine failure"].value_counts().sort_index().values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax = axes[1]
    g = df.groupby("Type")["Machine failure"].agg(["mean", "count"]).loc[["L", "M", "H"]]
    ax.bar(g.index, g["mean"] * 100, color="tab:orange")
    ax.axhline(df["Machine failure"].mean() * 100, color="k", ls="--", lw=1, label="overall rate")
    for i, (m, c) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m * 100, f"{m:.1%}\n(n={c})", ha="center", va="bottom", fontsize=8)
    ax.set(title="Failure rate by product type", ylabel="% failures", ylim=(0, g["mean"].max() * 130))
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def make_ai4i_preprocessor(scaled=True):
    """One-hot on Type (categories seen in training) + StandardScaler on numeric features if scaled
    (logistic regression); raw numeric features otherwise (Isolation Forest). Fit on the training set only."""
    num = StandardScaler() if scaled else "passthrough"
    return ColumnTransformer([("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), AI4I_CATEGORICAL),
                              ("num", num, AI4I_NUMERIC)])


def run_ai4i_methods(ai, min_precision=MIN_PRECISION_MAINTENANCE):
    """Weighted logistic regression (C=1) and Isolation Forest (100 trees, max_samples=4096): Credit Card
    hyperparameters KEPT, models and transformations REFITTED on AI4I. Threshold chosen on validation
    (max recall subject to precision >= min_precision). Returns (res, rows_val, best_cfg); displays the table."""
    lr = Pipeline([("prep", make_ai4i_preprocessor(scaled=True)),
                   ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=SEED))])
    r_lr = evaluate_config("logreg | class_weight | C=1.0", lr, ai["Xtr"], ai["ytr"], ai["Xva"], ai["yva"], min_precision,
                           extra={"model": "logreg"})
    prep_if = make_ai4i_preprocessor(scaled=False).fit(ai["Xtr"])
    iso = IsolationForest(n_estimators=100, max_samples=4096, random_state=SEED, n_jobs=-1)
    r_if = evaluate_detector("IF | n_estimators=100, max_samples=4096", iso, prep_if.transform(ai["Xtr"]),
                             prep_if.transform(ai["Xva"]), ai["yva"], min_precision, extra={"model": "IF"})
    res = {"Logistic regression": r_lr, "Isolation Forest": r_if, "prep_if": prep_if}
    rows_val = {k: res[k]["row"] for k in ("Logistic regression", "Isolation Forest")}
    best_cfg = {k: rows_val[k]["config"] for k in rows_val}
    cols = ["AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "F1", "FP", "FN",
            "alerts@default", "precision@default", "t_fit (s)", "t_score (s)"]
    display(pd.DataFrame(rows_val).T[cols].round(3))
    return res, rows_val, best_cfg


def evaluate_ai4i_on_test(ai, res, frozen, selected, rows_val, min_precision=MIN_PRECISION_MAINTENANCE):
    """Evaluate both methods ONCE on the test set (models and thresholds frozen on validation);
    final table (★ = selected on validation) with unnecessary interventions and missed failures."""
    Xte, yte = ai["Xte"], ai["yte"]
    test_res = {
        "Logistic regression": evaluate_on_test(res["Logistic regression"]["model"], "proba", Xte, yte,
                                                frozen["Logistic regression"]["threshold"]),
        "Isolation Forest": evaluate_on_test(res["Isolation Forest"]["model"], "score_samples", res["prep_if"].transform(Xte), yte,
                                             frozen["Isolation Forest"]["threshold"])}
    final = pd.DataFrame([{
        "method": ("★ " if lab == selected else "") + lab, "preprocessing": AI4I_DESCRIPTIONS[lab][0], "tuning": AI4I_DESCRIPTIONS[lab][1],
        "AP": r["AP"], "AUC-ROC": r["AUC-ROC"], "frozen threshold": frozen[lab]["threshold"],
        "precision": r["precision"], "recall": r["recall"], "F1": r["F1"], "bal. acc.": r["balanced_accuracy"],
        "interventions": r["n_alerts"], "unnecessary interventions (FP)": r["FP"], "missed failures (FN)": r["FN"],
        "fit time (s)": rows_val[lab]["t_fit (s)"], "test scoring time (s)": r["t_score_test (s)"],
        f"constraint ≥ {min_precision} met on test": r["precision"] >= min_precision}
        for lab, r in test_res.items()]).set_index("method")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        display(final.round(3))
    return test_res, final


def compare_thresholds_across_datasets(frozen_cc, frozen_ai):
    """Frozen thresholds of each method on Credit Card and AI4I: numeric values do not transfer
    (different scores and prevalences); each threshold is re-chosen on the dataset's validation set."""
    t = pd.DataFrame({"Credit Card threshold": {k: frozen_cc[k]["threshold"] for k in frozen_ai},
                      "AI4I threshold": {k: frozen_ai[k]["threshold"] for k in frozen_ai}})
    display(t.round(6))
    return t


def ap_vs_prevalence(test_cc, test_ai, methods=("Logistic regression", "Isolation Forest")):
    """Test AP divided by prevalence (the AP of a random score) on Credit Card and AI4I.

    The AP / prevalence ratio ("lift over random") shows that raw APs of the two datasets are not
    comparable: their positive rates differ by a factor of about 20."""
    rows = {}
    for lab in methods:
        for data, res in [("Credit Card", test_cc), ("AI4I", test_ai)]:
            r = res[lab]
            prev = r["n_fraud"] / r["n_total"]
            rows[(lab, data)] = {"test AP": r["AP"], "prevalence (random AP)": prev, "AP / prevalence": r["AP"] / prev}
    t = pd.DataFrame(rows).T
    t.index.names = ["method", "dataset"]
    display(t.round(4))
    return t


# ------------------- Part 3bis: unsupervised methods beyond the course (ECOD, GMM) -------------------
class ECODDetector:
    """ECOD (Li et al., IEEE TKDE 2022) implemented from scratch: Empirical Cumulative distribution
    functions for Outlier Detection.

    Idea: for each feature j, the empirical CDF of the training set gives the left-tail probability
    P(X_j <= x) and the right-tail probability P(X_j >= x). An observation with several features in a rare
    tail is suspicious. Assumption: features roughly independent (the product of marginal probabilities
    approximates the joint tail probability).
    Score = sum over features of -log(tail probability), depending on `tail`:
      'left'/'right': a single tail; 'auto': tail chosen by the sign of the feature's skewness;
      'both': max(sum of left tails, sum of right tails); 'max': max of the three (original ECOD, Eq. 6).
    No learned hyperparameter; invariant to any monotone transformation of the features (no scaling needed).
    scikit-learn-like interface: `fit`, `score_samples` (NEGATED anomaly score: lower = more suspicious), `offset_`.
    """

    def __init__(self, tail="max", contamination=0.1):
        """tail: score construction ('auto', 'both', 'max'...); contamination: prior share of anomalies,
        only used for the default threshold `offset_` (the retained threshold is chosen on validation)."""
        self.tail, self.contamination = tail, contamination

    def fit(self, X):
        """Store, for each feature, the sorted training values (empirical CDF) and the skewness;
        no label is used."""
        X = np.asarray(X, dtype=float)
        self.n_, self.d_ = X.shape
        self.sorted_ = np.sort(X, axis=0)               # one sorted column per feature
        self.skew_ = skew(X, axis=0)
        self.offset_ = -np.quantile(self._anomaly(X), 1 - self.contamination)   # default threshold (prior contamination)
        return self

    def _tail_scores(self, X):
        """(m, d) matrices of -log left- and right-tail probabilities."""
        left = np.empty(X.shape)
        right = np.empty(X.shape)
        for j in range(self.d_):
            col = self.sorted_[:, j]
            p_left = np.searchsorted(col, X[:, j], side="right") / self.n_           # P(X <= x)
            p_right = (self.n_ - np.searchsorted(col, X[:, j], side="left")) / self.n_  # P(X >= x)
            left[:, j] = -np.log(np.clip(p_left, 1.0 / self.n_, None))     # clip: point outside the training range
            right[:, j] = -np.log(np.clip(p_right, 1.0 / self.n_, None))
        return left, right

    def _anomaly(self, X):
        """ECOD anomaly score (higher = more suspicious) according to the `tail` construction."""
        left, right = self._tail_scores(np.asarray(X, dtype=float))
        o_left, o_right = left.sum(1), right.sum(1)
        o_auto = np.where(self.skew_ < 0, left, right).sum(1)
        return {"left": o_left, "right": o_right, "auto": o_auto, "both": np.maximum(o_left, o_right),
                "max": np.maximum(np.maximum(o_left, o_right), o_auto)}[self.tail]

    def score_samples(self, X):
        """scikit-learn convention: negated anomaly score (lower = more suspicious)."""
        return -self._anomaly(X)

    def contributions(self, X):
        """Per-feature contribution to the score (-log of the skewness-selected tail): interpretability."""
        left, right = self._tail_scores(np.asarray(X, dtype=float))
        return np.where(self.skew_ < 0, left, right)


class GMMDetector:
    """Gaussian mixture (EM): anomaly score = -log-likelihood. scikit-learn-like interface
    (`score_samples` = log-likelihood, lower = more suspicious; `offset_` = default threshold at prior contamination)."""

    def __init__(self, n_components=5, covariance_type="full", contamination=0.1, seed=SEED):
        """n_components: number of Gaussians; covariance_type: 'full' or 'diag';
        contamination: only used for the default threshold `offset_`."""
        self.n_components, self.covariance_type = n_components, covariance_type
        self.contamination, self.seed = contamination, seed

    def fit(self, X):
        """Fit the Gaussian mixture by EM on the training set, without labels."""
        self.gmm_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                    random_state=self.seed, reg_covar=1e-4).fit(X)
        self.offset_ = np.quantile(self.gmm_.score_samples(X), self.contamination)
        return self

    def score_samples(self, X):
        """Log-likelihood of each observation (lower = less likely = more suspicious)."""
        return self.gmm_.score_samples(X)


def run_new_methods(cc, min_precision=MIN_PRECISION_FRAUD, prep_scaled=None):
    """ECOD (3 score variants, raw features) and GMM (3 configurations, scaled features),
    fitted on the full training set WITHOUT labels; threshold chosen on validation (same rule).
    Returns (new, res_new) and displays the validation table."""
    Xtr, Xva, yva = cc["Xtr"], cc["Xva"], cc["yva"]
    if prep_scaled is None:
        prep_scaled = make_preprocessor(scaled=True).fit(Xtr)
    Ztr, Zva = prep_scaled.transform(Xtr), prep_scaled.transform(Xva)
    new = {}
    for tail in ("auto", "both", "max"):
        name = f"ECOD | tail={tail}"
        new[name] = evaluate_detector(name, ECODDetector(tail=tail), Xtr.to_numpy(), Xva.to_numpy(), yva, min_precision,
                                      extra={"model": "ECOD", "n_fit": len(Xtr)})
    for k, cov in ((3, "full"), (8, "full"), (8, "diag")):
        name = f"GMM | n_components={k}, covariance={cov}"
        new[name] = evaluate_detector(name, GMMDetector(k, cov), Ztr, Zva, yva, min_precision,
                                      extra={"model": "GMM", "n_fit": len(Ztr)})
    res = pd.DataFrame([v["row"] for v in new.values()]).set_index("config")
    cols = ["AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "F1", "FP", "FN",
            "F1@default", "alerts@default", "t_fit (s)", "t_score (s)"]
    display(res[cols].round(3))
    return new, res


def collect_best_detectors(best_unsup, best_new, unsup, new):
    """Best configuration of each unsupervised detector (IF, LOF, ECOD, GMM):
    (names, models, validation scores, rows at the operating threshold)."""
    pool = {**unsup, **new}
    cfg = {"Isolation Forest": best_unsup["IF"]["config"], "LOF": best_unsup["LOF"]["config"],
           "ECOD": best_new["ECOD"]["config"], "GMM": best_new["GMM"]["config"]}
    return cfg, pool, {l: pool[c]["scores_val"] for l, c in cfg.items()}, {l: pool[c]["row"] for l, c in cfg.items()}


def plot_ecod_contributions(detector, X_val, y_val, top=12):
    """Mean contribution of each feature to the ECOD score (validation): frauds vs normal."""
    contrib = detector.contributions(X_val.to_numpy())
    names = list(X_val.columns)
    fr, no = contrib[y_val == 1].mean(0), contrib[y_val == 0].mean(0)
    idx = np.argsort(fr - no)[::-1][:top]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(idx))
    ax.bar(x - .2, no[idx], .4, label="normal (mean)", color="tab:blue")
    ax.bar(x + .2, fr[idx], .4, label="fraud (mean)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in idx])
    ax.set(title=f"ECOD: mean contribution per feature (-log tail probability), top {top} most discriminative features (validation)",
           ylabel="-log P(tail)")
    ax.legend()
    plt.tight_layout()
    return fig


def collect_detectors(bests, pool):
    """Best configuration of each family, across families.

    bests: list of dicts {family: row} (outputs of `best_per_family`); pool: all fitted models.
    Returns (cfg, validation scores, rows at the operating threshold) indexed by family."""
    cfg = {fam: row["config"] for b in bests for fam, row in b.items()}
    return cfg, {f: pool[c]["scores_val"] for f, c in cfg.items()}, {f: pool[c]["row"] for f, c in cfg.items()}


# ------------------------- Part 3: EasyEnsemble (supervised, imbalance) -------------------------
def run_easyensemble(cc, min_precision=MIN_PRECISION_FRAUD):
    """EasyEnsemble (Liu, Wu, Zhou, 2009; `imblearn.ensemble.EasyEnsembleClassifier`).

    Idea: draw `n_estimators` subsamples of the training set keeping ALL frauds and a random draw
    (without replacement) of the same number of normal rows (1:1 ratio); train an AdaBoost (decision
    stumps) on each and average the probabilities. Undersampling only affects the training set.
    Configurations (3): (number of subsamples, number of AdaBoost estimators) =
    (10, 50), (30, 50), (10, 200). Threshold and configuration chosen on validation (same rule)."""
    ee = {}
    for n_bags, n_ada in ((10, 50), (30, 50), (10, 200)):
        name = f"EasyEnsemble | bags={n_bags}, AdaBoost={n_ada}"
        est = EasyEnsembleClassifier(n_estimators=n_bags, estimator=AdaBoostClassifier(n_estimators=n_ada, random_state=SEED),
                                     random_state=SEED, n_jobs=-1)
        ee[name] = evaluate_config(name, est, cc["Xtr"], cc["ytr"], cc["Xva"], cc["yva"], min_precision,
                                   extra={"model": "EE", "imbalance": "undersampling x bags"})
    res = pd.DataFrame([v["row"] for v in ee.values()]).set_index("config")
    cols = ["AP", "AUC-ROC", "threshold", "precision", "recall", "n_alerts", "feasible", "F1", "FP", "FN",
            "alerts@default", "precision@default", "t_fit (s)", "t_score (s)"]
    display(res[cols].round(3))
    return ee, res


def evaluate_part3_on_test(ee, best_ee, test_res, rows_val, cc, min_precision=MIN_PRECISION_FRAUD):
    """Part 3: evaluate ONCE on the test set the EasyEnsemble configuration retained on validation
    (frozen model and threshold) and compare it with IF and LOF (already evaluated with frozen thresholds).

    best_ee: retained row ({'config', 'threshold', ...}); rows_val: validation rows {label: row}
    for 'EasyEnsemble', 'Isolation Forest', 'LOF'. Returns (test_res3, validation vs test table)."""
    r = evaluate_on_test(ee[best_ee["config"]]["model"], "proba", cc["Xte"], cc["yte"], best_ee["threshold"])
    test_res3 = {"EasyEnsemble": r, "Isolation Forest": test_res["Isolation Forest"], "LOF": test_res["LOF"]}
    rows = []
    for lab, t in test_res3.items():
        v = rows_val[lab]
        rows.append({"method": lab, "val AP": v["AP"], "test AP": t["AP"], "val AUC": v["AUC-ROC"], "test AUC": t["AUC-ROC"],
                     "val precision": v["precision"], "test precision": t["precision"],
                     "val recall": v["recall"], "test recall": t["recall"], "test F1": t["F1"],
                     "test alerts": t["n_alerts"], "test FP": t["FP"], "test FN": t["FN"],
                     "fit time (s)": v["t_fit (s)"], "test scoring (s)": t["t_score_test (s)"],
                     f"constraint ≥ {min_precision} (test)": t["precision"] >= min_precision})
    table = pd.DataFrame(rows).set_index("method")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        display(table.round(3))
    return test_res3, table


# ------------------- Part 5.1: robustness to contamination of the normal reference (Mouse) -------------------
def generate_synthetic_anomalies(X_ref, n, kind, rng, min_dist_factor=3.0, local_sigma=0.1):
    """Synthetic anomalies built WITHOUT labels, from the reference `X_ref` only.

    kind='uniform': points drawn uniformly in the bounding box of X_ref enlarged by 20 % on each side
                    ("background" anomalies, spread everywhere, including near the true anomalies).
    kind='local'  : random normal point + isotropic Gaussian noise with std `local_sigma` (0.1:
                    large perturbations around the shape).
    In both cases, points closer than `min_dist_factor` times the median nearest-neighbour distance
    of the reference are rejected (so that they really lie outside the dense area)."""
    nn = NearestNeighbors(n_neighbors=2).fit(X_ref)
    d_min = min_dist_factor * np.median(nn.kneighbors(X_ref)[0][:, 1])
    lo, hi = X_ref.min(0), X_ref.max(0)
    span = hi - lo
    out = []
    while len(out) < n:
        if kind == "uniform":
            cand = rng.uniform(lo - 0.2 * span, hi + 0.2 * span, size=(4 * max(n, 1), X_ref.shape[1]))
        elif kind == "local":
            cand = X_ref[rng.randint(len(X_ref), size=4 * max(n, 1))] + rng.normal(0, local_sigma, size=(4 * max(n, 1), X_ref.shape[1]))
        else:
            raise ValueError(kind)
        out.extend(cand[nn.kneighbors(cand, n_neighbors=1)[0][:, 0] > d_min].tolist())
    return np.array(out[:n])


def contaminate_training(X_train, frac, kind, seed):
    """Replace a fraction `frac` of the normal training observations with synthetic anomalies (same size).
    Returns (contaminated X, replaced indices)."""
    rng = np.random.RandomState(seed)
    n_inj = int(round(frac * len(X_train)))
    idx = rng.choice(len(X_train), n_inj, replace=False)
    Xc = X_train.copy()
    if n_inj:
        Xc[idx] = generate_synthetic_anomalies(X_train, n_inj, kind, rng)
    return Xc, idx


def mouse_robustness(X, y, fracs=(0.0, 0.05, 0.10), seeds=(42, 43, 44), kinds=("uniform", "local"), n_neighbors=20):
    """IF (100 trees) and LOF (k=20, novelty=True), FIXED hyperparameters, refitted on a normal training set
    contaminated at 0 / 5 / 10 %; evaluated on the SAME real held-out test set (no injection in the test).
    Returns a long table (generator, contamination, seed, method, AP, AUC) and displays the summary."""
    idx_train, idx_test = split_novelty(X, y, normal_test_frac=0.2, seed=SEED)      # same split as in 1.3
    X_tr, X_te, y_te = X[idx_train], X[idx_test], y[idx_test]
    rows = []
    for kind in kinds:
        for frac in fracs:
            for seed in seeds:
                Xc, _ = contaminate_training(X_tr, frac, kind, seed)
                iso = IsolationForest(n_estimators=100, random_state=seed).fit(Xc)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(Xc)
                for name, det in (("Isolation Forest", iso), (f"LOF (k={n_neighbors})", lof)):
                    s = orient_scores(det.score_samples(X_te), higher_is_anomaly=False)
                    rows.append({"generator": kind, "contamination": frac, "seed": seed, "method": name, **ranking_metrics(y_te, s)})
    df = pd.DataFrame(rows)
    summary = df.groupby(["generator", "method", "contamination"])["AP"].agg(["mean", "std", "min", "max"]).round(3)
    display(summary)
    return df


def plot_mouse_robustness(df):
    """AP (mean, std over seeds, dots = seeds) as a function of contamination, per generator."""
    kinds = list(df["generator"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.5 * len(kinds), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        for (name, c) in zip(df["method"].unique(), ("tab:orange", "tab:red")):
            d = df[(df["generator"] == kind) & (df["method"] == name)]
            g = d.groupby("contamination")["AP"].agg(["mean", "std"])
            ax.errorbar(g.index * 100, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=4, color=c, label=name)
            ax.scatter(d["contamination"] * 100, d["AP"], color=c, alpha=.35, s=22)
        ax.set(title=f"Synthetic anomalies: '{kind}'", xlabel="contamination of the normal training set (%)",
               xticks=sorted(df["contamination"].unique() * 100))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("AP on the real test set (mean ± std, 3 seeds)")
    plt.tight_layout()
    return fig


def plot_mouse_contamination_examples(X, y, kinds=("uniform", "local"), fracs=(0.0, 0.05, 0.10), seed=42):
    """Contaminated training set (one seed): normal points, injected synthetic anomalies, and the true test
    anomalies (not used for training) to visualise how close they are."""
    idx_train, idx_test = split_novelty(X, y, normal_test_frac=0.2, seed=SEED)
    X_tr, X_te, y_te = X[idx_train], X[idx_test], y[idx_test]
    fig, axes = plt.subplots(len(kinds), len(fracs), figsize=(5 * len(fracs), 4.8 * len(kinds)))
    for r, kind in enumerate(kinds):
        for c, frac in enumerate(fracs):
            ax = axes[r, c]
            Xc, idx = contaminate_training(X_tr, frac, kind, seed)
            keep = np.ones(len(Xc), bool)
            keep[idx] = False
            ax.scatter(*Xc[keep].T, s=10, c="tab:blue", alpha=.5, label="normal (training)")
            if len(idx):
                ax.scatter(*Xc[idx].T, s=40, c="tab:orange", marker="^", label=f"injected synthetic ({len(idx)})")
            ax.scatter(*X_te[y_te == 1].T, s=70, c="tab:red", marker="x", label="true anomaly (test)")
            ax.set(title=f"'{kind}', contamination {frac:.0%}", aspect="equal")
            ax.legend(fontsize=7, loc="lower center")
    plt.tight_layout()
    return fig


# ------------------- Part 5.2: SHAP explanations of Isolation Forest (Credit Card) -------------------
def _if_path_length(model, X):
    """Mean path length E[h(x)] of a sklearn IsolationForest, recovered from `score_samples`
    (score = -2^(-E[h]/c(psi)))."""
    from sklearn.ensemble._iforest import _average_path_length
    c = float(_average_path_length(np.array([model.max_samples_]))[0])
    return -np.log2(-model.score_samples(X)) * c


def shap_isolation_forest(model, X_train, X_test, y_test, threshold, n_background=50, per_group=2, seed=SEED):
    """SHAP explanations of Isolation Forest for 6 test observations (2 detected frauds, 2 false alarms,
    2 missed frauds; filled with other cases if a category is missing).

    - Background: `n_background` observations drawn at random from the TRAINING set only.
    - `TreeExplainer` explains the MEAN PATH LENGTH E[h(x)] (depth + leaf-size correction): the longer it
      is, the more NORMAL the observation. Contributions therefore have the OPPOSITE sign of a contribution
      to suspiciousness: we return `contrib_susp = -shap` (positive = makes it more suspicious).
    - Checks: (1) additivity on the path length; (2) consistency with `KernelExplainer` applied directly
      to the final anomaly score (different scale, same signs expected).
    """
    import shap
    rng = np.random.RandomState(seed)
    score = orient_scores(model.score_samples(np.asarray(X_test)), higher_is_anomaly=False)   # anomaly score (higher = more suspicious)
    alert = score >= threshold
    y_test = np.asarray(y_test)
    groups = {"detected fraud": np.flatnonzero(alert & (y_test == 1)),
              "false alarm": np.flatnonzero(alert & (y_test == 0)),
              "missed fraud": np.flatnonzero(~alert & (y_test == 1))}
    chosen, labels, missing = [], [], []
    for name, pool_idx in groups.items():
        take = rng.choice(pool_idx, size=min(per_group, len(pool_idx)), replace=False) if len(pool_idx) else np.array([], int)
        chosen += list(take)
        labels += [name] * len(take)
        if len(take) < per_group:
            missing.append(name)
    if missing:          # missing or too small category: fill with other cases (detected/missed frauds first)
        rest = [i for i in np.flatnonzero(y_test == 1) if i not in chosen]
        while len(chosen) < 3 * per_group and rest:
            i = rest.pop(0)
            chosen.append(i)
            labels.append("extra")
    chosen = np.array(chosen)
    Xe = np.asarray(X_test)[chosen]
    bg = np.asarray(X_train)[rng.choice(len(X_train), n_background, replace=False)]
    names = list(X_test.columns) if hasattr(X_test, "columns") else [f"x{j}" for j in range(Xe.shape[1])]

    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    sv = np.asarray(explainer.shap_values(Xe, check_additivity=False))
    base = float(np.ravel(explainer.expected_value)[0])
    f_model = _if_path_length(model, Xe)
    f_bg = _if_path_length(model, bg).mean()
    checks = pd.DataFrame({"model E[h]": f_model, "base + Σ SHAP": base + sv.sum(1),
                           "mean E[h] background": f_bg, "anomaly score": score[chosen]}, index=labels)

    # independent check: KernelExplainer on the final anomaly score
    f_anom = lambda A: orient_scores(model.score_samples(A), higher_is_anomaly=False)
    ker = shap.KernelExplainer(f_anom, bg)
    np.random.seed(seed)                                  # KernelExplainer samples at random: fixed seed
    sv_k = np.asarray(ker.shap_values(Xe, nsamples=1000, silent=True))
    from scipy.stats import spearmanr
    agree = [spearmanr(-sv[i], sv_k[i])[0] for i in range(len(chosen))]
    sign_match = [float(np.mean(np.sign(-sv[i][np.argsort(-np.abs(sv_k[i]))[:5]]) == np.sign(sv_k[i][np.argsort(-np.abs(sv_k[i]))[:5]])))
                  for i in range(len(chosen))]
    return {"idx": chosen, "labels": labels, "names": names, "contrib_susp": -sv, "shap_raw": sv, "shap_kernel": sv_k,
            "base": base, "checks": checks, "score": score[chosen], "missing": missing,
            "spearman_tree_vs_kernel": agree, "sign_match_top5": sign_match, "y": y_test[chosen]}


def plot_shap_if(res, top=8):
    """For each observation: the `top` features with the largest contributions to suspiciousness
    (red: more suspicious; blue: more normal)."""
    n = len(res["idx"])
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(5.3 * int(np.ceil(n / 2)), 7.5))
    for ax, k in zip(np.ravel(axes), range(n)):
        v = res["contrib_susp"][k]
        o = np.argsort(-np.abs(v))[:top][::-1]
        ax.barh([res["names"][j] for j in o], v[o], color=["tab:red" if x > 0 else "tab:blue" for x in v[o]])
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{res['labels'][k]} (obs. {res['idx'][k]}, score {res['score'][k]:.2f})", fontsize=9)
        ax.set_xlabel("contribution to suspiciousness")
    plt.tight_layout()
    return fig


def plot_shap_heatmap(res, top=12):
    """Contributions (to suspiciousness) of the most influential features, observation by observation."""
    tot = np.abs(res["contrib_susp"]).sum(0)
    cols = np.argsort(-tot)[:top]
    M = res["contrib_susp"][:, cols]
    lim = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(top))
    ax.set_xticklabels([res["names"][j] for j in cols])
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{l} (obs. {i})" for l, i in zip(res["labels"], res["idx"])])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="contribution to suspiciousness (red = more suspicious)")
    ax.set_title("SHAP / Isolation Forest: per-feature contributions (6 test observations)")
    plt.tight_layout()
    return fig


def report_shap_checks(res):
    """Checks of the SHAP explanations: additivity on the path length and consistency with KernelExplainer."""
    display(res["checks"].round(3))
    ok = np.allclose(res["checks"]["model E[h]"], res["checks"]["base + Σ SHAP"], atol=1e-3)
    print("Output explained by TreeExplainer = mean path length E[h(x)]; base + Σ SHAP = E[h(x)] for the 6 observations:", ok)
    print("Rank correlation (-tree SHAP vs kernel SHAP on the final score):", np.round(res["spearman_tree_vs_kernel"], 2))
    print("Sign agreement on the 5 most influential features:", np.round(res["sign_match_top5"], 2))
    print("Missing categories filled with other cases:", res["missing"] or "none")
    top = np.argsort(-np.abs(res["contrib_susp"]), axis=1)[:, :3]
    return pd.DataFrame({f"{l} (obs. {i})": ", ".join(f"{res['names'][j]} ({res['contrib_susp'][k][j]:+.2f})" for j in top[k])
                         for k, (l, i) in enumerate(zip(res["labels"], res["idx"]))}, index=["top 3 features"]).T


# ---------------- TEST-set plots (thresholds frozen on validation; nothing on the training set) ----------------
def plot_val_vs_test(val_rows, test_rows, min_precision):
    """Precision and recall at the frozen threshold: validation (where the threshold was chosen) vs test."""
    names = list(val_rows)
    x = np.arange(len(names))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 3.4 * len(names)), 4.6))
    for ax, key, title in zip(axes, ["precision", "recall"], ["Precision", "Recall"]):
        ax.bar(x - w / 2, [val_rows[n][key] for n in names], w, label="validation", color="tab:blue")
        ax.bar(x + w / 2, [test_rows[n][key] for n in names], w, label="test (frozen threshold)", color="tab:orange")
        for i, n in enumerate(names):
            ax.text(i - w / 2, val_rows[n][key] + .01, f"{val_rows[n][key]:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, test_rows[n][key] + .01, f"{test_rows[n][key]:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set(title=title + ": validation vs test", ylim=(0, 1.12))
        if key == "precision":
            ax.axhline(min_precision, color="k", ls="--", lw=1, label=f"constraint {min_precision}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(cms, title):
    """Confusion matrices (counts) side by side; cms: {name: 2x2 array [[TN, FP], [FN, TP]]}.
    Colours use a log scale so that the large "normal" cell does not dominate the others."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.9))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm)
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                        color="white" if np.log1p(cm[i, j]) > 0.6 * np.log1p(cm.max()) else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no alert", "alert"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["actual normal", "actual positive"])
        ax.set_title(name, fontsize=10)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def test_rows(test_res, min_precision):
    """Test precision, recall and constraint check, in the format expected by the plots."""
    return {lab: {"precision": r["precision"], "recall": r["recall"], "feasible": r["precision"] >= min_precision}
            for lab, r in test_res.items()}


def report_test_graphs(test_res, rows_val, y_test, min_precision, subject="Credit Card"):
    """Three plots on the TEST set (models and thresholds frozen on validation):
    confusion matrices, validation vs test precision/recall, test ROC and precision-recall curves."""
    rows_te = test_rows(test_res, min_precision)
    plot_confusion_matrices({lab: r["confusion_matrix"] for lab, r in test_res.items()},
                            f"{subject}: confusion matrices on the test set (frozen thresholds; log colour scale)")
    plt.show()
    plot_val_vs_test({lab: rows_val[lab] for lab in test_res}, rows_te, min_precision)
    plt.show()
    plot_pr_roc({lab: r["scores"] for lab, r in test_res.items()}, y_test, min_precision, rows_te, split="test")
    plt.show()
    return rows_te
