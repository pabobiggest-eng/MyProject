from ucimlrepo import fetch_ucirepo
import pandas as pd

data = fetch_ucirepo(id=296)
df = pd.concat([data.data.features, data.data.targets], axis=1)

# CSV mein save karo (agli baar isi se load karenge)
df.to_csv("diabetic_data.csv", index=False)

# Data ko samjho
print("Shape:", df.shape)
print("\nTarget (readmitted):")
print(df["readmitted"].value_counts())

print("\nStay days (regression target):")
print(df["time_in_hospital"].describe())

print("\nMissing values (percent):")
missing = (df.isnull().mean() * 100).sort_values(ascending=False)
print(missing[missing > 0].round(1))
"""
AI-Powered Clinical Decision Support Tool
==========================================
Dataset : Diabetes 130-US Hospitals (1999-2008), UCI repository (id=296)
Task 1  : CLASSIFICATION -> will the patient be readmitted?
Task 2  : REGRESSION     -> how many days will the patient stay in hospital?

Run:  python clinical_project.py
All results (tables, plots, models) are saved automatically.
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# CONFIG  (change these if needed)
# ----------------------------------------------------------------------------
TARGET_MODE = "any"          # "any"     = readmitted at all (<30 or >30)  -> Yes/No
                             # "early30" = readmitted within 30 days only  -> very imbalanced
FIRST_ENCOUNTER_ONLY = True  # keep 1 row per patient (avoids data leakage between train/test)
THRESHOLD_STRATEGY = "macro_f1"  # "macro_f1" = best overall balance of both classes
                             # "balanced_pr" = precision ~= recall (smallest gap between them)
TUNE = True                  # hyper-parameter search for the boosting model (takes a few minutes)
CV_FOLDS = 3
TEST_SIZE = 0.20
RANDOM_STATE = 42

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = BASE / "models"

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
# discharge codes: expired / hospice -> these patients cannot be readmitted
EXPIRED_HOSPICE = [11, 13, 14, 19, 20, 21]


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def save_fig(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   [plot saved] results/figures/{name}")


def md_table(df, floatfmt="{:.4f}"):
    """Small markdown-table writer (no extra package needed)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_data():
    cache = DATA_DIR / "diabetes_130_raw.csv"
    if cache.exists():
        print(f"Loading cached data: {cache}")
        return pd.read_csv(cache, low_memory=False, na_values="?")
    print("Downloading dataset from UCI (needs internet, one time only)...")
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=296)
    raw = pd.concat([d.data.ids, d.data.features, d.data.targets], axis=1)
    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(cache, index=False)
    print(f"Saved a local copy: {cache}")
    return raw


# ----------------------------------------------------------------------------
# 2. CLEANING + FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def diag_group(code):
    """Group ICD-9 diagnosis codes into broad disease categories."""
    if pd.isna(code):
        return "Unknown"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def add_features(df):
    """Engineered features. Safe to call again on new patient data (used by the app too)."""
    df = df.copy()
    meds = [c for c in MED_COLS if c in df.columns]
    if meds:
        df["num_meds_used"] = (df[meds] != "No").sum(axis=1)
        df["num_meds_changed"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    return df


def clean_data(raw):
    log = []
    df = raw.copy()
    log.append(f"Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # a) first encounter per patient only (a patient with many visits would
    #    otherwise appear in both train and test -> fake high scores)
    if FIRST_ENCOUNTER_ONLY and {"patient_nbr", "encounter_id"} <= set(df.columns):
        before = len(df)
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
        log.append(f"Kept first encounter per patient: removed {before - len(df):,} repeat visits")
    df = df.drop(columns=[c for c in ["encounter_id", "patient_nbr"] if c in df.columns])

    # b) patients who died / went to hospice cannot be readmitted -> remove
    dd = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    before = len(df)
    df = df[~dd.isin(EXPIRED_HOSPICE)]
    log.append(f"Removed expired/hospice patients: {before - len(df):,} rows")

    # c) invalid gender
    before = len(df)
    df = df[df["gender"].isin(["Male", "Female"])]
    log.append(f"Removed invalid gender rows: {before - len(df):,}")

    # d0) for these two lab columns "None" means "test not done" -> a real category, not missing
    for c in ["max_glu_serum", "A1Cresult"]:
        if c in df.columns:
            df[c] = df[c].fillna("Not_tested")

    # d) drop columns with >50% missing values
    miss = df.isnull().mean()
    too_missing = miss[miss > 0.5].index.tolist()
    df = df.drop(columns=too_missing)
    log.append(f"Dropped columns with >50% missing: {too_missing}")

    # e) drop constant columns (single value, no information)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    log.append(f"Dropped constant columns: {const}")

    # f) age bucket '[40-50)' -> number 45
    lo = df["age"].str.extract(r"\[(\d+)-")[0].astype(float)
    df["age_num"] = lo + 5
    df = df.drop(columns=["age"])

    # g) ICD-9 codes -> disease groups
    for c in ["diag_1", "diag_2", "diag_3"]:
        if c in df.columns:
            df[c + "_group"] = df[c].map(diag_group)
            df = df.drop(columns=[c])

    # h) id-like columns are categories, not numbers
    for c in CODE_COLS:
        df[c] = df[c].astype(str)

    # i) remaining missing values
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")
    log.append("Missing categorical values -> 'Unknown' / 'Not_tested' (numeric ones imputed inside the model pipeline)")

    # j) engineered features
    df = add_features(df)
    log.append("Added features: num_meds_used, num_meds_changed, total_prior_visits")

    # k) targets
    if TARGET_MODE == "early30":
        df["target_clf"] = (df["readmitted"] == "<30").astype(int)
    else:
        df["target_clf"] = (df["readmitted"] != "NO").astype(int)

    df = df.reset_index(drop=True)
    log.append(f"Clean data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, log


# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
def eda_missing(raw):
    m = raw.replace("?", np.nan).isnull().mean() * 100
    m = m[m > 0].sort_values(ascending=False)
    if m.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(x=m.values, y=m.index, color="#c0392b")
    plt.xlabel("% missing")
    plt.title("Missing values per column (raw data)")
    save_fig("01_missing_values.png")


def eda_target(df):
    vc = df["readmitted"].value_counts()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(x=vc.index, y=vc.values, ax=ax[0], palette="viridis")
    ax[0].set_title("Original readmitted classes")
    vb = df["target_clf"].value_counts().sort_index()
    labels = ["No", "Yes"] if TARGET_MODE == "any" else ["Not <30 days", "<30 days"]
    sns.barplot(x=labels, y=vb.values, ax=ax[1], palette="magma")
    ax[1].set_title(f"Classification target ({TARGET_MODE})")
    ax[2].pie(vb.values, labels=labels, autopct="%1.1f%%", colors=["#3498db", "#e74c3c"])
    ax[2].set_title("Class balance")
    save_fig("02_target_distribution.png")
    ratio = vb.max() / vb.min()
    print(f"   Class counts: {dict(zip(labels, vb.values))}  | imbalance ratio = {ratio:.2f} : 1")
    return ratio


def skew_kurt_table(X, skew_thresh=1.0):
    num = X.select_dtypes(include="number")
    rows = []
    for c in num.columns:
        s0, k0 = num[c].skew(), num[c].kurt()
        if abs(s0) > skew_thresh and num[c].min() >= 0:
            t = np.log1p(num[c])
            s1, k1, applied = t.skew(), t.kurt(), "log1p"
        else:
            s1, k1, applied = s0, k0, "-"
        rows.append([c, s0, k0, applied, s1, k1])
    return pd.DataFrame(rows, columns=["feature", "skew_before", "kurtosis_before",
                                       "applied", "skew_after", "kurtosis_after"])


def eda_skew_plots(df, sk):
    cols = sk["feature"].tolist()
    n = len(cols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    for ax, c in zip(axes.ravel(), cols):
        r = sk[sk.feature == c].iloc[0]
        sns.histplot(df[c], bins=30, ax=ax, color="#2980b9")
        ax.set_title(f"{c}\nskew={r.skew_before:.2f} kurt={r.kurtosis_before:.2f}", fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    save_fig("03_distributions_skew_kurtosis_before.png")

    tcols = sk[sk["applied"] == "log1p"]["feature"].tolist()
    if tcols:
        fig, axes = plt.subplots(2, len(tcols), figsize=(3.6 * len(tcols), 6), squeeze=False)
        for i, c in enumerate(tcols):
            r = sk[sk.feature == c].iloc[0]
            sns.histplot(df[c], bins=30, ax=axes[0, i], color="#c0392b")
            axes[0, i].set_title(f"{c} (before)\nskew={r.skew_before:.2f}", fontsize=9)
            sns.histplot(np.log1p(df[c]), bins=30, ax=axes[1, i], color="#27ae60")
            axes[1, i].set_title(f"log1p({c}) (after)\nskew={r.skew_after:.2f}", fontsize=9)
            axes[0, i].set_xlabel("")
            axes[1, i].set_xlabel("")
        save_fig("04_skew_fix_log_transform.png")


def eda_boxplots(df, cols):
    fig, axes = plt.subplots(2, int(np.ceil(len(cols) / 2)), figsize=(16, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(y=df[c], ax=ax, color="#f39c12")
        ax.set_title(c, fontsize=9)
        ax.set_ylabel("")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    plt.suptitle("Outlier check (boxplots)", y=1.02)
    save_fig("05_boxplots_outliers.png")


def eda_correlation(df):
    num = df.select_dtypes(include="number").copy()
    plt.figure(figsize=(11, 9))
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3)
    plt.title("Correlation heatmap (numeric features + targets)")
    save_fig("06_correlation_heatmap.png")
    pairs = []
    cols = [c for c in corr.columns if c not in ("target_clf",)]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.6:
                pairs.append((a, b, float(corr.loc[a, b])))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, t, title in [(ax[0], "target_clf", "Correlation with readmission"),
                        (ax[1], "time_in_hospital", "Correlation with length of stay")]:
        s = corr[t].drop(["target_clf", "time_in_hospital"], errors="ignore").sort_values()
        sns.barplot(x=s.values, y=s.index, ax=a, color="#8e44ad")
        a.set_title(title)
    save_fig("07_correlation_with_targets.png")
    return pairs


def eda_rates(df):
    cols = ["age_num", "diag_1_group", "A1Cresult", "insulin", "admission_type_id", "number_inpatient"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, c in zip(axes.ravel(), cols):
        g = df.groupby(c)["target_clf"].agg(["mean", "count"])
        g = g[g["count"] >= 50]
        if c == "number_inpatient":
            g = g.loc[g.index <= 8]
        g = g.sort_index() if c in ("age_num", "number_inpatient") else g.sort_values("mean")
        sns.barplot(x=g.index.astype(str), y=g["mean"], ax=ax, color="#16a085")
        ax.set_title(f"Readmission rate by {c}", fontsize=10)
        ax.set_ylabel("rate")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    save_fig("08_readmission_rate_by_feature.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=df["time_in_hospital"], ax=ax[0], color="#2980b9")
    ax[0].set_title("Length of stay (regression target)")
    sns.boxplot(x=df["age_num"], y=df["time_in_hospital"], ax=ax[1], color="#e67e22")
    ax[1].set_title("Length of stay by age")
    save_fig("09_length_of_stay.png")


# ----------------------------------------------------------------------------
# 4. PIPELINES
# ----------------------------------------------------------------------------
def build_preprocessor(X, skewed_cols):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    skew_cols = [c for c in skewed_cols if c in num_cols]
    plain_cols = [c for c in num_cols if c not in skew_cols]
    plain = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    skew = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                        sparse_output=False)
    return ColumnTransformer([("num", plain, plain_cols), ("skew", skew, skew_cols),
                              ("cat", cat, cat_cols)])


def get_models(task, pos_weight=1.0):
    if task == "clf":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=30, max_features="sqrt",
                n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, class_weight="balanced",
                random_state=RANDOM_STATE),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                n_jobs=-1, random_state=RANDOM_STATE)
        except ImportError:
            print("   (xgboost not installed -> skipped. Optional: pip install xgboost)")
    else:
        models = {
            "Ridge (Linear)": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=15, max_features=0.5,
                n_jobs=-1, random_state=RANDOM_STATE),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE)
        except ImportError:
            pass
    return models


def tuned_booster(task, pre):
    base = (HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
            if task == "clf" else HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    pipe = Pipeline([("pre", pre), ("model", base)])
    grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0.0, 0.5, 1.0],
        "model__max_iter": [200, 300, 500],
    }
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE) if task == "clf" \
        else KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(pipe, grid, n_iter=8, cv=cv,
                              scoring="roc_auc" if task == "clf" else "r2",
                              random_state=RANDOM_STATE, n_jobs=1, refit=True)


# ----------------------------------------------------------------------------
# 5. CLASSIFICATION
# ----------------------------------------------------------------------------
def clf_metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def pick_threshold(y, proba):
    """Choose the decision threshold on out-of-fold TRAIN predictions (never on test data)."""
    ths = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y, (proba >= t).astype(int), average="macro") for t in ths]
    if THRESHOLD_STRATEGY == "balanced_pr":
        # smallest |precision - recall|, but only among thresholds that are still useful
        # (macro-F1 within 95% of the best one) -> avoids degenerate "predict nobody" thresholds
        floor = 0.95 * max(scores)
        gaps = []
        for t, sc in zip(ths, scores):
            pr = (proba >= t).astype(int)
            p_, r_ = precision_score(y, pr, zero_division=0), recall_score(y, pr, zero_division=0)
            gaps.append(abs(p_ - r_) if (sc >= floor and p_ > 0 and r_ > 0) else np.inf)
        best = float(ths[int(np.argmin(gaps))])
    else:
        best = float(ths[int(np.argmax(scores))])
    return best, ths, scores


def bootstrap_clf(y, proba, thr, n=300):
    """95% confidence intervals for the test metrics (bootstrap resampling)."""
    rng = np.random.RandomState(RANDOM_STATE)
    y, proba = np.asarray(y), np.asarray(proba)
    vals = {"accuracy": [], "f1": [], "macro_f1": [], "roc_auc": []}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        pr = (proba[i] >= thr).astype(int)
        vals["accuracy"].append(accuracy_score(y[i], pr))
        vals["f1"].append(f1_score(y[i], pr, zero_division=0))
        vals["macro_f1"].append(f1_score(y[i], pr, average="macro"))
        vals["roc_auc"].append(roc_auc_score(y[i], proba[i]))
    point = clf_metrics(y, proba, thr)
    return pd.DataFrame([{"metric": k, "estimate": point[k], "ci_low": np.percentile(v, 2.5),
                          "ci_high": np.percentile(v, 97.5)} for k, v in vals.items()])


def bootstrap_reg(y, pred, n=300):
    rng = np.random.RandomState(RANDOM_STATE)
    y, pred = np.asarray(y), np.clip(np.asarray(pred), 1, 14)
    vals = {"MAE": [], "RMSE": [], "R2": []}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        vals["MAE"].append(mean_absolute_error(y[i], pred[i]))
        vals["RMSE"].append(float(np.sqrt(mean_squared_error(y[i], pred[i]))))
        vals["R2"].append(r2_score(y[i], pred[i]))
    point = reg_metrics(y, pred)
    return pd.DataFrame([{"metric": k, "estimate": point[k], "ci_low": np.percentile(v, 2.5),
                          "ci_high": np.percentile(v, 97.5)} for k, v in vals.items()])


def subgroup_table(Xte, yte, proba, thr):
    """Fairness check: does the model work equally well for every gender / race / age group?"""
    d = pd.DataFrame({"y": np.asarray(yte), "p": np.asarray(proba)}, index=Xte.index)
    d["pred"] = (d["p"] >= thr).astype(int)
    groups = {"gender": Xte["gender"], "race": Xte["race"],
              "age_group": pd.cut(Xte["age_num"], [0, 40, 60, 80, 100],
                                  labels=["<40", "40-59", "60-79", "80+"])}
    rows = []
    for gname, g in groups.items():
        for val in pd.Series(g).dropna().unique():
            m = (g == val).values
            if m.sum() < 30:
                continue
            sub = d[m]
            rows.append({
                "group": gname, "value": str(val), "n": int(m.sum()),
                "actual_rate": sub["y"].mean(),
                "accuracy": accuracy_score(sub["y"], sub["pred"]),
                "precision": precision_score(sub["y"], sub["pred"], zero_division=0),
                "recall": recall_score(sub["y"], sub["pred"], zero_division=0),
                "roc_auc": roc_auc_score(sub["y"], sub["p"]) if sub["y"].nunique() == 2 else np.nan,
            })
    return pd.DataFrame(rows)


def decile_table(y, proba):
    """Sort patients by predicted risk into 10 groups: is the top group really riskier?"""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    t = d.groupby("decile").agg(patients=("y", "size"), readmit_rate=("y", "mean"),
                                avg_risk_score=("p", "mean")).reset_index()
    t["lift"] = t["readmit_rate"] / d["y"].mean()
    return t


def try_shap(best, Xte, task, fname):
    """Optional SHAP summary plot. Skipped (with a message) if 'shap' is not installed."""
    try:
        import shap

        pre, model = best.named_steps["pre"], best.named_steps["model"]
        sample = Xte.sample(min(300, len(Xte)), random_state=RANDOM_STATE)
        Z = pre.transform(sample)
        names = list(pre.get_feature_names_out())
        try:
            if isinstance(model, (LogisticRegression, Ridge)):
                sv = shap.LinearExplainer(model, Z).shap_values(Z)
            else:
                sv = shap.TreeExplainer(model).shap_values(Z)
        except Exception:
            f = (lambda x: model.predict_proba(x)[:, 1]) if task == "clf" else model.predict
            sv = shap.Explainer(f, shap.sample(Z, 100))(Z).values
        if isinstance(sv, list):
            sv = sv[-1]
        if getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, -1]
        plt.figure()
        shap.summary_plot(sv, Z, feature_names=names, show=False, max_display=15)
        save_fig(fname)
        return True
    except Exception as e:
        print(f"   (SHAP plot skipped: {type(e).__name__}. Optional: pip install shap)")
        return False


def run_classification(Xtr, Xte, ytr, yte, sk):
    banner("CLASSIFICATION: " + ("readmission within 30 days" if TARGET_MODE == "early30"
                                  else "readmission (Yes / No)"))
    skewed = sk[sk["applied"] == "log1p"]["feature"].tolist()
    pre = build_preprocessor(Xtr, skewed)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {"Baseline (always majority)": DummyClassifier(strategy="most_frequent")}
    models.update(get_models("clf", pos_weight))
    fitted, probas, rows = {}, {}, []

    for name, m in models.items():
        t0 = time.time()
        pipe = Pipeline([("pre", clone(pre)), ("model", m)])
        cv_sc = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="roc_auc") \
            if "Baseline" not in name else np.array([0.5])
        cv_auc, cv_sd = cv_sc.mean(), cv_sc.std()
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        r = clf_metrics(yte, p, 0.5)
        r.update({"model": name, "cv_roc_auc": cv_auc, "cv_std": cv_sd})
        rows.append(r)
        fitted[name], probas[name] = pipe, p
        print(f"   {name:<28} acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  "
              f"auc={r['roc_auc']:.3f}  ({time.time() - t0:.0f}s)")

    if TUNE:
        print("   Tuning Hist Gradient Boosting (random search)...")
        t0 = time.time()
        search = tuned_booster("clf", clone(pre))
        search.fit(Xtr, ytr)
        name = "Hist Gradient Boosting (tuned)"
        p = search.best_estimator_.predict_proba(Xte)[:, 1]
        r = clf_metrics(yte, p, 0.5)
        r.update({"model": name, "cv_roc_auc": search.best_score_,
                  "cv_std": search.cv_results_["std_test_score"][search.best_index_]})
        rows.append(r)
        fitted[name], probas[name] = search.best_estimator_, p
        print(f"   {name:<28} acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  "
              f"auc={r['roc_auc']:.3f}  ({time.time() - t0:.0f}s)  best={search.best_params_}")

    res = pd.DataFrame(rows).set_index("model")
    cols = ["accuracy", "balanced_acc", "precision", "recall", "f1", "macro_f1",
            "roc_auc", "pr_auc", "cv_roc_auc", "cv_std"]
    res = res[cols]

    # best model chosen by cross-validation on TRAIN data (test set is never used to choose)
    cand = res.drop(index="Baseline (always majority)")
    best_name = cand["cv_roc_auc"].idxmax()
    best = fitted[best_name]
    print(f"\n   BEST MODEL (by CV ROC-AUC): {best_name}")

    # ---- threshold tuning on out-of-fold train predictions (no test leakage)
    oof = cross_val_predict(clone(best), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    thr, ths, scores = pick_threshold(ytr, oof)
    print(f"   Threshold chosen on train OOF predictions: {thr:.3f} (default was 0.50)")

    p_best = probas[best_name]
    m_default = clf_metrics(yte, p_best, 0.5)
    m_tuned = clf_metrics(yte, p_best, thr)
    gap_d = abs(m_default["precision"] - m_default["recall"])
    gap_t = abs(m_tuned["precision"] - m_tuned["recall"])
    print(f"   Precision/Recall gap  @0.50 : {gap_d:.3f}   |   @{thr:.2f} : {gap_t:.3f}")
    thr_table = pd.DataFrame([{"setting": "default 0.50", "threshold": 0.5, **m_default},
                              {"setting": "tuned", "threshold": thr, **m_tuned}])

    # ---- plots
    plt.figure(figsize=(7, 6))
    for n, p in probas.items():
        if "Baseline" in n:
            continue
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend(fontsize=8)
    save_fig("10_clf_roc_curves.png")

    plt.figure(figsize=(7, 6))
    for n, p in probas.items():
        if "Baseline" in n:
            continue
        pr, rc, _ = precision_recall_curve(yte, p)
        plt.plot(rc, pr, label=f"{n} (AP={average_precision_score(yte, p):.3f})")
    plt.axhline(yte.mean(), color="k", ls="--", label="Random")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=8)
    save_fig("11_clf_precision_recall_curves.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    pr_, rc_ = [], []
    for t in ths:
        pr_.append(precision_score(ytr, (oof >= t).astype(int), zero_division=0))
        rc_.append(recall_score(ytr, (oof >= t).astype(int), zero_division=0))
    ax[0].plot(ths, pr_, label="Precision")
    ax[0].plot(ths, rc_, label="Recall")
    ax[0].plot(ths, scores, label="Macro F1", lw=2)
    ax[0].axvline(thr, color="red", ls="--", label=f"chosen = {thr:.2f}")
    ax[0].set_xlabel("Threshold")
    ax[0].set_title("Threshold tuning (train out-of-fold)")
    ax[0].legend()
    bars = thr_table.set_index("setting")[["accuracy", "precision", "recall", "f1", "macro_f1"]]
    bars.T.plot(kind="bar", ax=ax[1], rot=0)
    ax[1].set_title("Before vs after threshold tuning (test)")
    ax[1].set_ylim(0, 1)
    save_fig("12_clf_threshold_tuning.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, t, title in [(ax[0], 0.5, "Threshold 0.50"), (ax[1], thr, f"Tuned threshold {thr:.2f}")]:
        ConfusionMatrixDisplay.from_predictions(yte, (p_best >= t).astype(int), ax=a,
                                                cmap="Blues", colorbar=False)
        a.set_title(f"{best_name}\n{title}", fontsize=9)
    save_fig("13_clf_confusion_matrices.png")

    plot_df = res.drop(index="Baseline (always majority)")[["accuracy", "f1", "roc_auc"]]
    plot_df.plot(kind="bar", figsize=(10, 4.5), rot=15)
    plt.ylim(0, 1)
    plt.title("Classification model comparison (test set, threshold 0.50)")
    save_fig("14_clf_model_comparison.png")

    # permutation importance on original features
    idx = np.random.RandomState(RANDOM_STATE).choice(len(Xte), min(4000, len(Xte)), replace=False)
    pi = permutation_importance(best, Xte.iloc[idx], yte.iloc[idx], scoring="roc_auc",
                                n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 7))
    sns.barplot(x=imp.head(15).values, y=imp.head(15).index, color="#2c3e50")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.title(f"Top 15 features - {best_name}")
    save_fig("15_clf_feature_importance.png")

    # ---- extra rigor: overfitting check, confidence intervals, calibration, fairness, deciles
    ptr = best.predict_proba(Xtr)[:, 1]
    overfit = pd.DataFrame([
        {"set": "train", **{k: v for k, v in clf_metrics(ytr, ptr, thr).items()
                            if k in ("accuracy", "f1", "roc_auc")}},
        {"set": "test", **{k: v for k, v in clf_metrics(yte, p_best, thr).items()
                           if k in ("accuracy", "f1", "roc_auc")}}])
    gap_auc = overfit.loc[0, "roc_auc"] - overfit.loc[1, "roc_auc"]
    print(f"   Overfitting check: train AUC - test AUC = {gap_auc:.3f} "
          f"({'OK' if gap_auc < 0.10 else 'possible overfitting'})")

    ci = bootstrap_clf(yte, p_best, thr)
    print("   95% confidence intervals (test, bootstrap):")
    for _, r_ in ci.iterrows():
        print(f"      {r_['metric']:<9} {r_['estimate']:.3f}  [{r_['ci_low']:.3f}, {r_['ci_high']:.3f}]")

    brier = brier_score_loss(yte, p_best)
    frac, mean_pred = calibration_curve(yte, p_best, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac, "o-", label=f"{best_name} (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted risk score")
    plt.ylabel("Observed readmission rate")
    plt.title("Calibration curve (test set)")
    plt.legend()
    save_fig("19_clf_calibration.png")

    dec = decile_table(yte, p_best)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=dec["decile"], y=dec["readmit_rate"], color="#e74c3c")
    plt.axhline(yte.mean(), color="k", ls="--", label=f"Overall rate = {yte.mean():.2f}")
    plt.xlabel("Risk decile (10 = highest predicted risk)")
    plt.ylabel("Actual readmission rate")
    plt.title("Do high-risk predictions really have more readmissions?")
    plt.legend()
    save_fig("20_clf_risk_deciles.png")
    print(f"   Top-risk decile readmission rate = {dec['readmit_rate'].iloc[-1]:.3f} "
          f"vs lowest decile {dec['readmit_rate'].iloc[0]:.3f} (lift {dec['lift'].iloc[-1]:.2f}x)")

    sub = subgroup_table(Xte, yte, p_best, thr)
    shap_ok = try_shap(best, Xte, "clf", "21_clf_shap_summary.png")

    return {"overfit": overfit, "ci": ci, "brier": brier, "deciles": dec, "subgroups": sub,
            "shap": shap_ok, "table": res, "best_name": best_name, "best": best, "threshold": thr,
            "thr_table": thr_table, "importance": imp, "imbalance_gap": (gap_d, gap_t)}


# ----------------------------------------------------------------------------
# 6. REGRESSION
# ----------------------------------------------------------------------------
def reg_metrics(y, pred):
    pred = np.clip(pred, 1, 14)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
        "within_1_day_%": float((np.abs(y - pred) <= 1).mean() * 100),
    }


def run_regression(Xtr, Xte, ytr, yte, sk):
    banner("REGRESSION: length of stay (days in hospital)")
    skewed = sk[sk["applied"] == "log1p"]["feature"].tolist()
    pre = build_preprocessor(Xtr, skewed)
    kf = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {"Baseline (predict mean)": DummyRegressor(strategy="mean")}
    models.update(get_models("reg"))
    fitted, preds, rows = {}, {}, []

    for name, m in models.items():
        t0 = time.time()
        pipe = Pipeline([("pre", clone(pre)), ("model", m)])
        cv_sc = cross_val_score(pipe, Xtr, ytr, cv=kf, scoring="r2") \
            if "Baseline" not in name else np.array([0.0])
        cv_r2, cv_sd = cv_sc.mean(), cv_sc.std()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        r = reg_metrics(yte, p)
        r.update({"model": name, "cv_R2": cv_r2, "cv_std": cv_sd})
        rows.append(r)
        fitted[name], preds[name] = pipe, np.clip(p, 1, 14)
        print(f"   {name:<28} MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r['R2']:.3f}  "
              f"({time.time() - t0:.0f}s)")

    if TUNE:
        print("   Tuning Hist Gradient Boosting (random search)...")
        t0 = time.time()
        search = tuned_booster("reg", clone(pre))
        search.fit(Xtr, ytr)
        name = "Hist Gradient Boosting (tuned)"
        p = search.best_estimator_.predict(Xte)
        r = reg_metrics(yte, p)
        r.update({"model": name, "cv_R2": search.best_score_,
                  "cv_std": search.cv_results_["std_test_score"][search.best_index_]})
        rows.append(r)
        fitted[name], preds[name] = search.best_estimator_, np.clip(p, 1, 14)
        print(f"   {name:<28} MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r['R2']:.3f}  "
              f"({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows).set_index("model")[["MAE", "RMSE", "R2", "within_1_day_%", "cv_R2", "cv_std"]]
    cand = res.drop(index="Baseline (predict mean)")
    best_name = cand["cv_R2"].idxmax()
    best = fitted[best_name]
    p_best = preds[best_name]
    print(f"\n   BEST MODEL (by CV R2): {best_name}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    jitter = np.random.RandomState(0).normal(0, 0.12, len(yte))
    ax[0].scatter(yte + jitter, p_best, alpha=0.15, s=8)
    ax[0].plot([1, 14], [1, 14], "r--")
    ax[0].set_xlabel("Actual days")
    ax[0].set_ylabel("Predicted days")
    ax[0].set_title(f"Actual vs predicted\n{best_name}")
    resid = yte.values - p_best
    ax[1].scatter(p_best, resid, alpha=0.15, s=8)
    ax[1].axhline(0, color="r", ls="--")
    ax[1].set_xlabel("Predicted days")
    ax[1].set_ylabel("Residual")
    ax[1].set_title("Residual plot")
    sns.histplot(resid, bins=40, kde=True, ax=ax[2], color="#8e44ad")
    ax[2].set_title(f"Residual distribution (skew={pd.Series(resid).skew():.2f})")
    save_fig("16_reg_diagnostics.png")

    plot_df = res.drop(index="Baseline (predict mean)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c in zip(ax, ["MAE", "RMSE", "R2"]):
        sns.barplot(x=plot_df.index, y=plot_df[c], ax=a, palette="crest")
        a.set_title(c)
        a.set_xlabel("")
        a.tick_params(axis="x", rotation=25)
    save_fig("17_reg_model_comparison.png")

    idx = np.random.RandomState(RANDOM_STATE).choice(len(Xte), min(4000, len(Xte)), replace=False)
    pi = permutation_importance(best, Xte.iloc[idx], yte.iloc[idx], scoring="r2",
                                n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 7))
    sns.barplot(x=imp.head(15).values, y=imp.head(15).index, color="#16a085")
    plt.xlabel("Drop in R2 when feature is shuffled")
    plt.title(f"Top 15 features for length of stay - {best_name}")
    save_fig("18_reg_feature_importance.png")

    ptr = np.clip(best.predict(Xtr), 1, 14)
    overfit = pd.DataFrame([
        {"set": "train", "MAE": reg_metrics(ytr, ptr)["MAE"], "R2": reg_metrics(ytr, ptr)["R2"]},
        {"set": "test", "MAE": reg_metrics(yte, p_best)["MAE"], "R2": reg_metrics(yte, p_best)["R2"]}])
    gap_r2 = overfit.loc[0, "R2"] - overfit.loc[1, "R2"]
    print(f"   Overfitting check: train R2 - test R2 = {gap_r2:.3f} "
          f"({'OK' if gap_r2 < 0.10 else 'possible overfitting'})")
    ci = bootstrap_reg(yte, p_best)
    for _, r_ in ci.iterrows():
        print(f"      {r_['metric']:<5} {r_['estimate']:.3f}  [{r_['ci_low']:.3f}, {r_['ci_high']:.3f}]")
    shap_ok = try_shap(best, Xte, "reg", "22_reg_shap_summary.png")

    return {"overfit": overfit, "ci": ci, "shap": shap_ok, "table": res, "best_name": best_name, "best": best, "importance": imp}


# ----------------------------------------------------------------------------
# 7. REPORT
# ----------------------------------------------------------------------------
def write_report(clean_log, clf, reg, sk, imbalance_ratio, n_train, n_test, corr_pairs):
    RESULTS_DIR.mkdir(exist_ok=True)
    clf["table"].round(4).to_csv(RESULTS_DIR / "metrics_classification.csv")
    reg["table"].round(4).to_csv(RESULTS_DIR / "metrics_regression.csv")
    clf["thr_table"].round(4).to_csv(RESULTS_DIR / "threshold_tuning.csv", index=False)
    clf["subgroups"].round(4).to_csv(RESULTS_DIR / "subgroup_fairness.csv", index=False)
    clf["deciles"].round(4).to_csv(RESULTS_DIR / "risk_deciles.csv", index=False)
    clf["ci"].round(4).to_csv(RESULTS_DIR / "confidence_intervals_classification.csv", index=False)
    reg["ci"].round(4).to_csv(RESULTS_DIR / "confidence_intervals_regression.csv", index=False)
    sk.round(4).to_csv(RESULTS_DIR / "skewness_kurtosis.csv", index=False)

    b = clf["table"].loc[clf["best_name"]]
    t = clf["thr_table"].iloc[1]
    r = reg["table"].loc[reg["best_name"]]
    target_txt = "readmission within 30 days" if TARGET_MODE == "early30" else "any readmission (Yes/No)"

    L = []
    L.append("# Results\n")
    L.append(f"*Auto-generated by `clinical_project.py`.* Train rows: **{n_train:,}**, test rows: **{n_test:,}**.\n")
    L.append("## 1. Data cleaning steps\n")
    L += [f"- {x}" for x in clean_log]
    L.append("\n![missing](figures/01_missing_values.png)\n")
    L.append("## 2. Class balance / imbalance check\n")
    L.append(f"Classification target: **{target_txt}**. Imbalance ratio = **{imbalance_ratio:.2f} : 1**. "
             "Handled with `class_weight='balanced'` and threshold tuning (below).\n")
    L.append("![target](figures/02_target_distribution.png)\n")
    L.append("## 3. Skewness and kurtosis\n")
    L.append("Features with |skew| > 1 get a `log1p` transform inside the model pipeline.\n")
    L.append(md_table(sk.round(3)) + "\n")
    L.append("![dist](figures/03_distributions_skew_kurtosis_before.png)\n")
    L.append("![skewfix](figures/04_skew_fix_log_transform.png)\n")
    L.append("![box](figures/05_boxplots_outliers.png)\n")
    L.append("## 4. Correlation\n")
    L.append("![corr](figures/06_correlation_heatmap.png)\n")
    L.append("![corr2](figures/07_correlation_with_targets.png)\n")
    if corr_pairs:
        L.append("Highly correlated feature pairs (|r| >= 0.6) - tree models handle this well; "
                 "Logistic/Ridge are regularised:\n")
        L += [f"- `{a}` vs `{b}`: r = {r_:.2f}" for a, b, r_ in corr_pairs]
        L.append("")
    else:
        L.append("No feature pairs with |r| >= 0.6 (no serious multicollinearity).\n")
    L.append("![rates](figures/08_readmission_rate_by_feature.png)\n")
    L.append("![los](figures/09_length_of_stay.png)\n")
    L.append("## 5. Classification results (test set, threshold 0.50)\n")
    L.append(md_table(clf["table"].reset_index()) + "\n")
    L.append(f"**Best model:** {clf['best_name']}\n")
    L.append("### Threshold tuning (precision / recall balance)\n")
    L.append(md_table(clf["thr_table"]) + "\n")
    L.append(f"Precision-recall gap went from **{clf['imbalance_gap'][0]:.3f}** (threshold 0.50) "
             f"to **{clf['imbalance_gap'][1]:.3f}** (tuned threshold {clf['threshold']:.2f}).\n")
    for f in ["10_clf_roc_curves", "11_clf_precision_recall_curves", "12_clf_threshold_tuning",
              "13_clf_confusion_matrices", "14_clf_model_comparison", "15_clf_feature_importance"]:
        L.append(f"![{f}](figures/{f}.png)\n")
    L.append("### Overfitting check (train vs test)\n")
    L.append(md_table(clf["overfit"]) + "\n")
    L.append("### 95% confidence intervals (bootstrap, test set)\n")
    L.append(md_table(clf["ci"]) + "\n")
    L.append(f"### Calibration and risk deciles\nBrier score: **{clf['brier']:.4f}**. Class weights shift "
             "predicted values upwards, so the model output is a *risk score* (good for ranking patients), "
             "not an exact probability.\n")
    L.append("![calib](figures/19_clf_calibration.png)\n")
    L.append(md_table(clf["deciles"]) + "\n")
    L.append("![deciles](figures/20_clf_risk_deciles.png)\n")
    L.append("### Fairness check by subgroup\n")
    L.append("Performance should be similar across groups; big gaps would mean the tool is unfair.\n")
    L.append(md_table(clf["subgroups"]) + "\n")
    if clf["shap"]:
        L.append("### SHAP\n![shap](figures/21_clf_shap_summary.png)\n")
    L.append("## 6. Regression results (length of stay, test set)\n")
    L.append(md_table(reg["table"].reset_index()) + "\n")
    L.append(f"**Best model:** {reg['best_name']}\n")
    for f in ["16_reg_diagnostics", "17_reg_model_comparison", "18_reg_feature_importance"]:
        L.append(f"![{f}](figures/{f}.png)\n")
    L.append("### Overfitting check and 95% confidence intervals\n")
    L.append(md_table(reg["overfit"]) + "\n")
    L.append(md_table(reg["ci"]) + "\n")
    if reg["shap"]:
        L.append("![shap](figures/22_reg_shap_summary.png)\n")
    L.append("## 7. Summary\n")
    L.append(f"- Classification ({target_txt}): best = **{clf['best_name']}**; "
             f"accuracy **{t['accuracy']:.3f}**, F1 **{t['f1']:.3f}**, macro-F1 **{t['macro_f1']:.3f}**, "
             f"ROC-AUC **{b['roc_auc']:.3f}** (tuned threshold {clf['threshold']:.2f}).")
    L.append(f"- Regression (length of stay): best = **{reg['best_name']}**; "
             f"MAE **{r['MAE']:.2f} days**, RMSE **{r['RMSE']:.2f}**, R2 **{r['R2']:.3f}**, "
             f"within 1 day: **{r['within_1_day_%']:.1f}%**.")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n   [report saved] results/RESULTS.md")


def save_models(clf, reg, Xclf, Xreg):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf["best"], MODEL_DIR / "classifier.joblib", compress=3)
    joblib.dump(reg["best"], MODEL_DIR / "regressor.joblib", compress=3)
    defaults, options = {}, {}
    for c in Xclf.columns:
        if pd.api.types.is_numeric_dtype(Xclf[c]):
            defaults[c] = float(Xclf[c].median())
        else:
            defaults[c] = str(Xclf[c].mode()[0])
            options[c] = sorted(Xclf[c].astype(str).unique().tolist())
    meta = {"target_mode": TARGET_MODE, "threshold": clf["threshold"],
            "clf_model": clf["best_name"], "reg_model": reg["best_name"],
            "clf_features": Xclf.columns.tolist(), "reg_features": Xreg.columns.tolist(),
            "defaults": defaults, "options": options}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("   [models saved] models/classifier.joblib, regressor.joblib, meta.json")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    for d in (DATA_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    banner("STEP 1: LOAD DATA")
    raw = load_data()
    print(f"   Shape: {raw.shape}")
    eda_missing(raw)

    banner("STEP 2: CLEANING + FEATURE ENGINEERING")
    df, clean_log = clean_data(raw)
    for line in clean_log:
        print("   -", line)

    banner("STEP 3: EDA (distributions, imbalance, skewness, kurtosis, correlation)")
    imbalance_ratio = eda_target(df)
    if imbalance_ratio > 1.5:
        print("   ! Imbalance detected -> class weights + threshold tuning will be used.")
    else:
        print("   Classes are roughly balanced (class weights + threshold tuning used anyway, to be safe).")

    X_all = df.drop(columns=["readmitted", "target_clf"])
    sk = skew_kurt_table(X_all)
    print("\n   Skewness / kurtosis:")
    print(sk.round(2).to_string(index=False))
    eda_skew_plots(df, sk)
    eda_boxplots(df, [c for c in sk["feature"] if c not in ("age_num",)][:8])
    corr_pairs = eda_correlation(df)
    eda_rates(df)

    banner("STEP 4: TRAIN / TEST SPLIT")
    tr_idx, te_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                      stratify=df["target_clf"])
    print(f"   Train: {len(tr_idx):,}   Test: {len(te_idx):,}  (same split for both tasks)")

    # classification features (length of stay IS allowed as a feature here)
    Xc = df.drop(columns=["readmitted", "target_clf"])
    clf = run_classification(Xc.loc[tr_idx], Xc.loc[te_idx], df.loc[tr_idx, "target_clf"],
                             df.loc[te_idx, "target_clf"], sk)

    # regression features (target time_in_hospital removed, otherwise the model would cheat)
    Xr = df.drop(columns=["readmitted", "target_clf", "time_in_hospital"])
    sk_r = sk[sk.feature != "time_in_hospital"]
    reg = run_regression(Xr.loc[tr_idx], Xr.loc[te_idx], df.loc[tr_idx, "time_in_hospital"],
                         df.loc[te_idx, "time_in_hospital"], sk_r)

    banner("STEP 5: SAVE REPORT + MODELS")
    write_report(clean_log, clf, reg, sk, imbalance_ratio, len(tr_idx), len(te_idx), corr_pairs)
    save_models(clf, reg, Xc, Xr)

    banner("FINAL SUMMARY")
    t = clf["thr_table"].iloc[1]
    r = reg["table"].loc[reg["best_name"]]
    print(f"   Classification best : {clf['best_name']}")
    print(f"      accuracy={t['accuracy']:.3f}  precision={t['precision']:.3f}  recall={t['recall']:.3f}  "
          f"F1={t['f1']:.3f}  macroF1={t['macro_f1']:.3f}  (threshold {clf['threshold']:.2f})")
    print(f"   Regression best     : {reg['best_name']}")
    print(f"      MAE={r['MAE']:.2f} days  RMSE={r['RMSE']:.2f}  R2={r['R2']:.3f}")
    print(f"\n   Done in {(time.time() - t_start) / 60:.1f} minutes. Open results/RESULTS.md")


if __name__ == "__main__":
    main()