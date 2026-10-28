"""
attrition_core.py
=================
Core logic for the Employee Attrition project (CatBoost).
Used by BOTH the Jupyter notebook and the Streamlit app, so the two always agree.

Contents (mapped to the grading rubric)
---------------------------------------
1  Dataset understanding   -> load_data, guess_target, encode_target, describe_dataset
2  Data preprocessing      -> Preprocessor (fit on train only -> no data leakage)
3  Manual implementation   -> ManualGradientBoosting (from scratch), manual_metrics, manual_roc_auc
4  Modelling               -> CatBoost (train_pipeline), baselines in compare_models
5  Parameter tuning        -> random_search_catboost, tune_knn_k
6  Evaluation              -> manual_metrics, evaluate, best_f1_threshold
7  Comparison              -> compare_models
8  Visualisation           -> plot_* functions
9  Interpretation          -> catboost_shap, auto_insights
"""
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_OK = True
except ImportError:  # the notebook / app show a clear message in this case
    CatBoostClassifier, Pool, CATBOOST_OK = None, None, False


def require_catboost():
    if not CATBOOST_OK:
        raise ImportError("CatBoost is not installed. Run:  pip install catboost")


# =====================================================================
# 1. DATASET LOADING & UNDERSTANDING
# =====================================================================
def load_data(source):
    """Load a CSV / Excel file from a path or an uploaded file object."""
    name = getattr(source, "name", str(source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except Exception as e:  # noqa
            last_err = e
    raise ValueError(f"Could not read file: {last_err}")


_TARGET_HINTS = ["attrition", "left", "churn", "turnover", "resign", "exit", "quit"]
_POS = {"yes", "y", "1", "true", "t", "left", "attrited", "resigned", "quit", "exited", "leaver"}
_NEG = {"no", "n", "0", "false", "f", "stayed", "retained", "active", "current"}


def guess_target(df):
    """Pick the most likely attrition column (user can override)."""
    for c in df.columns:
        if any(h in c.lower() for h in _TARGET_HINTS):
            return c
    for c in df.columns[::-1]:
        if df[c].nunique(dropna=True) == 2:
            return c
    return df.columns[-1]


def encode_target(s):
    """Convert Yes/No, True/False, 1/0, Left/Stayed ... into 1 (left) / 0 (stayed)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = sorted(s.dropna().unique())
        if len(vals) != 2:
            raise ValueError(f"Target must be binary; found {len(vals)} distinct values.")
        return (s == vals[-1]).astype(float).where(s.notna())
    low = s.astype("object").where(s.notna(), None).map(lambda v: None if v is None else str(v).strip().lower())
    out = low.map(lambda v: np.nan if v is None else (1.0 if v in _POS else (0.0 if v in _NEG else -1.0)))
    bad = sorted(set(low[out == -1.0].dropna()))
    if bad:
        raise ValueError(f"Unrecognised target values: {bad[:5]}. Use Yes/No, 1/0 or True/False.")
    return out


def describe_dataset(df, target):
    """Feature table: dtype, #unique, %missing, example value."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "feature": c,
            "role": "TARGET" if c == target else "feature",
            "dtype": str(s.dtype),
            "type": "numeric" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else "categorical",
            "unique": s.nunique(dropna=True),
            "missing_%": round(100 * s.isna().mean(), 2),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 2. PREPROCESSING  (fit on TRAIN only, then reuse on test / new uploads)
# =====================================================================
class Preprocessor:
    """
    Steps (each one is logged in `log_` so it can be explained in the report):
      1. Drop useless columns  : constant, ID-like, >60 % missing
      2. Missing values        : numeric -> median, categorical -> 'Missing'
      3. Outlier treatment     : IQR capping (winsorising) on continuous columns
      4. Categorical handling  : kept as text for CatBoost (native encoding);
                                 one-hot encoded for the other models
      5. Scaling               : z-score for the models that need it (KNN, LogReg, manual GBM)
    """

    def __init__(self, cap_outliers=True, iqr_k=1.5, max_missing=0.6, id_ratio=0.95):
        self.cap_outliers, self.iqr_k = cap_outliers, iqr_k
        self.max_missing, self.id_ratio = max_missing, id_ratio

    @staticmethod
    def _is_num(s):
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

    def fit(self, X):
        n = len(X)
        self.log_, self.dropped_ = [], {}
        for c in X.columns:
            s = X[c]
            nun = s.nunique(dropna=True)
            int_like = (not self._is_num(s)) or bool((s.dropna() % 1 == 0).all())
            name_hint = bool(re.search(r"(^|_)id$|number$|^id_", c.lower()))
            if s.isna().mean() > self.max_missing:
                self.dropped_[c] = "more than 60% missing"
            elif nun <= 1:
                self.dropped_[c] = "constant (no information)"
            elif n > 20 and int_like and (nun / n > self.id_ratio or (name_hint and nun / n > 0.5)):
                self.dropped_[c] = "ID-like (almost every value unique)"
        keep = [c for c in X.columns if c not in self.dropped_]
        self.feature_order_ = keep
        self.num_cols_ = [c for c in keep if self._is_num(X[c])]
        self.cat_cols_ = [c for c in keep if c not in self.num_cols_]
        if self.dropped_:
            self.log_.append("Dropped columns: " + "; ".join(f"{k} ({v})" for k, v in self.dropped_.items()))

        # missing values
        self.medians_ = X[self.num_cols_].median()
        miss = X[keep].isna().sum()
        miss = miss[miss > 0]
        self.log_.append("Missing values -> numeric: median, categorical: 'Missing'. Affected: "
                         + (", ".join(f"{k} ({v})" for k, v in miss.items()) if len(miss) else "none"))

        # outliers (only continuous-like columns: > 10 distinct values)
        self.bounds_ = {}
        outl = {}
        for c in self.num_cols_:
            s = X[c].fillna(self.medians_[c])
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if self.cap_outliers and iqr > 0 and s.nunique() > 10:
                lo, hi = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
                self.bounds_[c] = (lo, hi)
                k = int(((s < lo) | (s > hi)).sum())
                if k:
                    outl[c] = k
        self.log_.append("Outliers capped with the IQR rule (1.5xIQR): "
                         + (", ".join(f"{k} ({v})" for k, v in outl.items()) if outl else "none found"))

        # categorical levels (for one-hot)
        self.levels_ = {}
        for c in self.cat_cols_:
            s = X[c].astype("object").where(X[c].notna(), "Missing").astype(str)
            self.levels_[c] = list(s.value_counts().index[:30])
        self.log_.append(f"Categorical columns ({len(self.cat_cols_)}): {self.cat_cols_}")
        self.log_.append(f"Numeric columns ({len(self.num_cols_)}): {self.num_cols_}")

        # scaling stats
        Z = self._clean(X)
        self.mean_ = Z[self.num_cols_].mean()
        self.std_ = Z[self.num_cols_].std().replace(0, 1).fillna(1)
        return self

    def _clean(self, X):
        out = {}
        for c in self.feature_order_:
            if c in self.num_cols_:
                s = pd.to_numeric(X[c], errors="coerce") if c in X else pd.Series(np.nan, index=X.index)
                s = s.fillna(self.medians_[c])
                if c in self.bounds_:
                    s = s.clip(*self.bounds_[c])
                out[c] = s
            else:
                if c in X:
                    s = X[c].astype("object").where(X[c].notna(), "Missing").astype(str)
                else:
                    s = pd.Series("Missing", index=X.index)
                out[c] = s
        return pd.DataFrame(out, index=X.index)

    def transform_cb(self, X):
        """Clean (not scaled) frame with categoricals kept as text -> for CatBoost."""
        return self._clean(X)

    def transform_num(self, X):
        """Fully numeric, scaled, one-hot matrix -> for KNN / LogReg / RF / manual GBM."""
        Z = self._clean(X)
        parts = [((Z[self.num_cols_] - self.mean_) / self.std_)]
        for c in self.cat_cols_:
            parts.append(pd.DataFrame({f"{c}={lvl}": (Z[c] == lvl).astype(float) for lvl in self.levels_[c]},
                                      index=Z.index))
        return pd.concat(parts, axis=1)


# =====================================================================
# 3. ALGORITHM IMPLEMENTED FROM SCRATCH
# =====================================================================
class ManualGradientBoosting:
    """
    Gradient boosting for binary classification, written from scratch with NumPy.
    This is the same idea CatBoost uses (add small trees one after another, each one
    correcting the mistakes of the ensemble so far) in its simplest form.

    Weak learner : decision stump (a tree with ONE split).
    Loss         : logistic loss (log-loss).
    Per round    : 1. p = sigmoid(F)                       current probabilities
                   2. g = p - y ,  h = p(1 - p)            gradient & hessian of log-loss
                   3. for every feature/threshold:
                          gain = GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam)
                   4. pick best split; leaf value = -G / (H + lam)    (Newton step)
                   5. F += learning_rate * leaf value
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_bins=16, reg_lambda=1.0):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_bins, self.reg_lambda = max_bins, reg_lambda

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        lam = self.reg_lambda
        qs = np.linspace(0, 1, self.max_bins + 2)[1:-1]
        self.thresholds_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(d)]
        bins = np.column_stack([np.digitize(X[:, j], self.thresholds_[j], right=True) for j in range(d)])

        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_ = np.log(p0 / (1 - p0))
        F = np.full(n, self.base_)
        self.stumps_, self.loss_history_ = [], []
        self.gain_importance_ = np.zeros(d)

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            g, h = p - y, p * (1 - p)
            G, H = g.sum(), h.sum()
            parent = G * G / (H + lam)
            best = (0.0, None, None)
            for j in range(d):
                k = len(self.thresholds_[j])
                if k == 0:
                    continue
                gb = np.bincount(bins[:, j], weights=g, minlength=k + 1)
                hb = np.bincount(bins[:, j], weights=h, minlength=k + 1)
                GL, HL = np.cumsum(gb)[:k], np.cumsum(hb)[:k]      # left = x <= threshold
                GR, HR = G - GL, H - HL
                gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - parent
                gain[(HL < 1e-6) | (HR < 1e-6)] = -np.inf
                b = int(np.argmax(gain))
                if gain[b] > best[0]:
                    best = (gain[b], j, b)
            gain, j, b = best
            if j is None:
                break
            t = self.thresholds_[j][b]
            left = X[:, j] <= t
            GL, HL = g[left].sum(), h[left].sum()
            lv = -GL / (HL + lam)
            rv = -(G - GL) / (H - HL + lam)
            self.stumps_.append((j, t, lv, rv))
            self.gain_importance_[j] += gain
            F += self.learning_rate * np.where(left, lv, rv)
            p = self._sigmoid(F)
            self.loss_history_.append(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.base_)
        for j, t, lv, rv in self.stumps_:
            F += self.learning_rate * np.where(X[:, j] <= t, lv, rv)
        return F

    def staged_decision(self, X, rounds):
        """Raw scores after each boosting round listed in `rounds` (1-based). Used for the animation."""
        X = np.asarray(X, dtype=float)
        F, want, out = np.full(len(X), self.base_), set(rounds), {}
        for r, (j, t, lv, rv) in enumerate(self.stumps_, 1):
            F = F + self.learning_rate * np.where(X[:, j] <= t, lv, rv)
            if r in want:
                out[r] = F.copy()
        return out

    def predict_proba(self, X):
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------- manual evaluation metrics ----------
def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def manual_metrics(y_true, y_pred):
    """Accuracy / precision / recall / F1 computed by hand from the confusion matrix."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def manual_roc_auc(y_true, score):
    """ROC-AUC via the rank (Mann-Whitney U) formula: P(score of a leaver > score of a stayer)."""
    y_true = np.asarray(y_true).astype(int)
    ranks = pd.Series(np.asarray(score)).rank(method="average").values
    n1, n0 = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(y_true, proba, threshold=0.5):
    m = manual_metrics(y_true, (np.asarray(proba) >= threshold).astype(int))
    m["ROC-AUC"] = manual_roc_auc(y_true, proba)
    return m


def best_f1_threshold(y_true, proba, grid=np.arange(0.05, 0.96, 0.01)):
    scores = [manual_metrics(y_true, (proba >= t).astype(int))["F1"] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


# =====================================================================
# 4/5. MODELLING + PARAMETER TUNING
# =====================================================================
DEFAULT_CB_PARAMS = dict(iterations=200, depth=6, learning_rate=0.08, l2_leaf_reg=3,
                         auto_class_weights="Balanced")

CB_SPACE = dict(
    iterations=[150, 200, 300],
    depth=[3, 4, 5, 6],
    learning_rate=[0.05, 0.08, 0.1, 0.15],
    l2_leaf_reg=[1, 3, 5, 9],
    auto_class_weights=["Balanced", None],
)


def make_catboost(params, cat_cols, seed=42):
    require_catboost()
    return CatBoostClassifier(**params, cat_features=list(cat_cols), random_seed=seed,
                              verbose=0, allow_writing_files=False)


def random_search_catboost(X, y, cat_cols, n_iter=8, cv=3, seed=42, space=None, progress=None):
    """Random search + stratified k-fold CV. Selection metric: ROC-AUC (robust to class imbalance)."""
    require_catboost()
    space = space or CB_SPACE
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    y = np.asarray(y)
    rows = []
    for i in range(n_iter):
        params = {k: v[rng.randint(len(v))] for k, v in space.items()}
        aucs, f1s = [], []
        for tr, va in skf.split(X, y):
            m = make_catboost(params, cat_cols, seed)
            m.fit(X.iloc[tr], y[tr])
            pr = m.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y[va], pr))
            f1s.append(manual_metrics(y[va], (pr >= 0.5).astype(int))["F1"])
        rows.append({"trial": i + 1, **params, "cv_auc": np.mean(aucs), "cv_f1": np.mean(f1s)})
        if progress:
            progress((i + 1) / n_iter)
    res = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = {k: res.loc[0, k] for k in space}
    best = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
            for k, v in best.items()}
    return best, res


def tune_knn_k(X, y, ks=(3, 5, 7, 11, 15, 21, 31), cv=5, seed=42):
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    y = np.asarray(y)
    rows = []
    for k in ks:
        aucs = []
        for tr, va in skf.split(X, y):
            m = KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        rows.append({"k": k, "cv_auc": np.mean(aucs)})
    df = pd.DataFrame(rows)
    return int(df.loc[df.cv_auc.idxmax(), "k"]), df


# =====================================================================
# 7. COMPARISON
# =====================================================================
def compare_models(Xtr_num, ytr, Xte_num, yte, cb_proba, seed=42, knn_k=15):
    """Train baselines (same split) and compare with the already-evaluated CatBoost probabilities."""
    Xtr_num, Xte_num = np.asarray(Xtr_num, float), np.asarray(Xte_num, float)
    probas = {}
    probas["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        Xtr_num, ytr).predict_proba(Xte_num)[:, 1]
    probas[f"KNN (k={knn_k})"] = KNeighborsClassifier(n_neighbors=knn_k).fit(
        Xtr_num, ytr).predict_proba(Xte_num)[:, 1]
    probas["Random Forest"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=3, random_state=seed, n_jobs=-1
    ).fit(Xtr_num, ytr).predict_proba(Xte_num)[:, 1]
    gbm = ManualGradientBoosting(n_estimators=150, learning_rate=0.1).fit(Xtr_num, ytr)
    probas["Manual GBM (scratch)"] = gbm.predict_proba(Xte_num)[:, 1]
    probas["CatBoost (tuned)"] = np.asarray(cb_proba)
    table = pd.DataFrame({k: evaluate(yte, p) for k, p in probas.items()}).T
    table = table[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].astype(float)
    return table.sort_values("ROC-AUC", ascending=False), probas, gbm


# =====================================================================
# 9. INTERPRETATION  (SHAP from CatBoost, no extra library needed)
# =====================================================================
def catboost_shap(model, X, cat_cols):
    pool = Pool(X, cat_features=list(cat_cols))
    sv = model.get_feature_importance(pool, type="ShapValues")
    return pd.DataFrame(sv[:, :-1], columns=X.columns, index=X.index), float(sv[0, -1])


def importance_table(model, columns):
    imp = pd.DataFrame({"feature": list(columns), "importance": model.get_feature_importance()})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def auto_insights(raw, y, imp, top_n=6):
    """Plain-English, data-driven findings for the top drivers (works for any uploaded dataset)."""
    y = pd.Series(np.asarray(y), index=raw.index)
    overall = y.mean()
    out = [f"Overall attrition rate in this data is {overall:.1%}."]
    for f in imp["feature"].head(top_n):
        if f not in raw.columns:
            continue
        s = raw[f]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            a, b = s[y == 1].mean(), s[y == 0].mean()
            direction = "lower" if a < b else "higher"
            out.append(f"{f}: employees who left have a {direction} average ({a:,.1f}) than those who stayed ({b:,.1f}).")
        else:
            g = y.groupby(s.astype(str)).agg(["mean", "count"])
            g = g[g["count"] >= max(10, int(0.01 * len(y)))]
            if len(g) >= 2:
                hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
                out.append(f"{f}: attrition is {g.loc[hi, 'mean']:.1%} for '{hi}' versus {g.loc[lo, 'mean']:.1%} for '{lo}'.")
    return out


# =====================================================================
# 8. VISUALISATION
# =====================================================================
PALETTE = {"stay": "#008F87", "leave": "#FF686B", "accent": "#292929"}


def _fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_class_balance(y):
    fig, ax = _fig(4.5, 3.5)
    c = pd.Series(np.asarray(y)).value_counts().sort_index()
    bars = ax.bar(["Stayed", "Left"], [c.get(0, 0), c.get(1, 0)], color=[PALETTE["stay"], PALETTE["leave"]])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}\n({b.get_height()/c.sum():.0%})",
                ha="center", va="bottom")
    ax.set_title("Class balance (target)"); ax.set_ylim(0, c.max() * 1.25)
    return fig


def plot_missing(df):
    m = df.isna().mean().mul(100)
    m = m[m > 0].sort_values()
    fig, ax = _fig(6, max(2.5, 0.35 * max(len(m), 1) + 1))
    if len(m):
        ax.barh(m.index, m.values, color=PALETTE["leave"]); ax.set_xlabel("% missing")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center"); ax.axis("off")
    ax.set_title("Missing values per column")
    return fig


def plot_corr(Z_num, y, top=15):
    d = Z_num.copy(); d["__target__"] = np.asarray(y)
    corr = d.corr()["__target__"].drop("__target__").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = _fig(6.5, 5)
    ax.barh(corr.index, corr.values, color=[PALETTE["leave"] if v > 0 else PALETTE["stay"] for v in corr.values])
    ax.axvline(0, color="k", lw=0.8); ax.set_title("Correlation of numeric features with attrition")
    return fig


def plot_rate_by_category(raw, y, col, max_levels=12):
    s = raw[col].astype(str)
    g = pd.Series(np.asarray(y), index=raw.index).groupby(s).agg(["mean", "count"])
    g = g.sort_values("mean").tail(max_levels)
    fig, ax = _fig(6.5, max(2.5, 0.4 * len(g) + 1))
    ax.barh(g.index, g["mean"] * 100, color=PALETTE["leave"])
    ax.axvline(np.mean(y) * 100, color="k", ls="--", lw=1, label="overall")
    ax.set_xlabel("Attrition rate (%)"); ax.set_title(f"Attrition rate by {col}"); ax.legend()
    return fig


def plot_numeric_dist(raw, y, col):
    fig, ax = _fig(6, 3.8)
    yy = np.asarray(y)
    s = pd.to_numeric(raw[col], errors="coerce")
    ax.hist(s[yy == 0].dropna(), bins=25, alpha=0.6, color=PALETTE["stay"], label="Stayed", density=True)
    ax.hist(s[yy == 1].dropna(), bins=25, alpha=0.6, color=PALETTE["leave"], label="Left", density=True)
    ax.set_title(f"{col}: leavers vs stayers"); ax.legend()
    return fig


def plot_confusion(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = _fig(4.2, 3.8)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred: Stay", "Pred: Leave"]); ax.set_yticks([0, 1], ["Actual: Stay", "Actual: Leave"])
    ax.set_title(title); ax.spines[:].set_visible(False)
    return fig


def plot_roc(y_true, probas):
    fig, ax = _fig(5.5, 4.5)
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=2.5 if "CatBoost" in name else 1.3, label=f"{name} (AUC {manual_roc_auc(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_model_comparison(table):
    fig, ax = _fig(7.5, 4)
    table[["Recall", "F1", "ROC-AUC"]].plot.bar(ax=ax, rot=20, width=0.8, color=["#FF686B", "#008F87", "#292929"])
    ax.set_ylim(0, 1); ax.set_title("Model comparison (test set)"); ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def plot_threshold_curve(y_true, proba):
    grid = np.arange(0.05, 0.96, 0.01)
    ms = [manual_metrics(y_true, (proba >= t).astype(int)) for t in grid]
    fig, ax = _fig(6, 4)
    for k in ("Precision", "Recall", "F1"):
        ax.plot(grid, [m[k] for m in ms], label=k)
    ax.set_xlabel("Decision threshold"); ax.set_title("Precision / recall trade-off"); ax.legend()
    return fig


def plot_importance(imp, top=15):
    d = imp.head(top)[::-1]
    fig, ax = _fig(6.5, 0.32 * len(d) + 1.5)
    ax.barh(d["feature"], d["importance"], color=PALETTE["accent"]); ax.set_title("CatBoost feature importance")
    return fig


def plot_shap_summary(shap_df, X, top=10, seed=0):
    rng = np.random.RandomState(seed)
    order = shap_df.abs().mean().sort_values(ascending=False).index[:top][::-1]
    fig, ax = _fig(7, 0.45 * len(order) + 1.5)
    for i, c in enumerate(order):
        v = X[c]
        if pd.api.types.is_numeric_dtype(v):
            r = v.rank(pct=True).values; sc = ax.scatter(shap_df[c], i + rng.uniform(-.25, .25, len(v)),
                                                         c=r, cmap="coolwarm", s=8, alpha=.7)
        else:
            ax.scatter(shap_df[c], i + rng.uniform(-.25, .25, len(v)), c="grey", s=8, alpha=.5)
    ax.set_yticks(range(len(order)), order); ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("SHAP value (impact on attrition risk)  |  red = high feature value, blue = low, grey = categorical")
    ax.xaxis.label.set_size(8); ax.set_title("SHAP summary")
    return fig


def plot_shap_waterfall(shap_row, X_row, top=8):
    s = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = _fig(6.5, 0.4 * len(s) + 1.5)
    ax.barh([f"{k} = {X_row[k]}" for k in s.index], s.values,
            color=[PALETTE["leave"] if v > 0 else PALETTE["stay"] for v in s.values])
    ax.axvline(0, color="k", lw=.8); ax.set_title("Why this prediction? (red pushes toward leaving)")
    return fig


def plot_loss_curve(gbm):
    fig, ax = _fig(5.5, 3.8)
    ax.plot(gbm.loss_history_, color=PALETTE["stay"]); ax.set_xlabel("Boosting round"); ax.set_ylabel("Training log-loss")
    ax.set_title("Manual GBM: loss decreases as stumps are added")
    return fig


def plot_tuning(res):
    fig, ax = _fig(6, 3.8)
    ax.plot(range(1, len(res) + 1), res["cv_auc"], "o-", color=PALETTE["accent"])
    ax.set_xlabel("Rank of configuration"); ax.set_ylabel("CV ROC-AUC"); ax.set_title("Random-search results (best first)")
    return fig



# =====================================================================
# 8b. ANIMATED VISUALS  (matplotlib -> self-contained HTML player; works in Jupyter and Streamlit)
# =====================================================================
from matplotlib.animation import FuncAnimation
plt.rcParams["animation.embed_limit"] = 200   # MB


def _to_html(fig, update, n_frames, fps=10, mode="loop"):
    """Turn a figure + update(i) function into an HTML animation player. Returns (html, height_px)."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    html = anim.to_jshtml(fps=fps, default_mode=mode)
    height = int(fig.get_figheight() * fig.dpi) + 100
    plt.close(fig)
    return html, height


def _ease(t):  # smooth start/stop
    return t * t * (3 - 2 * t)


def animate_roc(y_true, probas, n=22):
    """ROC curves of all models drawing themselves."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=80)
    ax.spines[["top", "right"]].set_visible(False)
    curves, lines = [], []
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        curves.append((fpr, tpr))
        (ln,) = ax.plot([], [], lw=3 if "CatBoost" in name else 1.5,
                        label=f"{name} (AUC {manual_roc_auc(y_true, p):.3f})")
        lines.append(ln)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - who separates leavers from stayers best?")
    ax.legend(fontsize=8, loc="lower right")

    def update(i):
        frac = min((i + 1) / n, 1.0)
        for ln, (f, t) in zip(lines, curves):
            k = max(2, int(len(f) * frac))
            ln.set_data(f[:k], t[:k])
        return lines
    return _to_html(fig, update, n + 5, fps=8)


def animate_threshold(y_true, proba):
    """Slide the decision threshold and watch precision / recall / F1 and the error counts change live."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba)
    th = np.round(np.arange(0.05, 0.951, 0.05), 3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), dpi=75, gridspec_kw={"width_ratios": [1.5, 1]})
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    bins = np.linspace(0, 1, 21)
    c0, _ = np.histogram(p[y == 0], bins); c1, _ = np.histogram(p[y == 1], bins)
    ctr, w = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    a1.bar(ctr, c0, w * 0.95, color=PALETTE["stay"], alpha=0.75, label="Actually stayed")
    a1.bar(ctr, c1, w * 0.95, color=PALETTE["leave"], alpha=0.75, label="Actually left")
    vline = a1.axvline(0.5, color="k", lw=2.5)
    a1.set_xlabel("Model's predicted probability of leaving"); a1.set_ylabel("Employees")
    a1.set_xlim(0, 1); a1.legend(loc="upper right")
    a1.set_title("Everyone right of the line is flagged 'will leave'", fontsize=10)
    names = ["Precision", "Recall", "F1"]
    bars = a2.barh(names, [0, 0, 0], color=["#008F87", "#FF686B", "#292929"])
    a2.set_xlim(0, 1.12); a2.invert_yaxis()
    a2.set_title("Metrics at this threshold", fontsize=10)
    txt = [a2.text(0.02, i, "", va="center", fontweight="bold") for i in range(3)]
    sup = fig.suptitle("", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        t = th[i]
        m = manual_metrics(y, (p >= t).astype(int))
        vline.set_xdata([t, t])
        for b, tx, k in zip(bars, txt, names):
            b.set_width(m[k]); tx.set_x(m[k] + 0.02); tx.set_text(f"{m[k]:.2f}")
        sup.set_text(f"Threshold {t:.2f}  -  caught {m['TP']} of {m['TP'] + m['FN']} leavers, "
                     f"{m['FP']} false alarms")
        return []
    return _to_html(fig, update, len(th), fps=4, mode="reflect")


def animate_boosting(gbm, X, y_true, n_frames=22):
    """Watch the from-scratch boosting model learn: scores of leavers vs stayers separate round by round."""
    if not gbm.stumps_:
        raise ValueError("The manual GBM has no stumps.")
    y = np.asarray(y_true).astype(int)
    rounds = sorted(set(np.linspace(1, len(gbm.stumps_), n_frames).astype(int)))
    stages = gbm.staged_decision(X, rounds)
    bins = np.linspace(0, 1, 26)
    cnt, aucs = [], []
    for r in rounds:
        pr = 1 / (1 + np.exp(-np.clip(stages[r], -30, 30)))
        cnt.append((np.histogram(pr[y == 0], bins)[0], np.histogram(pr[y == 1], bins)[0]))
        aucs.append(manual_roc_auc(y, pr))
    ymax = max(max(c0.max(), c1.max()) for c0, c1 in cnt) * 1.1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), dpi=75)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    ctr, w = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    b0 = a1.bar(ctr, cnt[0][0], w * 0.95, color=PALETTE["stay"], alpha=0.75, label="Actually stayed")
    b1 = a1.bar(ctr, cnt[0][1], w * 0.95, color=PALETTE["leave"], alpha=0.75, label="Actually left")
    a1.set_ylim(0, ymax); a1.set_xlim(0, 1); a1.legend(loc="upper right")
    a1.set_xlabel("Predicted probability of leaving"); a1.set_ylabel("Test employees")
    a1.set_title("Predictions separate as stumps are added", fontsize=10)
    (ln,) = a2.plot([], [], "o-", color=PALETTE["accent"])
    a2.set_xlim(0, rounds[-1]); a2.set_ylim(max(0.4, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    a2.set_xlabel("Boosting round"); a2.set_ylabel("Test ROC-AUC"); a2.set_title("Quality of ranking", fontsize=10)
    sup = fig.suptitle("", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        for rect, h in zip(b0, cnt[i][0]): rect.set_height(h)
        for rect, h in zip(b1, cnt[i][1]): rect.set_height(h)
        ln.set_data(rounds[:i + 1], aucs[:i + 1])
        sup.set_text(f"Manual gradient boosting - round {rounds[i]}   (AUC {aucs[i]:.3f})")
        return []
    return _to_html(fig, update, len(rounds), fps=6)


def animate_model_race(table):
    """Bar race: bars grow to each model's ROC-AUC, then morph to F1, then Recall."""
    metrics = [m for m in ["ROC-AUC", "F1", "Recall"] if m in table.columns]
    names = list(table.index)
    cur = np.zeros(len(names))
    frames = []
    for m in metrics:
        tgt = table[m].values.astype(float)
        for k in range(8):
            frames.append((m, cur + (tgt - cur) * _ease((k + 1) / 8)))
        frames += [(m, tgt)] * 5
        cur = tgt
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=80)
    ax.spines[["top", "right"]].set_visible(False)
    colors = [PALETTE["accent"] if "CatBoost" in n else ("#E0A800" if "Manual" in n else PALETTE["stay"]) for n in names]
    bars = ax.barh(names, np.zeros(len(names)), color=colors)
    ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    txt = [ax.text(0.01, i, "", va="center", fontweight="bold") for i in range(len(names))]
    fig.tight_layout()

    def update(i):
        m, vals = frames[i]
        for b, tx, v in zip(bars, txt, vals):
            b.set_width(v); tx.set_x(v + 0.01); tx.set_text(f"{v:.3f}")
        ax.set_title(f"Model comparison - {m}", fontsize=12, fontweight="bold")
        return []
    return _to_html(fig, update, len(frames), fps=8)


def animate_importance(imp, top=12):
    """Feature-importance bars growing one by one."""
    d = imp.head(top)
    names, vals = list(d["feature"])[::-1], np.array(d["importance"])[::-1]
    n_steps = 16
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(d) + 1.6), dpi=80)
    ax.spines[["top", "right"]].set_visible(False)
    bars = ax.barh(names, np.zeros(len(d)), color=PALETTE["accent"])
    ax.set_xlim(0, vals.max() * 1.12); ax.set_title("What drives attrition? (CatBoost feature importance)")
    fig.tight_layout()

    def update(i):
        # features appear from the least to the most important
        for k, b in enumerate(bars):
            start = k * (n_steps / (len(bars) + 6))
            frac = np.clip((i - start) / 8, 0, 1)
            b.set_width(vals[k] * _ease(frac))
        return []
    return _to_html(fig, update, n_steps + 8, fps=8)


def animate_tuning(res):
    """Random search: each trial appears as a dot, the line shows the best score found so far."""
    r = res.sort_values("trial")
    x, y = r["trial"].values, r["cv_auc"].values
    best = np.maximum.accumulate(y)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=80)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.5, x.max() + 0.5)
    pad = max((y.max() - y.min()) * 0.3, 0.005)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("Tuning trial"); ax.set_ylabel("Cross-validated ROC-AUC")
    (pts,) = ax.plot([], [], "o", color=PALETTE["stay"], ms=9, alpha=0.7, label="trial")
    (cur,) = ax.plot([], [], "o", color="#F58518", ms=13, label="current trial")
    (bl,) = ax.plot([], [], "-", color=PALETTE["accent"], lw=2.5, drawstyle="steps-post", label="best so far")
    ax.legend(loc="lower right", fontsize=8)

    def update(i):
        k = min(i, len(x) - 1)
        pts.set_data(x[:k + 1], y[:k + 1]); cur.set_data([x[k]], [y[k]]); bl.set_data(x[:k + 1], best[:k + 1])
        ax.set_title(f"CatBoost hyper-parameter search - trial {x[k]}: AUC {y[k]:.3f} (best {best[k]:.3f})",
                     fontsize=10)
        return []
    return _to_html(fig, update, len(x) + 4, fps=2)


def build_animations(b):
    """All animations for a trained bundle -> {title: (html, height)}. A failing one is skipped, not fatal."""
    y = b["yte"].values if hasattr(b["yte"], "values") else b["yte"]
    jobs = {
        "How the from-scratch boosting model learns": lambda: animate_boosting(b["gbm"], b["Xte_num"], y),
        "Hyper-parameter search": lambda: animate_tuning(b["tuning"]),
        "ROC curves": lambda: animate_roc(y, b["probas"]),
        "Moving the decision threshold": lambda: animate_threshold(y, b["cb_proba"]),
        "Model comparison race": lambda: animate_model_race(b["table"]),
        "Feature importance": lambda: animate_importance(b["importance"]),
    }
    if b.get("tuning") is None:
        jobs.pop("Hyper-parameter search")
    out = {}
    for k, f in jobs.items():
        try:
            out[k] = f()
        except Exception as e:  # noqa
            print(f"[animation skipped] {k}: {e}")
    return out

# =====================================================================
# END-TO-END PIPELINE (used by the Streamlit app)
# =====================================================================
def prepare_xy(df, target):
    y_all = encode_target(df[target])
    keep = y_all.notna()
    X = df.loc[keep].drop(columns=[target]).reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    dup = pd.concat([X, y.rename("__y__")], axis=1).duplicated().values
    return X.loc[~dup].reset_index(drop=True), y.loc[~dup].reset_index(drop=True), int(dup.sum())


def train_pipeline(df, target, test_size=0.2, tune=True, n_iter=8, cv=3, seed=42, progress=None):
    require_catboost()
    X, y, n_dups = prepare_xy(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = Preprocessor().fit(Xtr)
    Xtr_cb, Xte_cb = pre.transform_cb(Xtr), pre.transform_cb(Xte)
    Xtr_num, Xte_num = pre.transform_num(Xtr), pre.transform_num(Xte)

    if tune:
        best, tuning = random_search_catboost(Xtr_cb, ytr, pre.cat_cols_, n_iter=n_iter, cv=cv, seed=seed,
                                              progress=progress)
    else:
        best, tuning = dict(DEFAULT_CB_PARAMS), None
    cb = make_catboost(best, pre.cat_cols_, seed).fit(Xtr_cb, ytr)
    cb_proba = cb.predict_proba(Xte_cb)[:, 1]

    k, knn_tbl = tune_knn_k(Xtr_num.values, ytr.values, cv=cv, seed=seed)
    table, probas, gbm = compare_models(Xtr_num, ytr, Xte_num, yte, cb_proba, seed=seed, knn_k=k)

    imp = importance_table(cb, Xtr_cb.columns)
    shap_df, _ = catboost_shap(cb, Xte_cb, pre.cat_cols_)
    return dict(pre=pre, cb=cb, gbm=gbm, best_params=best, tuning=tuning, knn_k=k, table=table, probas=probas,
                cb_proba=cb_proba, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, Xte_cb=Xte_cb, Xtr_num=Xtr_num, Xte_num=Xte_num,
                importance=imp, shap=shap_df, n_duplicates=n_dups, target=target)


def predict_new(bundle, df_new, threshold=0.5):
    """Score a NEW upload (target column optional)."""
    d = df_new.drop(columns=[bundle["target"]], errors="ignore")
    Z = bundle["pre"].transform_cb(d)
    p = bundle["cb"].predict_proba(Z)[:, 1]
    out = df_new.copy()
    out["attrition_probability"] = np.round(p, 4)
    out["risk_level"] = pd.cut(p, [-1, 0.3, 0.6, 2], labels=["Low", "Medium", "High"])
    out["predicted_to_leave"] = (p >= threshold).astype(int)
    return out.sort_values("attrition_probability", ascending=False)


# =====================================================================
# DEMO DATA  (synthetic, IBM-HR-like schema) - ONLY for trying the app
# =====================================================================
def make_sample_data(n=1470, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 61, n)
    tot_years = np.clip(age - 18 - rng.integers(0, 8, n), 0, 40)
    level = np.clip(1 + (tot_years // 7) + rng.integers(-1, 2, n), 1, 5)
    income = np.round(2000 + level * 2800 + tot_years * 90 + rng.normal(0, 1200, n)).clip(1009, 19999).astype(int)
    yrs_co = np.clip((tot_years * rng.uniform(0.1, 0.9, n)).astype(int), 0, 40)
    d = pd.DataFrame({
        "Age": age,
        "BusinessTravel": rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[.7, .2, .1]),
        "DailyRate": rng.integers(100, 1500, n),
        "Department": rng.choice(["Research & Development", "Sales", "Human Resources"], n, p=[.65, .30, .05]),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Education": rng.integers(1, 6, n),
        "EducationField": rng.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], n),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": rng.integers(1, 5, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "JobInvolvement": rng.integers(1, 5, n),
        "JobLevel": level,
        "JobRole": rng.choice(["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
                               "Healthcare Representative", "Sales Representative"], n),
        "JobSatisfaction": rng.integers(1, 5, n),
        "MaritalStatus": rng.choice(["Married", "Single", "Divorced"], n, p=[.46, .32, .22]),
        "MonthlyIncome": income,
        "NumCompaniesWorked": rng.integers(0, 10, n),
        "Over18": "Y",
        "OverTime": rng.choice(["Yes", "No"], n, p=[.28, .72]),
        "PercentSalaryHike": rng.integers(11, 26, n),
        "PerformanceRating": rng.choice([3, 4], n, p=[.85, .15]),
        "StandardHours": 80,
        "StockOptionLevel": rng.integers(0, 4, n),
        "TotalWorkingYears": tot_years,
        "TrainingTimesLastYear": rng.integers(0, 7, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "YearsAtCompany": yrs_co,
        "YearsInCurrentRole": np.clip(yrs_co - rng.integers(0, 4, n), 0, 20),
        "YearsSinceLastPromotion": np.clip(rng.integers(0, 8, n), 0, yrs_co + 1),
        "YearsWithCurrManager": np.clip(yrs_co - rng.integers(0, 5, n), 0, 20),
    })
    logit = (-1.6 + 1.5 * (d.OverTime == "Yes") + 0.8 * (d.MaritalStatus == "Single")
             + 0.6 * (d.BusinessTravel == "Travel_Frequently") - 0.35 * (d.JobSatisfaction - 2.5)
             - 0.3 * (d.WorkLifeBalance - 2.5) - 0.00012 * (d.MonthlyIncome - 6500)
             - 0.03 * (d.TotalWorkingYears) - 0.35 * d.StockOptionLevel + 0.03 * d.DistanceFromHome
             + 0.5 * (d.JobRole == "Sales Representative"))
    d["Attrition"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    # inject a few missing values so the missing-value step has something to do
    for col, k in [("MonthlyIncome", 12), ("JobSatisfaction", 9), ("Department", 7)]:
        d.loc[rng.choice(n, k, replace=False), col] = np.nan
    return d