"""
Employee Attrition Predictor (CatBoost) - Streamlit app
Run:  streamlit run app.py
Upload any HR CSV/Excel file -> choose the target column -> train -> evaluate -> explain -> score new employees.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

import attrition_core as core
from hero import hero_html

st.set_page_config(page_title="Employee Attrition - CatBoost", page_icon="📉", layout="wide")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@800&family=DM+Sans:wght@400;500;700&display=swap');
:root { --yellow:#FFD43B; --ink:#292929; --teal:#008F87; --coral:#FF686B; --cream:#FFF8DC; --line:#E8E3D4; --muted:#737373; }
html, body, .stApp, .stMarkdown, p, label, li { font-family:'DM Sans', Inter, system-ui, sans-serif; color:var(--ink); }
.block-container { padding-top:1.4rem; max-width:1250px; }
h1, h2, h3, h4 { font-family:'Barlow Condensed', Oswald, Impact, 'Arial Narrow', sans-serif !important;
                 font-weight:800 !important; text-transform:uppercase; letter-spacing:.01em; color:var(--ink); }
h2, h3 { font-size:1.9rem !important; line-height:1.05; }
/* KPI cards */
.kpi { background:#fff; border-radius:24px; padding:18px 22px; border:1px solid var(--line); border-top:7px solid var(--k);
       box-shadow:0 6px 18px rgba(41,41,41,.07); }
.kpi .v { font-family:'Barlow Condensed', Oswald, Impact, sans-serif; font-size:2.6rem; font-weight:800; color:var(--k); line-height:1; }
.kpi .l { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.16em; margin-top:6px; font-weight:700; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); padding-bottom:6px; }
.stTabs [data-baseweb="tab"] { background:var(--cream); border-radius:999px; padding:8px 18px; font-weight:700; border:1px solid var(--line); height:auto; }
.stTabs [aria-selected="true"] { background:var(--yellow); border-color:var(--ink); }
.stTabs [aria-selected="true"] p { color:var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
/* sidebar + buttons */
section[data-testid="stSidebar"] { background:var(--cream); border-right:1px solid var(--line); }
.stButton > button[kind="primary"] { background:var(--coral); border:0; color:#fff; font-weight:800; letter-spacing:.04em;
        text-transform:uppercase; border-radius:999px; padding:.65rem 1.2rem; box-shadow:0 6px 16px rgba(255,104,107,.38); }
.stButton > button[kind="primary"]:hover { background:var(--ink); color:#fff; }
.stDownloadButton > button { border-radius:999px; border:2px solid var(--ink); font-weight:700; }
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; border:1px solid var(--line); }
</style>
""", unsafe_allow_html=True)
components.html(hero_html(), height=420, scrolling=False)


def kpi(col, label, value, color):
    col.markdown(f'<div class="kpi" style="--k:{color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


if not core.CATBOOST_OK:
    st.error("CatBoost is not installed. Run `pip install catboost` and restart the app.")
    st.stop()

# --------------------------------------------------------------- sidebar: data
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    use_demo = st.checkbox("Use synthetic demo data", value=up is None,
                           help="Synthetic data in the IBM-HR layout, only to try the app.")

if up is not None:
    try:
        df = core.load_data(up)
        source_name = up.name
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
elif use_demo:
    df = core.make_sample_data()
    source_name = "demo"
    st.info("Showing **synthetic demo data**. Upload your real dataset in the sidebar.")
else:
    st.info("⬅️ Upload a CSV / Excel file to begin.")
    st.stop()

with st.sidebar:
    cols = list(df.columns)
    target = st.selectbox("Target column (attrition)", cols, index=cols.index(core.guess_target(df)))
    st.header("2 · Training options")
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    tune = st.checkbox("Tune CatBoost (random search + CV)", value=True)
    n_iter = st.slider("Search iterations", 3, 25, 8, disabled=not tune)
    cv = st.slider("CV folds", 3, 5, 3)
    train_btn = st.button("🚀 Train model", type="primary", use_container_width=True)
    st.header("3 · Decision")
    threshold = st.slider("Attrition threshold", 0.05, 0.95, 0.50, 0.01,
                          help="Employees with probability above this are flagged as 'will leave'.")

data_key = (source_name, df.shape, tuple(df.columns), target)

try:
    y_enc = core.encode_target(df[target])
except Exception as e:
    st.error(f"Target problem: {e}")
    st.stop()

if train_btn:
    bar = st.progress(0.0, text="Preparing…")
    try:
        with st.spinner("Training models (CatBoost tuning can take a minute)…"):
            bundle = core.train_pipeline(df, target, test_size=test_size, tune=tune, n_iter=n_iter, cv=cv,
                                         progress=lambda f: bar.progress(min(f, 1.0), text="Tuning CatBoost…"))
        st.session_state["bundle"], st.session_state["key"] = bundle, data_key
        st.session_state["just_trained"] = True
    except Exception as e:
        st.error(f"Training failed: {e}")
    bar.empty()

bundle = st.session_state.get("bundle")
if bundle is not None and st.session_state.get("key") != data_key:
    st.warning("The data or target changed since the last training - click **Train model** again.")

(t_data, t_anim, t_pre, t_models, t_eval, t_xai, t_pred) = st.tabs(
    ["1 Dataset", "✨ Animated results", "2 Preprocessing", "3 Models & comparison", "4 CatBoost evaluation",
     "5 Explainability", "6 Predict"])

# --------------------------------------------------------------- 1 dataset
with t_data:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Employees (rows)", f"{len(df):,}", "#292929")
    kpi(c2, "Columns", df.shape[1], "#008F87")
    kpi(c3, "Attrition rate", f"{y_enc.mean():.1%}", "#FF686B")
    kpi(c4, "Missing cells", int(df.isna().sum().sum()), "#C99700")
    st.write("")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Feature summary")
    st.dataframe(core.describe_dataset(df, target), use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    with a:
        show(core.plot_class_balance(y_enc.dropna()))
    with b:
        show(core.plot_missing(df))
    st.caption("The target is imbalanced, so accuracy alone is misleading - look at recall, F1 and ROC-AUC.")

if bundle is None:
    for t in (t_anim, t_pre, t_models, t_eval, t_xai, t_pred):
        with t:
            st.info("Click **🚀 Train model** in the sidebar first.")
    st.stop()

pre, cb = bundle["pre"], bundle["cb"]
yte, p_cb = bundle["yte"].values, bundle["cb_proba"]

# --------------------------------------------------------------- animated results
with t_anim:
    if st.session_state.pop("just_trained", False):
        st.balloons()
        st.toast("Training complete! Open ✨ Animated results to generate the animations.", icon="✅")
    st.subheader("Your results, in motion")
    st.caption("Press ▶ on each player. You can drag the slider to scrub, and use the speed buttons.")
    if not bundle.get("anims"):
        st.info("Animations are created on demand so training stays fast (about 10 seconds).")
        if st.button("🎬 Generate animations", type="primary"):
            with st.spinner("Rendering animated visuals…"):
                bundle["anims"] = core.build_animations(bundle)
            st.rerun()
    anims = bundle.get("anims", {})
    captions = {
        "How the from-scratch boosting model learns":
            "Teal = employees who stayed, coral = employees who left. Every added stump pulls the two groups apart.",
        "Hyper-parameter search": "Each dot is one CatBoost configuration tried; the dark line is the best score so far.",
        "ROC curves": "The higher and further left a curve climbs, the better the model separates leavers from stayers.",
        "Moving the decision threshold":
            "Slide the cut-off: a lower threshold catches more leavers (recall up) but raises false alarms (precision down).",
        "Model comparison race": "Bars morph from ROC-AUC to F1 to Recall - dark is CatBoost, gold is the from-scratch model.",
        "Feature importance": "The factors CatBoost relies on most to predict who leaves.",
    }
    names = list(anims)
    # wide (two-panel) animations take a full row; the others share a row
    wide = {"How the from-scratch boosting model learns", "Moving the decision threshold"}
    for k in names:
        if k in wide:
            st.markdown(f"#### {k}")
            st.caption(captions.get(k, ""))
            components.html(anims[k][0], height=anims[k][1] + 10, scrolling=False)
    narrow = [k for k in names if k not in wide]
    for i in range(0, len(narrow), 2):
        cols = st.columns(2)
        for col, k in zip(cols, narrow[i:i + 2]):
            with col:
                st.markdown(f"#### {k}")
                st.caption(captions.get(k, ""))
                components.html(anims[k][0], height=anims[k][1] + 10, scrolling=False)

# --------------------------------------------------------------- 2 preprocessing
with t_pre:
    st.subheader("What the preprocessing did")
    st.caption("Fitted on the training split only, so nothing leaks from the test data.")
    if bundle["n_duplicates"]:
        st.write(f"• Removed **{bundle['n_duplicates']}** duplicate rows.")
    for line in pre.log_:
        st.write("•", line)
    st.write(f"• Train/test split: **{len(bundle['ytr'])} / {len(bundle['yte'])}** rows (stratified).")

# --------------------------------------------------------------- 3 comparison
with t_models:
    st.subheader("Model comparison on the held-out test set")
    st.dataframe(bundle["table"].style.format("{:.3f}").highlight_max(axis=0, color="#cdeccd"),
                 use_container_width=True)
    a, b = st.columns(2)
    with a:
        show(core.plot_model_comparison(bundle["table"]))
    with b:
        show(core.plot_roc(yte, bundle["probas"]))
    st.markdown("**Best parameters for CatBoost:** `" + str(bundle["best_params"]) + "`  \n"
                f"**Best k for KNN:** `{bundle['knn_k']}`")
    if bundle["tuning"] is not None:
        with st.expander("Random-search results"):
            st.dataframe(bundle["tuning"], use_container_width=True)
            show(core.plot_tuning(bundle["tuning"]))
    st.markdown("**Manual GBM (scratch)** is a gradient-boosting model written from scratch with NumPy:")
    show(core.plot_loss_curve(bundle["gbm"]))

# --------------------------------------------------------------- 4 evaluation
with t_eval:
    pred = (p_cb >= threshold).astype(int)
    m = core.evaluate(yte, p_cb, threshold)
    c = st.columns(5)
    for col, k, colr in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                            ["#292929", "#008F87", "#FF686B", "#C99700", "#292929"]):
        kpi(col, k, f"{m[k]:.3f}", colr)
    st.write("")
    a, b = st.columns(2)
    with a:
        show(core.plot_confusion(yte, pred, f"CatBoost @ threshold {threshold:.2f}"))
    with b:
        show(core.plot_threshold_curve(yte, p_cb))
    st.subheader("Manual calculation from the confusion matrix")
    tp, fp, tn, fn = m["TP"], m["FP"], m["TN"], m["FN"]
    n = tp + fp + tn + fn
    st.code(
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn}   (n={n})\n"
        f"Accuracy  = (TP+TN)/n        = ({tp}+{tn})/{n} = {m['Accuracy']:.4f}\n"
        f"Precision = TP/(TP+FP)       = {tp}/({tp}+{fp}) = {m['Precision']:.4f}\n"
        f"Recall    = TP/(TP+FN)       = {tp}/({tp}+{fn}) = {m['Recall']:.4f}\n"
        f"F1        = 2PR/(P+R)        = {m['F1']:.4f}\n"
        f"ROC-AUC   = rank (Mann-Whitney U) formula = {m['ROC-AUC']:.4f}", language="text")
    st.caption("Recall = share of real leavers caught; precision = share of flagged employees who really leave. "
               "Move the threshold in the sidebar to see the trade-off.")

# --------------------------------------------------------------- 5 explainability
with t_xai:
    a, b = st.columns(2)
    with a:
        show(core.plot_importance(bundle["importance"]))
    with b:
        show(core.plot_shap_summary(bundle["shap"], bundle["Xte_cb"]))
    st.subheader("Plain-English findings")
    train_clean = pre.transform_cb(bundle["Xtr"])
    for line in core.auto_insights(train_clean, bundle["ytr"], bundle["importance"]):
        st.write("•", line)
    st.caption("These findings are computed from your data - write your own business interpretation in the report.")

# --------------------------------------------------------------- 6 predict
with t_pred:
    st.subheader("A) Score one employee")
    Xtr = bundle["Xtr"]
    with st.form("single"):
        vals = {}
        cols_ui = st.columns(4)
        for i, f in enumerate(pre.feature_order_):
            with cols_ui[i % 4]:
                if f in pre.num_cols_:
                    med = float(pre.medians_[f])
                    is_int = pd.api.types.is_integer_dtype(Xtr[f])
                    lo, hi = float(Xtr[f].min()), float(Xtr[f].max())
                    if is_int:
                        vals[f] = st.number_input(f, int(lo), int(hi), int(round(med)), step=1)
                    else:
                        vals[f] = st.number_input(f, lo, hi, med)
                else:
                    levels = pre.levels_[f]
                    vals[f] = st.selectbox(f, levels)
        go = st.form_submit_button("Predict attrition risk")
    if go:
        one = pd.DataFrame([vals])
        res = core.predict_new(bundle, one, threshold).iloc[0]
        prob = float(res["attrition_probability"])
        st.metric("Probability of leaving", f"{prob:.1%}", help=f"Risk level: {res['risk_level']}")
        (st.error if prob >= threshold else st.success)(
            f"Risk level **{res['risk_level']}** - model says this employee "
            f"{'is likely to leave' if prob >= threshold else 'is likely to stay'} (threshold {threshold:.2f}).")
        Z = pre.transform_cb(one)
        sh, _ = core.catboost_shap(cb, Z, pre.cat_cols_)
        show(core.plot_shap_waterfall(sh.iloc[0], Z.iloc[0]))

    st.divider()
    st.subheader("B) Score a whole file")
    up2 = st.file_uploader("Upload a new employee file (same columns; target column optional)",
                           type=["csv", "xlsx", "xls"], key="batch")
    if up2 is not None:
        try:
            new = core.load_data(up2)
            scored = core.predict_new(bundle, new, threshold)
            st.write(f"Scored **{len(scored)}** employees - "
                     f"{int((scored['risk_level'] == 'High').sum())} high risk.")
            st.dataframe(scored, use_container_width=True)
            st.download_button("⬇️ Download results (CSV)", scored.to_csv(index=False).encode("utf-8"),
                               "attrition_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score this file: {e}")