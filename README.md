# Employee Attrition Prediction - CatBoost

## Files
| File | Purpose |
|---|---|
| `employee_attrition_catboost.ipynb` | Main submission: explained notebook, sections 1-10 match the rubric |
| `attrition_core.py` | All logic (preprocessing, from-scratch boosting, metrics, tuning, plots). Keep in the same folder |
| `app.py` | Streamlit web app with **file upload** (dynamic data) |
| `hero.py` | Animated header (CatBoost trees growing, risk score filling). Keep next to `app.py` |
| `.streamlit/config.toml` | Colour theme (yellow / charcoal / teal / coral / cream). Run Streamlit from the project folder |
| `REPORT_OUTLINE.md` | Structure for the written report (rubric item 11) |

## Animated visuals (no extra library needed)
After training, the app opens an **✨ Animated results** tab (also shown in the notebook) with six players:
boosting-model learning, tuning search, ROC curves drawing, threshold slider, model bar-race, feature-importance growth.
Press ▶, drag the slider to scrub, change speed. Rendering takes ~20 seconds after training.

## Run
```
pip install -r requirements.txt
jupyter notebook employee_attrition_catboost.ipynb     # set DATA_PATH in section 1
streamlit run app.py                                   # upload CSV/Excel in the sidebar
```
Recommended dataset: IBM HR Analytics Employee Attrition (Kaggle, `WA_Fn-UseC_-HR-Employee-Attrition.csv`).
Any file with a binary "left / stayed" column works (Yes/No, 1/0, True/False).
If no file is found the notebook falls back to SYNTHETIC demo data - never report results from it.

## Rubric coverage
| # | Criterion | Where |
|---|---|---|
| 1 | Dataset understanding | Notebook §1, app tab 1 |
| 2 | Preprocessing | `Preprocessor`, notebook §2, app tab 2 |
| 3 | Manual implementation | `ManualGradientBoosting` (boosting from scratch), notebook §3 |
| 4 | Algorithm application | CatBoost, notebook §4 |
| 5 | Parameter tuning | random search + CV (depth, lr, iterations, l2, class weights; KNN k), §5 |
| 6 | Metrics + manual calc | `manual_metrics`, `manual_roc_auc`, worked example in §6, verified against scikit-learn |
| 7 | Comparison | LogReg, KNN, Random Forest, manual GBM vs CatBoost, §7 |
| 8 | Visualisation | plots generated from the uploaded dataset, §8 |
| 9 | Interpretation | SHAP, auto-insights, §9 (+ your own words) |
| 10 | Documentation | markdown + comments throughout |
| 11 | Report | `REPORT_OUTLINE.md` |
