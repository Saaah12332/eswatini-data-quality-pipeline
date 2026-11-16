"""Bank early-warning risk score.

Builds a one-quarter-ahead distress score from capital, liquidity and asset
quality ratios for 20 institutions over 12 quarters. Compares logistic
regression and a random forest with a time-based (walk-forward) back-test.

The panel is synthetic. Real supervisory returns are confidential, so this
repository publishes a panel with the same shape as the exercise: the ratios
are noisy readings of a latent condition, and distress next quarter is an
event driven by that condition, not a hard cut on the reported ratios.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 7
QUARTERS = [f"{year}Q{q}" for year in (2022, 2023, 2024) for q in (1, 2, 3, 4)]
INSTITUTIONS = [
    "Aldermere Bank",
    "Blackwater Commercial",
    "Cedar & Holm",
    "Dunwich Savings",
    "Eastlane Mutual",
    "Fairhaven Trust",
    "Greyharbour Bank",
    "Highford Retail Bank",
    "Iverson Civic",
    "Juniper Building Society",
    "Keel & Moss",
    "Lowridge Bank",
    "Marrowgate Savings",
    "Northglass Bank",
    "Orchard Civic",
    "Plover Mutual",
    "Quayline Bank",
    "Redwillow Trust",
    "Southmere Building Society",
    "Thornfield Bank",
]

FEATURES = [
    "cet1_ratio",
    "leverage_ratio",
    "lcr",
    "loan_to_deposit",
    "npl_ratio",
    "provision_coverage",
]
FEATURE_LABELS = {
    "cet1_ratio": "CET1 ratio",
    "leverage_ratio": "Leverage ratio",
    "lcr": "Liquidity coverage ratio",
    "loan_to_deposit": "Loan-to-deposit ratio",
    "npl_ratio": "NPL ratio",
    "provision_coverage": "Provision coverage",
}

# Walk-forward: first origin trains through this quarter (inclusive),
# then predicts the next labeled feature-quarter. Feature quarters run
# 2022Q1–2024Q3 because the label is distress in the following quarter.
FIRST_TEST_FEATURE_QUARTER = "2022Q4"
# Threshold is chosen only on the earliest out-of-sample origins.
VAL_ORIGINS = 2


def generate_panel(rng: np.random.Generator) -> pd.DataFrame:
    """20 institutions × 12 quarters.

    Latent condition (higher = weaker) is persistent. Reported ratios are
    noisy readings of it. Distress next quarter is a Bernoulli event of that
    latent condition, so the ratios are informative but not a mechanical rule.
    """
    rows: list[dict] = []
    for name in INSTITUTIONS:
        # Most banks sit in a healthy range. A minority are structurally weaker.
        baseline = float(rng.beta(2.2, 6.0))
        if rng.random() < 0.25:
            baseline = float(rng.uniform(0.45, 0.75))
        state = baseline
        for quarter in QUARTERS:
            state = 0.88 * state + 0.12 * baseline + float(rng.normal(0, 0.045))
            state = float(np.clip(state, 0.01, 0.97))
            noise = rng.normal
            cet1 = float(np.clip(noise(17.4 - 11.5 * state, 0.95), 4.5, 24))
            leverage = float(np.clip(noise(7.6 - 5.0 * state, 0.40), 2.2, 12))
            lcr = float(np.clip(noise(178 - 115 * state, 12.0), 50, 280))
            ltd = float(np.clip(noise(74 + 52 * state, 6.0), 48, 150))
            npl = float(np.clip(noise(0.7 + 13.0 * state, 0.85), 0.15, 22))
            coverage = float(np.clip(noise(88 - 48 * state, 8.0), 20, 140))
            # Distress is next quarter's condition crossing a breach line,
            # plus a shock the six ratios do not observe. The ratios lead
            # the breach when weakness persists, but they do not determine it.
            shock = float(rng.normal(0, 0.12))
            distressed = int((state + shock) > 0.60)
            rows.append(
                {
                    "institution": name,
                    "quarter": quarter,
                    "latent_condition": round(state, 4),
                    "cet1_ratio": round(cet1, 2),
                    "leverage_ratio": round(leverage, 2),
                    "lcr": round(lcr, 1),
                    "loan_to_deposit": round(ltd, 1),
                    "npl_ratio": round(npl, 2),
                    "provision_coverage": round(coverage, 1),
                    "distress_this_quarter": distressed,
                }
            )
    panel = pd.DataFrame(rows)
    panel["quarter_idx"] = panel["quarter"].map({q: i for i, q in enumerate(QUARTERS)})
    panel = panel.sort_values(["institution", "quarter_idx"]).reset_index(drop=True)
    # Label available to the supervisor at the *next* quarter: predict it from this quarter's ratios.
    panel["distress_next_quarter"] = panel.groupby("institution")["distress_this_quarter"].shift(-1)
    return panel


def walk_forward_origins(labeled: pd.DataFrame) -> list[str]:
    feature_quarters = [q for q in QUARTERS if q in set(labeled["quarter"])]
    start = feature_quarters.index(FIRST_TEST_FEATURE_QUARTER)
    return feature_quarters[start:]


def fit_models(train: pd.DataFrame) -> dict[str, object]:
    x = train[FEATURES]
    y = train["distress_next_quarter"].astype(int)
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    logistic.fit(x, y)
    forest.fit(x, y)
    return {"Logistic regression": logistic, "Random forest": forest}


def predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURES])[:, 1]


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Operating point chosen only on the validation origins.

    The desk's target is to flag about 85% of distress cases without
    reviewing more than 40% of institutions. The cutoff is the one whose
    validation hit rate is closest to 85% inside that band. It is frozen
    before the holdout window is scored.
    """
    if y_true.sum() == 0:
        return 0.5
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 61)))
    best_t, best_key = 0.5, None
    for t in candidates:
        pred = scores >= t
        alert_rate = float(pred.mean())
        if alert_rate > 0.40 or alert_rate < 0.05:
            continue
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, pred.astype(int), average="binary", zero_division=0
        )
        key = (abs(float(recall) - 0.85), -float(precision))
        if best_key is None or key < best_key:
            best_key, best_t = key, float(t)
    return best_t


def classification_block(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return {
        "auc": None if auc is None else round(auc, 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "alert_rate": round(float(pred.mean()), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "distress_cases": int(y_true.sum()),
        "flagged_distress": int(tp),
    }


def roc_points(y_true: np.ndarray, scores: np.ndarray, n: int = 40) -> list[dict]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx]


def write_figures(roc: dict[str, list[dict]], quarterly: pd.DataFrame, importances: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, pts in roc.items():
        ax.plot([p["fpr"] for p in pts], [p["tpr"] for p in pts], label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], color="#b9b1a4", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (share of distress flagged)")
    ax.set_title("Time-based back-test: ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_backtest.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(quarterly))
    ax.bar(x - 0.15, quarterly["distress"], width=0.3, label="Distress cases", color="#8f3d32")
    ax.bar(x + 0.15, quarterly["flagged_distress"], width=0.3, label="Flagged one quarter ahead", color="#1e3a34")
    ax.set_xticks(x, quarterly["quarter"], rotation=0)
    ax.set_ylabel("Institutions")
    ax.set_title("Distress cases caught one quarter ahead")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "hits_by_quarter.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ordered = importances.sort_values("importance")
    ax.barh(ordered["label"], ordered["importance"], color="#1e3a34")
    ax.set_xlabel("Random forest importance (share)")
    ax.set_title("Which ratios the score uses")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=140)
    plt.close(fig)


def write_one_pager(payload: dict) -> None:
    h = payload["headline"]
    selected = payload["selected_model"]
    watch = payload["watchlist"]
    lines = [
        "# One-page supervisory summary",
        "",
        f"**As-of quarter:** {payload['as_of']}  ",
        f"**Score:** probability of distress next quarter, from {selected.lower()}.",
        "",
        "## Question",
        "",
        "Which of the 20 institutions look likely to be in distress next quarter, "
        "using only the capital, liquidity and asset-quality ratios already on the return?",
        "",
        "## Sample",
        "",
        "A balanced panel of 20 institutions and 12 quarters (2022Q1–2024Q4). "
        "Six ratios: CET1, leverage, liquidity coverage (LCR), loan-to-deposit, "
        "non-performing loans (NPL) and provision coverage. "
        "The label is distress in the *following* quarter, so every score is an early warning, not a description of the quarter just reported.",
        "",
        "## How the back-test works",
        "",
        "The models are refit every quarter. Training uses only earlier quarters — never the quarter being scored and never the future. "
        f"The alert threshold is the cutoff whose hit rate on the first {VAL_ORIGINS} out-of-sample quarters was closest to 85%, without alerting on more than 40% of institutions. It was then frozen. "
        f"The result below is the later holdout ({h['window']}), which the threshold did not see.",
        "",
        "## Result",
        "",
        f"**{h['flagged_distress']} of {h['distress_cases']} distress cases ({h['recall_pct']}%) were flagged one quarter ahead** "
        f"by the {selected.lower()} (AUC {h['auc']:.2f}). "
        f"Of the names sent for review, {h['precision_pct']}% were subsequently in distress. "
        f"The alert rate was {h['alert_rate_pct']}% of institution-quarters.",
        "",
        "Logistic regression is reported beside the forest so a linear, inspectable score is always available. "
        "The forest is used for the watchlist when it wins the validation AUC; otherwise the logistic score is used.",
        "",
        "## Watchlist going into the next quarter",
        "",
        "Scores below use the latest filed ratios "
        f"({payload['as_of']}) and the model trained only on history before that quarter. "
        "They are a queue for review, not a capital decision.",
        "",
    ]
    if not watch:
        lines.append("No institution is above the alert threshold this quarter.")
    else:
        lines.append("| Institution | Score | CET1 % | LCR % | NPL % | Leverage % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in watch:
            lines.append(
                f"| {row['institution']} | {row['score']:.2f} | {row['cet1_ratio']:.1f} | "
                f"{row['lcr']:.0f} | {row['npl_ratio']:.1f} | {row['leverage_ratio']:.1f} |"
            )
    lines += [
        "",
        "## What this score is not",
        "",
        "It does not use market prices, governance findings or qualitative supervision. "
        "A high score means the ratios resemble institutions that were in distress a quarter later. "
        "A low score is not a clean bill of health.",
        "",
        "## Caveat on the published data",
        "",
        "Institution-level supervisory returns cannot be published. "
        "The panel in `data/bank_panel.csv` is a synthetic panel with the same design: "
        "20 institutions, 12 quarters, the six ratios, and a next-quarter distress event that is "
        "correlated with those ratios but not a hard threshold on them. "
        "Every number in this note is computed by `python analysis.py`.",
        "",
    ]
    (REPORTS / "one_page_supervisory_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    panel = generate_panel(rng)
    panel.drop(columns=["latent_condition"]).to_csv(DATA / "bank_panel.csv", index=False)

    labeled = panel.dropna(subset=["distress_next_quarter"]).copy()
    labeled["distress_next_quarter"] = labeled["distress_next_quarter"].astype(int)
    origins = walk_forward_origins(labeled)

    oos_frames: list[pd.DataFrame] = []
    pooled_scores: dict[str, list[pd.DataFrame]] = {
        "Logistic regression": [],
        "Random forest": [],
    }
    last_models: dict[str, object] | None = None

    for origin in origins:
        origin_idx = int(labeled.loc[labeled["quarter"] == origin, "quarter_idx"].iloc[0])
        train = labeled[labeled["quarter_idx"] < origin_idx]
        test = labeled[labeled["quarter"] == origin]
        models = fit_models(train)
        last_models = models
        block = test[["institution", "quarter", "distress_next_quarter"] + FEATURES].copy()
        for name, model in models.items():
            block[name] = predict_proba(model, test)
            pooled_scores[name].append(block[["institution", "quarter", "distress_next_quarter", name]].copy())
        oos_frames.append(block)

    assert last_models is not None
    oos = oos_frames[0][["institution", "quarter", "distress_next_quarter"] + FEATURES].copy()
    for frame in oos_frames[1:]:
        oos = pd.concat([oos, frame[["institution", "quarter", "distress_next_quarter"] + FEATURES]], ignore_index=True)
    for name in pooled_scores:
        part = pd.concat(pooled_scores[name], ignore_index=True)
        oos[name] = part[name].values

    val_quarters = origins[:VAL_ORIGINS]
    eval_quarters = origins[VAL_ORIGINS:]
    val_mask = oos["quarter"].isin(val_quarters)
    eval_mask = oos["quarter"].isin(eval_quarters)

    model_reports = []
    thresholds = {}
    for name in ("Logistic regression", "Random forest"):
        y_val = oos.loc[val_mask, "distress_next_quarter"].to_numpy()
        s_val = oos.loc[val_mask, name].to_numpy()
        thresholds[name] = best_threshold(y_val, s_val)
        y_eval = oos.loc[eval_mask, "distress_next_quarter"].to_numpy()
        s_eval = oos.loc[eval_mask, name].to_numpy()
        block = classification_block(y_eval, s_eval, thresholds[name])
        block["name"] = name
        block["validation_auc"] = round(float(roc_auc_score(y_val, s_val)), 4)
        model_reports.append(block)

    # Select on validation AUC, not on the evaluation window.
    selected = max(
        ("Logistic regression", "Random forest"),
        key=lambda n: next(m["validation_auc"] for m in model_reports if m["name"] == n),
    )
    selected_block = next(m for m in model_reports if m["name"] == selected)

    # Latest quarter has no future label. Score it with a model trained on all labeled history.
    final_models = fit_models(labeled)
    latest = panel[panel["quarter"] == QUARTERS[-1]].copy()
    latest["score"] = predict_proba(final_models[selected], latest)
    latest["flagged"] = latest["score"] >= thresholds[selected]
    watch = (
        latest[latest["flagged"]]
        .sort_values("score", ascending=False)
        .to_dict(orient="records")
    )

    quarterly_rows = []
    for quarter, g in oos[eval_mask].groupby("quarter", sort=True):
        pred = g[selected] >= thresholds[selected]
        y = g["distress_next_quarter"].astype(bool)
        quarterly_rows.append(
            {
                "quarter": quarter,
                "distress": int(y.sum()),
                "flagged_distress": int((pred & y).sum()),
                "alerts": int(pred.sum()),
            }
        )
    quarterly = pd.DataFrame(quarterly_rows)

    coefs = final_models["Logistic regression"].named_steps["clf"].coef_[0]
    # Coefficients are on the standardised ratios: sign is the direction, magnitude is comparable.
    coefficients = [
        {
            "feature": f,
            "label": FEATURE_LABELS[f],
            "coefficient": round(float(c), 3),
        }
        for f, c in sorted(zip(FEATURES, coefs), key=lambda kv: -abs(kv[1]))
    ]
    importances_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "label": [FEATURE_LABELS[f] for f in FEATURES],
            "importance": final_models["Random forest"].feature_importances_,
        }
    )
    importances = [
        {"feature": r.feature, "label": r.label, "importance": round(float(r.importance), 4)}
        for r in importances_df.sort_values("importance", ascending=False).itertuples()
    ]

    roc = {
        name: roc_points(
            oos.loc[eval_mask, "distress_next_quarter"].to_numpy(),
            oos.loc[eval_mask, name].to_numpy(),
        )
        for name in ("Logistic regression", "Random forest")
    }
    write_figures(roc, quarterly, importances_df)

    precision_pct = round(selected_block["precision"] * 100)
    headline = {
        "model": selected,
        "window": f"{eval_quarters[0]}–{eval_quarters[-1]}",
        "validation_window": f"{val_quarters[0]}–{val_quarters[-1]}",
        "flagged_distress": selected_block["flagged_distress"],
        "distress_cases": selected_block["distress_cases"],
        "recall": selected_block["recall"],
        "recall_pct": round(selected_block["recall"] * 100),
        "precision": selected_block["precision"],
        "precision_pct": precision_pct,
        "precision_text": f"{precision_pct} percent",
        "alert_rate": selected_block["alert_rate"],
        "alert_rate_pct": round(selected_block["alert_rate"] * 100),
        "auc": selected_block["auc"],
        "threshold": selected_block["threshold"],
    }

    oos_out = oos.copy()
    oos_out["selected_score"] = oos_out[selected]
    oos_out["flagged"] = oos_out["selected_score"] >= thresholds[selected]
    oos_out.to_csv(DATA / "backtest_predictions.csv", index=False)

    latest_rows = []
    for r in latest.sort_values("score", ascending=False).itertuples():
        latest_rows.append(
            {
                "institution": r.institution,
                "quarter": r.quarter,
                "score": round(float(r.score), 4),
                "flagged": bool(r.flagged),
                "cet1_ratio": r.cet1_ratio,
                "leverage_ratio": r.leverage_ratio,
                "lcr": r.lcr,
                "loan_to_deposit": r.loan_to_deposit,
                "npl_ratio": r.npl_ratio,
                "provision_coverage": r.provision_coverage,
            }
        )

    payload = {
        "sample": {
            "institutions": 20,
            "quarters": 12,
            "quarter_start": QUARTERS[0],
            "quarter_end": QUARTERS[-1],
            "labeled_rows": int(len(labeled)),
            "features": FEATURES,
        },
        "as_of": QUARTERS[-1],
        "selected_model": selected,
        "headline": headline,
        "models": model_reports,
        "coefficients": coefficients,
        "importances": importances,
        "roc": roc,
        "by_quarter": quarterly_rows,
        "latest": latest_rows,
        "watchlist": [
            {
                "institution": r["institution"],
                "score": round(float(r["score"]), 4),
                "cet1_ratio": r["cet1_ratio"],
                "lcr": r["lcr"],
                "npl_ratio": r["npl_ratio"],
                "leverage_ratio": r["leverage_ratio"],
            }
            for r in latest_rows
            if r["flagged"]
        ],
        "base_rate_labeled": round(float(labeled["distress_next_quarter"].mean()), 4),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_one_pager(payload)

    web = ROOT.parents[1] / "src" / "data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "bank.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"headline": headline, "models": model_reports, "base_rate": payload["base_rate_labeled"]}, indent=2))


if __name__ == "__main__":
    main()
