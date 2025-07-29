
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------- 1) Load & filter -------------------------------------------------

def load_top_model_data(file_path, current_year=2025):
    from collections import Counter

    print("▶️  Loading and filtering data ...")
    data, model_counter = [], Counter()

    # Pass 1 — find most frequent model
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                attrs = {a["key"]: a.get("value") for a in item.get("attributes", [])}
                m = str(attrs.get("model")).strip()
                if m:
                    model_counter[m] += 1
            except:
                continue

    if not model_counter:
        raise ValueError("No valid models in file")

    top_model = model_counter.most_common(1)[0][0]
    print(f"📈 Analysing most frequent model: {top_model}")

    # Pass 2 — collect rows for the top model
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                attrs = {a["key"]: a.get("value") for a in item.get("attributes", [])}
                if str(attrs.get("model")).strip() != top_model:
                    continue
                data.append({
                    "price_eur": item.get("priceInfo", {}).get("priceCents", 0) / 100,
                    "constructionYear": pd.to_numeric(attrs.get("constructionYear"), errors="coerce"),
                    "mileage_km": pd.to_numeric(attrs.get("mileage", "").replace(".", ""), errors="coerce"),
                })
            except:
                continue

    df = pd.DataFrame(data).dropna()
    before = len(df)
    df = df[df["price_eur"] > 100]
    df["age"] = current_year - df["constructionYear"]
    df = df[df["age"] >= 0]
    after = len(df)
    print(f"✅ Filtered rows: kept {after}/{before}")

    return df, top_model

# ---------- 2) Helpers -------------------------------------------------------

def adjusted_r2(r2, n, p):
    return None if n - p - 1 <= 0 else 1 - (1 - r2) * (n - 1) / (n - p - 1)

def pipeline_for_degree(degree: int):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(fit_intercept=True)
    )

import statsmodels.api as sm

def print_coefficients_with_significance(model, X, y, feature_label=("age", "mileage_km")):
    # Get feature names from the pipeline
    poly = model.named_steps["polynomialfeatures"]
    lr = model.named_steps["linearregression"]
    feature_names = poly.get_feature_names_out(list(feature_label)).tolist()
    
    # Transform X using the polynomial transformer only
    X_poly = poly.transform(X)
    
    # Add constant for intercept
    X_poly = sm.add_constant(X_poly)

    # Fit using OLS
    ols_model = sm.OLS(y, X_poly).fit()

    # Print results
    print("📊 Coefficient Significance Test:")
    summary_df = pd.DataFrame({
    "Feature": ["const"] + feature_names,
    "Coefficient": ols_model.params,
    "P-Value": ols_model.pvalues,
    "T-Statistic": ols_model.tvalues,
    }).round(2)

    print(summary_df.to_string(index=False))
    return summary_df

# ---------- 3) Inner CV scoring (no test leakage) ----------------------------

def inner_cv_scores(X_train_outer, y_train_outer, degree, n_splits=50, inner_test_size=0.2, random_state=42):
    print(f"\n🔁 Evaluating degree={degree} with inner CV ({n_splits} ShuffleSplits, val size={int(inner_test_size*100)}%)")
    splitter = ShuffleSplit(n_splits=n_splits, test_size=inner_test_size, random_state=random_state)
    scores = []
    for i, (tr_idx, val_idx) in enumerate(splitter.split(X_train_outer), start=1):
        model = pipeline_for_degree(degree)
        model.fit(X_train_outer.iloc[tr_idx], y_train_outer.iloc[tr_idx])
        y_val_pred = model.predict(X_train_outer.iloc[val_idx])
        r2 = r2_score(y_train_outer.iloc[val_idx], y_val_pred)
        scores.append(r2)
        print(f"   • split {i:02d}/{n_splits}: inner-val R² = {r2:.4f}")
    mean_r2, std_r2 = float(np.mean(scores)), float(np.std(scores))
    print(f"   → degree {degree}: mean inner-CV R² = {mean_r2:.4f} ± {std_r2:.4f} "
          f"(min={min(scores):.4f}, max={max(scores):.4f})")
    return scores

# ---------- 4) Main flow -----------------------------------------------------

if __name__ == "__main__":
    # -------- paths / options
    path = "raw_listings/Toyota_2025-07-21.jsonl"
    degrees = (1, 2, 3)
    outer_test_size = 0.20
    inner_splits = 50
    inner_val_size = 0.20
    seed = 42

    print("🚀 Starting pipeline\n")

    # -------- load
    df, top_model_name = load_top_model_data(path)
    X, y = df[["age", "mileage_km"]], df["price_eur"]

    # -------- outer split (never used for model selection)
    print("\n✂️  Creating outer train/test split (80/20) ...")
    X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
        X, y, test_size=outer_test_size, random_state=seed
    )
    print(f"   outer-train: {len(y_train_outer)} rows | outer-test: {len(y_test_outer)} rows")

    # -------- evaluate degrees via inner CV (on outer train only)
    degree_stats = {}
    for d in degrees:
        scores = inner_cv_scores(
            X_train_outer, y_train_outer,
            degree=d, n_splits=inner_splits, inner_test_size=inner_val_size, random_state=seed
        )
        degree_stats[d] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    # -------- select best degree by mean inner-CV R²
    best_degree = max(degree_stats.keys(), key=lambda k: degree_stats[k]["mean"])
    best_mean = degree_stats[best_degree]["mean"]
    best_std = degree_stats[best_degree]["std"]
    print(f"\n🏆 Selected degree={best_degree} based on inner-CV mean R²={best_mean:.4f} ± {best_std:.4f}")

    # -------- refit final model on all outer-train with chosen degree
    print("\n🧠 Refitting FINAL model on full outer-train ...")
    final_model = pipeline_for_degree(best_degree)
    final_model.fit(X_train_outer, y_train_outer)
    
    summary_df = print_coefficients_with_significance(final_model, X_train_outer, y_train_outer)
    summary_df
    # -------- evaluate once on untouched outer-test
    print("\n🧪 Evaluating FINAL model on outer-test (never used during selection) ...")
    y_pred_outer = final_model.predict(X_test_outer)
    outer_r2 = r2_score(y_test_outer, y_pred_outer)
    adj_r2 = adjusted_r2(outer_r2, n=len(y_test_outer), p=len(summary_df ) -1 )
    
    # Add this after outer-test evaluation
    print(f"   Outer-train sample size = {len(y_train_outer)}")
    print(f"   Outer-test sample size = {len(y_test_outer)}")
    print(f"   Outer-test R²       = {outer_r2:.4f}")
    print(f"   Outer-test adj. R²  = {adj_r2:.4f}" if adj_r2 is not None else "   Outer-test adj. R²  = N/A")
    

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_outer, y_pred_outer, alpha=0.7)
    plt.plot([y_test_outer.min(), y_test_outer.max()], [y_test_outer.min(), y_test_outer.max()],
            color='red', linestyle='--', label="Ideal fit")
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title("Predicted vs Actual (Outer-Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # -------- summary
    print("\n✅ SUMMARY")
    for d in degrees:
        ds = degree_stats[d]
        print(f"   degree {d}: mean={ds['mean']:.4f} ± {ds['std']:.4f} "
              f"(min={ds['min']:.4f}, max={ds['max']:.4f})")
    print(f"\n🎯 Final choice: degree {best_degree} | Outer-test R²={outer_r2:.4f}")
    print("\n🎉 Done.")
    