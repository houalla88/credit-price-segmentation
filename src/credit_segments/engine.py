"""Three-leaf rule search with independent chronological evaluation."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

MISSING = "__MISSING__"


def validate(df, features, controls):
    required = {
        "application_id",
        "date",
        "product",
        "approved",
        "booked",
        "rate_pp",
        "gnos",
        *features,
        *controls,
    }
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    if (
        df.empty
        or df.application_id.isna().any()
        or df.application_id.duplicated().any()
    ):
        raise ValueError(
            "Application IDs must be present and unique; input cannot be empty"
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df.date, errors="raise", utc=True)
    if df.date.isna().any() or df["product"].isna().any():
        raise ValueError("Dates and products cannot be missing")
    for column in ["approved", "booked", "rate_pp", "gnos", *controls]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    for column in ["approved", "booked"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary")
    if (
        (df.booked > df.approved) | (df.gnos < 0) | ((df.booked == 0) & (df.gnos != 0))
    ).any():
        raise ValueError(
            "Booking requires approval; GNoS must be nonnegative and zero when not booked"
        )
    for feature in features:
        if (df[feature].dropna().astype(str) == MISSING).any():
            raise ValueError(f"Reserved category {MISSING} in {feature}")
        df[feature] = df[feature].fillna(MISSING).astype(str)
    return df.sort_values(["date", "application_id"]).reset_index(drop=True)


def mask_rule(df, rule):
    return (
        df[rule["feature"]].fillna(MISSING).astype(str).isin(rule["values"]).to_numpy()
    )


def assign(df, model):
    first = mask_rule(df, model["root"])
    second = mask_rule(df, model["child"])
    split = first if model["split_left"] else ~first
    leaves = np.where(split, np.where(second, 1, 2), 0)
    return np.array([model["labels"][str(int(x))] for x in leaves])


def estimate(df, controls, min_approved, min_rate_std):
    population = df.loc[df.approved == 1]
    if len(population) < min_approved or population.booked.nunique() < 2:
        return None
    if population.rate_pp.std() < min_rate_std:
        return None
    x = sm.add_constant(population[["rate_pp", *controls]], has_constant="add")
    if len(population) <= x.shape[1] + 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        return None
    fit = sm.OLS(population.booked, x).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["rate_pp"]
    values = [fit.params["rate_pp"], lo, hi]
    if not np.isfinite(values).all():
        return None
    return {
        "beta": float(values[0]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_approved": len(population),
    }


def candidate_rules(df, features):
    rules = []
    for feature in features:
        categories = sorted(df[feature].unique())
        if len(categories) > 8:
            raise ValueError(
                f"{feature}: maximum 8 categories; bin continuous variables upstream"
            )
        # Complementary subsets describe the same split. Fix the first category on the left.
        for size in range(len(categories)):
            for extra in combinations(categories[1:], size):
                values = [categories[0], *extra]
                if len(values) < len(categories):
                    rules.append({"feature": feature, "values": values})
    if len(rules) > 150:
        raise ValueError(
            "Too many candidate rules (>150); reduce features or categories"
        )
    return rules


def run(
    df,
    features,
    controls,
    product,
    min_share=0.25,
    min_approved=50,
    min_rate_std=0.15,
    train_fraction=0.7,
):
    if (
        not features
        or len(set(features)) != len(features)
        or len(set(controls)) != len(controls)
    ):
        raise ValueError("Features must be nonempty; lists cannot contain duplicates")
    if set(features) & set(controls):
        raise ValueError("Features and numeric controls must be distinct")
    forbidden = {
        "application_id",
        "date",
        "product",
        "approved",
        "booked",
        "rate_pp",
        "gnos",
    }
    if forbidden & (set(features) | set(controls)):
        raise ValueError(
            "Identifiers, outcomes and reserved columns cannot be predictors"
        )
    if (
        not 0 < min_share <= 1 / 3
        or not 0.5 <= train_fraction <= 0.9
        or min_approved < 5
        or min_rate_std <= 0
    ):
        raise ValueError(
            "Invalid share, train fraction, minimum count or rate dispersion"
        )
    clean = validate(df, features, controls)
    clean = clean.loc[clean["product"] == product].reset_index(drop=True)
    if len(clean) < 30:
        raise ValueError("Selected product needs at least 30 applications")
    cutoff = clean.date.iloc[min(int(len(clean) * train_fraction), len(clean) - 1)]
    train = clean.loc[clean.date < cutoff].copy()
    test = clean.loc[clean.date >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Need distinct dates for chronological train/test split")
    rules = candidate_rules(train, features)
    masks = [mask_rule(train, rule) for rule in rules]
    cache = {}

    def evaluate(mask):
        if mask.mean() < min_share:
            return None
        key = np.packbits(mask).tobytes()
        if key not in cache:
            cache[key] = estimate(train.loc[mask], controls, min_approved, min_rate_std)
        return cache[key]

    best = None
    best_score = -np.inf
    for root_idx, first in enumerate(masks):
        for split_left in (True, False):
            split = first if split_left else ~first
            fixed = evaluate(~split)
            if fixed is None:
                continue
            for child_idx, second in enumerate(masks):
                left = evaluate(split & second)
                right = evaluate(split & ~second)
                if left is None or right is None:
                    continue
                estimates = [fixed, left, right]
                score = sum(
                    (a["beta"] - b["beta"]) ** 2 for a, b in combinations(estimates, 2)
                )
                if score > best_score:
                    order = np.argsort([-e["beta"] for e in estimates])
                    labels = {
                        str(int(leaf)): label
                        for leaf, label in zip(order, ["S-", "S", "S+"])
                    }
                    best_score = score
                    best = {
                        "root": rules[root_idx],
                        "child": rules[child_idx],
                        "split_left": split_left,
                        "labels": labels,
                    }
    if best is None:
        raise ValueError(
            "No feasible three-segment partition; inspect counts, rate coverage and constraints"
        )
    best.update(
        version=1,
        product=product,
        features=features,
        controls=controls,
        cutoff=cutoff.isoformat(),
        min_share=min_share,
        min_approved=min_approved,
        min_rate_std=min_rate_std,
        objective=best_score,
        training_categories={f: sorted(train[f].unique()) for f in features},
    )
    records = []
    for name, frame in [("train", train), ("test", test)]:
        frame["segment"] = assign(frame, best)
        for label in ["S-", "S", "S+"]:
            group = frame.loc[frame.segment == label]
            approved = group.loc[group.approved == 1]
            effect = estimate(group, controls, min_approved, min_rate_std)
            records.append(
                dict(
                    dataset=name,
                    segment=label,
                    n=len(group),
                    share=len(group) / len(frame),
                    share_ok=len(group) / len(frame) >= min_share,
                    booking_rate=float(group.booked.mean()) if len(group) else None,
                    booking_given_approval=float(approved.booked.mean())
                    if len(approved)
                    else None,
                    gnos_per_application=float(group.gnos.mean())
                    if len(group)
                    else None,
                    rate_min=float(approved.rate_pp.min()) if len(approved) else None,
                    rate_max=float(approved.rate_pp.max()) if len(approved) else None,
                    estimate_ok=effect is not None,
                    **(effect or {}),
                )
            )
    clean["segment"] = assign(clean, best)
    clean["dataset"] = np.where(clean.date < cutoff, "train", "test")
    return best, pd.DataFrame(records), clean


def save_results(destination, model, report, assignments, input_hash):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    model["input_sha256"] = input_hash
    (path / "model.json").write_text(
        json.dumps(model, indent=2, allow_nan=False), encoding="utf-8"
    )
    report.to_csv(path / "report.csv", index=False)
    # Export identifiers only: never duplicate full application records by default.
    assignments[["application_id", "dataset", "segment"]].to_csv(
        path / "assignments.csv", index=False
    )
