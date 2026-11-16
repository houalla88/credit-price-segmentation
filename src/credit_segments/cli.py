"""Command line entry point."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import assign, run, save_results
from .reporting import diagnostics


def demo(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.choice(["18-34", "35-54", "55+"], n)
    rate = rng.uniform(3, 11, n)
    risk = rng.uniform(0, 1, n)
    slope = (
        pd.Series(age).map({"18-34": -0.09, "35-54": -0.05, "55+": -0.01}).to_numpy()
    )
    approved = rng.binomial(1, 0.85 - 0.2 * risk)
    probability = np.clip(0.65 + slope * (rate - 7) - 0.1 * risk, 0.02, 0.98)
    booked = approved * rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "application_id": [f"DEMO-{i:06}" for i in range(n)],
            "date": pd.date_range("2023-01-01", periods=n, freq="3h"),
            "product": "renovation",
            "approved": approved,
            "booked": booked,
            "rate_pp": rate,
            "gnos": booked * rng.uniform(5000, 25000, n),
            "risk_score": risk,
            "age_band": age,
            "employment": np.where(
                rng.random(n) < np.where(age == "55+", 0.7, 0.25),
                "senior_worker",
                "other_worker",
            ),
            "marital_status": rng.choice(["single", "couple"], n),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation tarifaire exploratoire, sans interface"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("demo")
    sample.add_argument("--output", default="demo.csv")
    sample.add_argument("--rows", type=int, default=6000)
    fit = commands.add_parser("fit")
    fit.add_argument("input")
    fit.add_argument("--output", required=True)
    fit.add_argument("--product", required=True)
    fit.add_argument("--features", nargs="+", required=True)
    fit.add_argument("--controls", nargs="*", default=[])
    fit.add_argument("--profile", nargs="*", default=[])
    fit.add_argument("--min-share", type=float, default=0.25)
    fit.add_argument("--min-approved", type=int, default=50)
    fit.add_argument("--min-rate-std", type=float, default=0.15)
    fit.add_argument("--train-fraction", type=float, default=0.7)
    predict = commands.add_parser("predict")
    predict.add_argument("input")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            if args.rows < 30:
                raise ValueError("At least 30 rows required")
            if Path(args.output).exists():
                raise ValueError("Output already exists")
            demo(args.rows).to_csv(args.output, index=False)
        elif args.command == "predict":
            model = json.loads(Path(args.model).read_text())
            frame = pd.read_csv(args.input)
            required = {"application_id", "product", *model["features"]}
            if not required <= set(frame):
                raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
            if not frame["product"].eq(model["product"]).all():
                raise ValueError("Prediction product does not match model")
            if (
                frame.application_id.isna().any()
                or frame.application_id.duplicated().any()
            ):
                raise ValueError("IDs must be present and unique")
            if Path(args.output).exists():
                raise ValueError("Output already exists")
            frame["segment"] = assign(frame, model)
            for feature in model["features"]:
                unseen = ~frame[feature].fillna("__MISSING__").astype(str).isin(
                    model["training_categories"][feature]
                )
                if unseen.any():
                    print(
                        f"Warning: {unseen.sum()} unseen categories in {feature}; routed by complement rules",
                        file=sys.stderr,
                    )
            frame[["application_id", "segment"]].to_csv(args.output, index=False)
        else:
            raw = Path(args.input).read_bytes()
            frame = pd.read_csv(args.input)
            columns = args.profile or args.features
            if not set(columns) <= set(frame):
                raise ValueError("Unknown profile columns")
            model, report, assignments = run(
                frame,
                args.features,
                args.controls,
                args.product,
                args.min_share,
                args.min_approved,
                args.min_rate_std,
                args.train_fraction,
            )
            save_results(
                args.output, model, report, assignments, hashlib.sha256(raw).hexdigest()
            )
            diagnostics(args.output, report, assignments, columns, args.min_share)
            print(report.to_string(index=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
