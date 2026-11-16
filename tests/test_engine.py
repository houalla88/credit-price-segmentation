import numpy as np
import pandas as pd
import pytest

from credit_segments.cli import demo
from credit_segments.engine import assign, run
from credit_segments.reporting import diagnostics, profiles


@pytest.fixture(scope="module")
def result():
    return run(demo(3000), ["age_band"], ["risk_score"], "renovation")


def test_constraints_and_temporal_separation(result):
    model, report, frame = result
    assert set(frame.segment) == {"S-", "S", "S+"}
    assert report.loc[report.dataset == "train", "share_ok"].all()
    assert (
        frame.loc[frame.dataset == "train", "date"].max()
        < frame.loc[frame.dataset == "test", "date"].min()
    )
    assert np.array_equal(assign(frame, model), frame.segment)
    estimates = report.loc[report.dataset == "test"].set_index("segment").beta
    assert estimates["S-"] > estimates["S"] > estimates["S+"]


def test_profiles_and_artifacts(result, tmp_path):
    _model, report, frame = result
    table = profiles(frame, ["age_band", "employment"])
    assert np.allclose(table.groupby(["feature", "segment"])["share"].sum(), 1)
    diagnostics(tmp_path, report, frame, ["age_band", "employment"], 0.25)
    for file in [
        "quality.png",
        "profiles.png",
        "economics.png",
        "profiles.csv",
        "profiles.md",
    ]:
        assert (tmp_path / file).stat().st_size > 0


def test_invalid_booking_rejected():
    data = demo(200)
    data.loc[0, ["approved", "booked"]] = [0, 1]
    with pytest.raises(ValueError, match="Booking requires"):
        run(data, ["age_band"], [], "renovation")


def test_no_price_variation_rejected():
    data = demo(500)
    data.rate_pp = 7.0
    with pytest.raises(ValueError, match="No feasible"):
        run(data, ["age_band"], [], "renovation")


def test_missing_categories_supported():
    data = demo(3000)
    data.loc[data.age_band == "55+", "age_band"] = None
    model, _, frame = run(data, ["age_band"], [], "renovation")
    assert "__MISSING__" in model["training_categories"]["age_band"]
    assert pd.notna(frame.segment).all()


def test_leakage_rejected():
    with pytest.raises(ValueError, match="reserved"):
        run(demo(200), ["booked"], [], "renovation")


def test_holdout_cannot_change_rules(result):
    model, _, _frame = result
    altered = demo(3000)
    cutoff = pd.Timestamp(model["cutoff"])
    mask = pd.to_datetime(altered.date, utc=True) >= cutoff
    altered.loc[mask, "booked"] = 0
    altered.loc[mask, "gnos"] = 0
    updated, _, _ = run(altered, ["age_band"], ["risk_score"], "renovation")
    assert model == updated
