"""Aggregate profiles and reproducible static diagnostics (no user interface)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["S-", "S", "S+"]


def profiles(frame, columns):
    rows = []
    for column in columns:
        values = frame[column].fillna("__MISSING__").astype(str)
        for category in sorted(values.unique()):
            overall = float((values == category).mean())
            for segment in LABELS:
                selected = frame.segment == segment
                n = int(selected.sum())
                count = int(((values == category) & selected).sum())
                share = count / n if n else np.nan
                rows.append(
                    {
                        "feature": column,
                        "category": category,
                        "segment": segment,
                        "count": count,
                        "n": n,
                        "share": share,
                        "population_share": overall,
                        "lift": share / overall,
                        "difference_pp": 100 * (share - overall),
                    }
                )
    return pd.DataFrame(rows)


def diagnostics(destination, report, frame, columns, min_share):
    path = Path(destination)
    table = profiles(frame.loc[frame.dataset == "test"], columns)
    table.to_csv(path / "profiles.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for index, dataset in enumerate(["train", "test"]):
        data = (
            report.loc[report.dataset == dataset].set_index("segment").reindex(LABELS)
        )
        position = np.arange(3) + (index - 0.5) * 0.18
        if "beta" in data:
            axes[0].errorbar(
                position,
                100 * data.beta,
                yerr=100
                * np.array([data.beta - data.ci_low, data.ci_high - data.beta]),
                fmt="o",
                capsize=4,
                label=dataset,
            )
        axes[1].bar(position, data.share * 100, width=0.18, label=dataset)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set(
        ylabel="Variation du booking (points) / +1 point de taux",
        title="Association ajustée et IC 95 % HC3",
    )
    axes[1].axhline(min_share * 100, color="red", linestyle="--", label="Minimum")
    axes[1].set(
        ylabel="Part des demandes (%)", title="Représentativité et stabilité temporelle"
    )
    for ax in axes:
        ax.set_xticks(range(3), LABELS)
        ax.legend()
    fig.savefig(path / "quality.png", dpi=160)
    plt.close(fig)
    if not table.empty:
        pivot = table.pivot(
            index=["feature", "category"], columns="segment", values="lift"
        ).reindex(columns=LABELS)
        fig, ax = plt.subplots(
            figsize=(7, max(4, len(pivot) * 0.28)), layout="constrained"
        )
        image = ax.imshow(
            pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=0, vmax=2
        )
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(len(pivot)), [f"{a} : {b}" for a, b in pivot.index])
        ax.set_title("Profil sur validation : indice de surreprésentation")
        fig.colorbar(
            image, ax=ax, label="Part dans le segment / part dans la population"
        )
        fig.savefig(path / "profiles.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    test = frame.loc[frame.dataset == "test"]
    for label in LABELS:
        group = test.loc[(test.segment == label) & (test.approved == 1)]
        if len(group):
            axes[0].hist(
                group.rate_pp,
                bins=np.linspace(test.rate_pp.min(), test.rate_pp.max(), 15),
                alpha=0.4,
                label=label,
            )
    axes[0].set(
        title="Couverture des taux sur validation",
        xlabel="Taux proposé (%)",
        ylabel="Offres approuvées",
    )
    axes[0].legend()
    data = report.loc[report.dataset == "test"].set_index("segment").reindex(LABELS)
    axes[1].bar(LABELS, data.gnos_per_application)
    axes[1].set(
        title="Production sur validation", ylabel="GNoS par demande (unité du fichier)"
    )
    fig.savefig(path / "economics.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Profils des segments — validation temporelle",
        "",
        "Descriptions marginales : ces proportions ne prouvent pas que les catégories sont présentes chez les mêmes personnes.",
        "",
    ]
    for label in LABELS:
        lines.extend([f"## {label}", ""])
        top = table.loc[
            (table.segment == label)
            & (table["count"] >= 10)
            & (table.difference_pp >= 5)
        ]
        top = top.sort_values("difference_pp", ascending=False).head(5)
        if top.empty:
            lines.append(
                "Aucune surreprésentation répondant aux seuils descriptifs (10 observations et +5 points)."
            )
        for row in top.itertuples():
            lines.append(
                f"- {row.feature} = {row.category} : {row.share:.1%}, contre {row.population_share:.1%} dans la population ; indice {row.lift:.2f}."
            )
        lines.append("")
    lines.extend(
        [
            "Les seuils sont descriptifs, sans test de significativité après sélection.",
            "Les intervalles du jeu train sont exploratoires ; ceux du jeu test sont conditionnels aux règles figées.",
            "Vérifier estimate_ok, share_ok et le maintien de l’ordre des coefficients dans report.csv.",
        ]
    )
    (path / "profiles.md").write_text("\n".join(lines), encoding="utf-8")
