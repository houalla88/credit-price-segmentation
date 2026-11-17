import numpy as np
import pandas as pd
import pytest

from credit_segments import delta as ds


def config(**kw):
    base = dict(
        variables_sociodemo=["age", "statut_marital", "logement", "emploi"],
        periode_avant=("2024-01-01", "2024-06-30"),
        periode_apres=("2024-07-01", "2024-12-31"),
        decoupage_continu={"age": [18, 35, 55, 120]},
        colonne_taux="rate_pp",
        colonne_booked="booked",
    )
    base.update(kw)
    return ds.Config(**base)


@pytest.fixture(scope="module")
def resultat():
    cfg = config()
    data = ds.preparer(ds.demo_deux_periodes(12000), cfg)
    cellules, data, score = ds.segmenter(data, cfg)
    return cfg, cellules, data, score


def test_regles_respectees(resultat):
    cfg, cellules, data, _ = resultat
    resume = ds.synthese(data, cfg)
    assert set(data.segment) == set(ds.SEGMENTS)
    assert resume.drop("Total").part_ok.all() and resume.drop("Total").effectif_ok.all()
    # Hausse de prix : S+ baisse le plus, S- le moins.
    assert (
        resume.loc["S+", "delta"] < resume.loc["S", "delta"] < resume.loc["S-", "delta"]
    )
    # Chaque cellule appartient à un seul segment, et les segments sont contigus en delta.
    assert data.groupby("cellule").segment.nunique().max() == 1
    rang = cellules.segment.map({"S+": 0, "S": 1, "S-": 2}).to_numpy()
    assert (np.diff(rang) >= 0).all()


def test_recherche_exhaustive_optimale(resultat):
    """Comparaison avec une boucle naïve sur toutes les coupures."""
    cfg, cellules, _, score = resultat
    best = -np.inf
    k, total = len(cellules), cellules.n_total.sum()
    for i in range(1, k):
        for j in range(i + 1, k):
            blocs = [cellules.iloc[:i], cellules.iloc[i:j], cellules.iloc[j:]]
            stats = [
                b[["n_avant", "n_apres", "gnos_avant", "gnos_apres"]].sum()
                for b in blocs
            ]
            if any(
                (s.n_avant + s.n_apres) / total < cfg.part_min_segment
                or min(s.n_avant, s.n_apres) < cfg.effectif_min_segment_par_periode
                for s in stats
            ):
                continue
            d = np.array([(s.gnos_apres / s.gnos_avant - 1) * 100 for s in stats])
            w = np.array([(s.n_avant + s.n_apres) / total for s in stats])
            m = (w * d).sum() / w.sum()
            best = max(best, (w * (d - m) ** 2).sum())
    assert score == pytest.approx(best)


def test_sens_baisse_inverse_les_labels(resultat):
    cfg, _, data, _ = resultat
    cfg_b = config(sens_prix="baisse")
    _, data_b, _ = ds.segmenter(data.drop(columns="segment"), cfg_b)
    inverse = data.segment.map({"S+": "S-", "S": "S", "S-": "S+"})
    assert (data_b.segment == inverse).all()


def test_permutation_detecte_signal_et_absence_de_signal(resultat):
    cfg, _, data, _ = resultat
    _, _, p = ds.test_permutation(data, cfg, n=19, graine=1)
    assert p <= 0.05
    # Placebo : premier semestre, aucun changement de prix.
    cfg_p = config(
        periode_avant=("2024-01-01", "2024-03-31"),
        periode_apres=("2024-04-01", "2024-06-30"),
        part_min_segment=0.2,
        effectif_min_segment_par_periode=100,
    )
    data_p = ds.preparer(ds.demo_deux_periodes(12000), cfg_p)
    _, _, p_placebo = ds.test_permutation(data_p, cfg_p, n=19, graine=1)
    assert p_placebo > 0.05


def test_profils_bootstrap_et_export(resultat, tmp_path):
    cfg, cellules, data, _ = resultat
    table = ds.profils(data, cfg)
    assert np.allclose(table.groupby(["variable", "segment"]).part_segment.sum(), 1)
    ic = ds.bootstrap(data, cfg, n=50)
    assert (ic.ic_bas <= ic.ic_haut).all()
    texte = ds.texte_pour_llm(cellules, ds.synthese(data, cfg), table, cfg)
    assert "# Segment S+" in texte and "DEMO-" not in texte
    dossier = ds.sauvegarder(
        tmp_path / "out", cfg, cellules, ds.synthese(data, cfg), ic, table, texte
    )
    assert (dossier / "description_pour_llm.md").stat().st_size > 0
    with pytest.raises(FileExistsError):
        ds.sauvegarder(dossier, cfg, cellules, ds.synthese(data, cfg), ic, table, texte)


def test_petites_cellules_et_manquants():
    cfg = config(effectif_min_cellule_par_periode=10_000)
    df = ds.demo_deux_periodes(6000)
    df.loc[:50, "statut_marital"] = None
    data = ds.preparer(df, cfg)
    assert (data.statut_marital == ds.MANQUANT).sum() > 0
    cellules = ds.table_cellules(data, cfg)
    assert (cellules.niveau_repli == "population totale").all()


@pytest.mark.parametrize(
    "kw",
    [
        dict(variables_sociodemo=[]),
        dict(metrique="gnos_total", delta="absolu"),
        dict(part_min_segment=0.5),
        dict(periode_apres=("2024-06-01", "2024-12-31")),
        dict(sens_prix="stable"),
    ],
)
def test_parametres_invalides(kw):
    with pytest.raises(ValueError):
        config(**kw).verifier()


def test_colonnes_absentes():
    with pytest.raises(ValueError, match="absentes"):
        ds.preparer(pd.DataFrame({"date": ["2024-01-01"], "gnos": [0]}), config())
