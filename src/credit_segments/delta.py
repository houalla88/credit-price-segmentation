"""Segmentation simplifiée par variation du GNoS entre deux périodes.

Approche volontairement descriptive, sans modèle statistique :

1. On croise les variables sociodémographiques choisies : chaque combinaison de
   modalités forme une « cellule » (ex. 35-54 | couple | salarié).
2. Pour chaque cellule, on calcule le delta de GNoS entre la période AVANT et la
   période APRÈS le changement de prix.
3. On trie les cellules par delta et on cherche, de façon exhaustive, les deux
   points de coupure qui forment trois segments :
   - règle 1 : maximiser l'écart entre les deltas des trois segments ;
   - règle 2 : chaque segment doit être suffisamment alimenté.
4. Les segments sont nommés S+ (plus sensible), S, S- (moins sensible).

Aucune ligne individuelle n'est exportée : seules des agrégations le sont.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ["S+", "S", "S-"]
MANQUANT = "(manquant)"
SEPARATEUR = " | "


@dataclass
class Config:
    """Paramètres de l'analyse. Chaque champ est documenté dans le notebook."""

    variables_sociodemo: list[str]
    periode_avant: tuple[str, str]
    periode_apres: tuple[str, str]
    colonne_date: str = "date"
    colonne_gnos: str = "gnos"
    colonne_taux: str | None = None
    colonne_booked: str | None = None
    decoupage_continu: dict[str, list[float]] = field(default_factory=dict)
    metrique: str = "gnos_total"  # "gnos_total" ou "gnos_par_demande"
    delta: str = "relatif"  # "relatif" (%) ou "absolu"
    sens_prix: str = "hausse"  # "hausse" ou "baisse"
    part_min_segment: float = 0.20
    effectif_min_segment_par_periode: int = 300
    effectif_min_cellule_par_periode: int = 30

    def verifier(self):
        if not self.variables_sociodemo:
            raise ValueError("VARIABLES_SOCIODEMO ne peut pas être vide")
        if len(set(self.variables_sociodemo)) != len(self.variables_sociodemo):
            raise ValueError("VARIABLES_SOCIODEMO contient des doublons")
        if self.metrique not in {"gnos_total", "gnos_par_demande"}:
            raise ValueError("METRIQUE doit valoir 'gnos_total' ou 'gnos_par_demande'")
        if self.delta not in {"relatif", "absolu"}:
            raise ValueError("TYPE_DELTA doit valoir 'relatif' ou 'absolu'")
        if self.metrique == "gnos_total" and self.delta == "absolu":
            raise ValueError(
                "Un delta absolu sur le GNoS total dépend de la taille du groupe : "
                "utiliser TYPE_DELTA='relatif' ou METRIQUE='gnos_par_demande'"
            )
        if self.sens_prix not in {"hausse", "baisse"}:
            raise ValueError("SENS_CHANGEMENT_PRIX doit valoir 'hausse' ou 'baisse'")
        if not 0 < self.part_min_segment <= 1 / 3:
            raise ValueError("PART_MIN_PAR_SEGMENT doit être dans ]0 ; 1/3]")
        if self.effectif_min_segment_par_periode < 1:
            raise ValueError("EFFECTIF_MIN_SEGMENT_PAR_PERIODE doit être positif")
        if self.effectif_min_cellule_par_periode < 1:
            raise ValueError("EFFECTIF_MIN_CELLULE_PAR_PERIODE doit être positif")
        a0, a1 = map(pd.Timestamp, self.periode_avant)
        b0, b1 = map(pd.Timestamp, self.periode_apres)
        if not (a0 <= a1 and b0 <= b1):
            raise ValueError("Chaque période doit avoir début <= fin")
        if a1 >= b0 and b1 >= a0:
            raise ValueError("Les deux périodes se chevauchent")


# --------------------------------------------------------------------------- #
# Préparation
# --------------------------------------------------------------------------- #
def preparer(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Valide, discrétise, affecte chaque ligne à une période et à une cellule."""
    cfg.verifier()
    requises = {cfg.colonne_date, cfg.colonne_gnos, *cfg.variables_sociodemo}
    requises |= {c for c in (cfg.colonne_taux, cfg.colonne_booked) if c}
    absentes = requises - set(df.columns)
    if absentes:
        raise ValueError(f"Colonnes absentes : {sorted(absentes)}")
    data = df.copy()
    data["_date"] = pd.to_datetime(data[cfg.colonne_date], errors="raise")
    if data["_date"].dt.tz is not None:  # dates avec fuseau : comparaison en heure locale
        data["_date"] = data["_date"].dt.tz_localize(None)
    data["_gnos"] = pd.to_numeric(data[cfg.colonne_gnos], errors="raise")
    if not np.isfinite(data["_gnos"]).all() or (data["_gnos"] < 0).any():
        raise ValueError("Le GNoS doit être numérique, fini et non négatif")

    debut_fin = {
        "avant": tuple(map(pd.Timestamp, cfg.periode_avant)),
        "apres": tuple(map(pd.Timestamp, cfg.periode_apres)),
    }
    data["periode"] = None
    for nom, (debut, fin) in debut_fin.items():
        # Comparaison au jour près : début et fin inclus (fin incluse jusqu'à 23:59).
        jour = data["_date"].dt.normalize()
        dedans = (jour >= debut.normalize()) & (jour <= fin.normalize())
        data.loc[dedans, "periode"] = nom
    data = data.loc[data["periode"].notna()].copy()
    for nom in debut_fin:
        if (data["periode"] == nom).sum() == 0:
            raise ValueError(f"Aucune ligne dans la période {nom}")

    for variable in cfg.variables_sociodemo:
        if variable in cfg.decoupage_continu:
            bornes = cfg.decoupage_continu[variable]
            valeurs = pd.to_numeric(data[variable], errors="raise")
            data[variable] = pd.cut(valeurs, bornes, right=False).astype(str)
            data.loc[valeurs.isna(), variable] = MANQUANT
        else:
            data[variable] = (
                data[variable].astype("string").fillna(MANQUANT).astype(str)
            )
        if data[variable].nunique() > 15:
            raise ValueError(
                f"{variable} a plus de 15 modalités : regrouper ou fournir des bornes "
                "dans DECOUPAGE_VARIABLES_CONTINUES"
            )
    data["cellule"] = data[cfg.variables_sociodemo].agg(SEPARATEUR.join, axis=1)
    return data


# --------------------------------------------------------------------------- #
# Calcul des deltas
# --------------------------------------------------------------------------- #
def _delta(n_a, s_a, n_b, s_b, cfg: Config):
    """Delta vectorisé à partir des effectifs (n) et sommes de GNoS (s)."""
    n_a, s_a, n_b, s_b = (np.asarray(x, dtype=float) for x in (n_a, s_a, n_b, s_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        if cfg.metrique == "gnos_total":
            a, b = s_a, s_b
        else:
            a, b = s_a / n_a, s_b / n_b
        return (b / a - 1) * 100 if cfg.delta == "relatif" else b - a


def unite_delta(cfg: Config) -> str:
    return "%" if cfg.delta == "relatif" else "unités de GNoS par demande"


def stats_par_groupe(data: pd.DataFrame, cle) -> pd.DataFrame:
    """Effectifs et sommes de GNoS par groupe et par période."""
    table = data.groupby([cle, "periode"], observed=True)["_gnos"].agg(["size", "sum"])
    table = table.unstack("periode").fillna(0)
    table.columns = [f"{'n' if m == 'size' else 'gnos'}_{p}" for m, p in table.columns]
    for col in ["n_avant", "n_apres", "gnos_avant", "gnos_apres"]:
        if col not in table:
            table[col] = 0.0
    return table[["n_avant", "n_apres", "gnos_avant", "gnos_apres"]]


def table_cellules(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Delta par cellule, avec repli hiérarchique pour les petites cellules.

    Une cellule trop petite (moins de EFFECTIF_MIN_CELLULE_PAR_PERIODE lignes dans
    l'une des périodes) reçoit, pour le tri uniquement, le delta de la cellule
    plus large qui la contient : on retire d'abord la DERNIÈRE variable de la
    liste, puis l'avant-dernière, etc. L'ordre de VARIABLES_SOCIODEMO compte.
    """
    variables = cfg.variables_sociodemo
    seuil = cfg.effectif_min_cellule_par_periode
    cellules = stats_par_groupe(data, "cellule")
    cellules["delta_brut"] = _delta(
        cellules.n_avant,
        cellules.gnos_avant,
        cellules.n_apres,
        cellules.gnos_apres,
        cfg,
    )
    niveaux = pd.DataFrame(
        [c.split(SEPARATEUR) for c in cellules.index],
        index=cellules.index,
        columns=variables,
    )
    cellules = pd.concat([niveaux, cellules], axis=1)
    cellules["delta_tri"] = np.nan
    cellules["niveau_repli"] = ""

    a_traiter = pd.Series(True, index=cellules.index)
    for k in range(len(variables), -1, -1):
        cles = variables[:k]
        if k == 0:
            stats = stats_par_groupe(data.assign(_tout="tout"), "_tout")
            ligne = stats.iloc[0]
            delta = _delta(
                ligne.n_avant, ligne.gnos_avant, ligne.n_apres, ligne.gnos_apres, cfg
            )
            cellules.loc[a_traiter, "delta_tri"] = float(delta)
            cellules.loc[a_traiter, "niveau_repli"] = "population totale"
            break
        cle = data[cles].agg(SEPARATEUR.join, axis=1) if k > 1 else data[cles[0]]
        stats = stats_par_groupe(data.assign(_cle=cle), "_cle")
        base_requise = not (
            cfg.metrique == "gnos_par_demande" and cfg.delta == "absolu"
        )
        ok = (stats.n_avant >= seuil) & (stats.n_apres >= seuil)
        if base_requise:  # un delta relatif exige un GNoS AVANT strictement positif
            ok &= stats.gnos_avant > 0
        deltas = pd.Series(
            _delta(
                stats.n_avant, stats.gnos_avant, stats.n_apres, stats.gnos_apres, cfg
            ),
            index=stats.index,
        )
        cle_cellule = niveaux[cles].agg(SEPARATEUR.join, axis=1)
        utilisable = a_traiter & cle_cellule.map(ok).fillna(False).astype(bool)
        cellules.loc[utilisable, "delta_tri"] = cle_cellule[utilisable].map(deltas)
        cellules.loc[utilisable, "niveau_repli"] = (
            "cellule complète"
            if k == len(variables)
            else "repli sur " + ", ".join(cles)
        )
        a_traiter &= ~utilisable
        if not a_traiter.any():
            break
    cellules["n_total"] = cellules.n_avant + cellules.n_apres
    return cellules.sort_values(["delta_tri", "n_total"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# Recherche des deux coupures
# --------------------------------------------------------------------------- #
def chercher_coupures(cellules: pd.DataFrame, cfg: Config):
    """Recherche exhaustive des coupures (i, j) sur les cellules triées par delta.

    Segments contigus [0, i), [i, j), [j, K). Score = variance inter-segments
    pondérée par les effectifs : somme_g w_g (delta_g - moyenne)^2. C'est le
    critère des « seuils naturels » de Fisher (1958) / Jenks, ici sous contrainte
    de taille minimale. Retourne (i, j, score) ou None si aucune partition n'est
    admissible.
    """
    k = len(cellules)
    if k < 3:
        return None
    cum = {
        c: np.concatenate([[0.0], np.cumsum(cellules[c].to_numpy(float))])
        for c in ["n_avant", "n_apres", "gnos_avant", "gnos_apres"]
    }
    total = cum["n_avant"][-1] + cum["n_apres"][-1]
    i, j = np.meshgrid(np.arange(1, k), np.arange(1, k), indexing="ij")
    valide = j > i
    bornes = [(np.zeros_like(i), i), (i, j), (j, np.full_like(i, k))]
    deltas, poids, admissible = [], [], valide.copy()
    for debut, fin in bornes:
        g = {c: cum[c][fin] - cum[c][debut] for c in cum}
        d = _delta(g["n_avant"], g["gnos_avant"], g["n_apres"], g["gnos_apres"], cfg)
        w = (g["n_avant"] + g["n_apres"]) / total
        admissible &= (
            (w >= cfg.part_min_segment)
            & (g["n_avant"] >= cfg.effectif_min_segment_par_periode)
            & (g["n_apres"] >= cfg.effectif_min_segment_par_periode)
            & np.isfinite(d)
        )
        deltas.append(d)
        poids.append(w)
    deltas, poids = np.stack(deltas), np.stack(poids)
    with np.errstate(invalid="ignore"):
        moyenne = (poids * deltas).sum(0) / poids.sum(0)
        score = (poids * (deltas - moyenne) ** 2).sum(0)
    score = np.where(admissible, score, -np.inf)
    if not np.isfinite(score).any():
        return None
    best = np.unravel_index(np.argmax(score), score.shape)
    return int(i[best]), int(j[best]), float(score[best])


def _etiquettes(cfg: Config):
    # Cellules triées du delta le plus bas au plus haut.
    # Hausse de prix : le plus sensible est celui dont le GNoS baisse le plus.
    return ["S+", "S", "S-"] if cfg.sens_prix == "hausse" else ["S-", "S", "S+"]


def segmenter(data: pd.DataFrame, cfg: Config):
    """Pipeline complet : cellules -> coupures -> segment par cellule et par ligne."""
    cellules = table_cellules(data, cfg)
    resultat = chercher_coupures(cellules, cfg)
    if resultat is None:
        raise ValueError(
            "Aucune partition en 3 segments ne respecte les tailles minimales : "
            "réduire PART_MIN_PAR_SEGMENT / EFFECTIF_MIN_SEGMENT_PAR_PERIODE, "
            "ou retirer des variables sociodémographiques"
        )
    i, j, score = resultat
    rang = np.arange(len(cellules))
    etiquettes = _etiquettes(cfg)
    cellules["segment"] = np.select([rang < i, rang < j], etiquettes[:2], etiquettes[2])
    data = data.copy()
    data["segment"] = data["cellule"].map(cellules["segment"])
    return cellules, data, score


# --------------------------------------------------------------------------- #
# Synthèse, robustesse, profils
# --------------------------------------------------------------------------- #
def synthese(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Indicateurs avant / après par segment, plus la population totale."""
    lignes = []
    groupes = [(s, data.loc[data.segment == s]) for s in SEGMENTS] + [("Total", data)]
    total = len(data)
    for nom, g in groupes:
        av, ap = g.loc[g.periode == "avant"], g.loc[g.periode == "apres"]
        ligne = {
            "segment": nom,
            "n_avant": len(av),
            "n_apres": len(ap),
            "part_population": len(g) / total,
            "gnos_avant": av._gnos.sum(),
            "gnos_apres": ap._gnos.sum(),
            "gnos_par_demande_avant": av._gnos.mean(),
            "gnos_par_demande_apres": ap._gnos.mean(),
            "delta": float(
                _delta(len(av), av._gnos.sum(), len(ap), ap._gnos.sum(), cfg)
            ),
        }
        if cfg.colonne_taux:
            ligne["taux_moyen_avant"] = pd.to_numeric(av[cfg.colonne_taux]).mean()
            ligne["taux_moyen_apres"] = pd.to_numeric(ap[cfg.colonne_taux]).mean()
            ligne["delta_taux_pp"] = (
                ligne["taux_moyen_apres"] - ligne["taux_moyen_avant"]
            )
        if cfg.colonne_booked:
            conv_a = pd.to_numeric(av[cfg.colonne_booked]).mean()
            conv_b = pd.to_numeric(ap[cfg.colonne_booked]).mean()
            ligne["conversion_avant"], ligne["conversion_apres"] = conv_a, conv_b
            booked_a = av.loc[pd.to_numeric(av[cfg.colonne_booked]) == 1, "_gnos"]
            booked_b = ap.loc[pd.to_numeric(ap[cfg.colonne_booked]) == 1, "_gnos"]
            ligne["ticket_moyen_avant"] = booked_a.mean()
            ligne["ticket_moyen_apres"] = booked_b.mean()
        ligne["part_ok"] = (
            nom == "Total" or ligne["part_population"] >= cfg.part_min_segment
        )
        ligne["effectif_ok"] = nom == "Total" or (
            min(ligne["n_avant"], ligne["n_apres"])
            >= cfg.effectif_min_segment_par_periode
        )
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("segment")


def bootstrap(data: pd.DataFrame, cfg: Config, n=300, graine=42) -> pd.DataFrame:
    """IC 95 % des deltas par segment et des écarts entre segments.

    Rééchantillonnage des lignes avec remise, séparément dans chaque période,
    segments FIGÉS. Les IC sont optimistes : ils ignorent que les segments ont
    été choisis sur ces mêmes données (biais de sélection).
    """
    rng = np.random.default_rng(graine)
    seg_codes = pd.Categorical(data.segment, categories=SEGMENTS).codes
    tirages = np.empty((n, 3))
    stats = {}
    for p in ["avant", "apres"]:
        masque = (data.periode == p).to_numpy()
        stats[p] = (seg_codes[masque], data._gnos.to_numpy()[masque])
    for b in range(n):
        res = {}
        for p, (codes, gnos) in stats.items():
            idx = rng.integers(0, len(codes), len(codes))
            res[f"n_{p}"] = np.bincount(codes[idx], minlength=3)
            res[f"s_{p}"] = np.bincount(codes[idx], weights=gnos[idx], minlength=3)
        tirages[b] = _delta(
            res["n_avant"], res["s_avant"], res["n_apres"], res["s_apres"], cfg
        )
    tirages = pd.DataFrame(tirages, columns=SEGMENTS)
    tirages["S- moins S+"] = tirages["S-"] - tirages["S+"]
    tirages["S moins S+"] = tirages["S"] - tirages["S+"]
    tirages["S- moins S"] = tirages["S-"] - tirages["S"]
    return tirages.quantile([0.025, 0.5, 0.975]).T.set_axis(
        ["ic_bas", "mediane", "ic_haut"], axis=1
    )


def test_permutation(data: pd.DataFrame, cfg: Config, n=200, graine=42):
    """Le score observé est-il supérieur à ce que le hasard produirait ?

    Hypothèse nulle : au sein de chaque période, le GNoS ne dépend pas du profil
    sociodémographique. On mélange les profils entre les lignes d'une même
    période (le mix de chaque période est conservé), puis on relance toute la
    recherche. p-valeur = part des scores simulés >= score observé.
    """
    rng = np.random.default_rng(graine)
    _, _, observe = segmenter(data, cfg)
    scores = []
    profil = data[cfg.variables_sociodemo + ["cellule"]].to_numpy()
    positions = {
        p: np.flatnonzero((data.periode == p).to_numpy()) for p in ["avant", "apres"]
    }
    for _ in range(n):
        melange = profil.copy()
        for pos in positions.values():
            melange[pos] = profil[rng.permutation(pos)]
        simule = data.copy()
        simule[cfg.variables_sociodemo + ["cellule"]] = melange
        res = chercher_coupures(table_cellules(simule, cfg), cfg)
        scores.append(res[2] if res else 0.0)
    scores = np.array(scores)
    p_valeur = (1 + (scores >= observe).sum()) / (n + 1)
    return observe, scores, p_valeur


def profils(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Composition marginale de chaque segment, modalité par modalité."""
    lignes = []
    for variable in cfg.variables_sociodemo:
        pop = data[variable].value_counts(normalize=True)
        for segment in SEGMENTS:
            g = data.loc[data.segment == segment, variable]
            parts = g.value_counts(normalize=True)
            comptes = g.value_counts()
            for modalite, part_pop in pop.items():
                part = float(parts.get(modalite, 0.0))
                lignes.append(
                    {
                        "variable": variable,
                        "modalite": modalite,
                        "segment": segment,
                        "effectif": int(comptes.get(modalite, 0)),
                        "part_segment": part,
                        "part_population": float(part_pop),
                        "indice": part / part_pop,
                        "ecart_pp": 100 * (part - part_pop),
                    }
                )
    return pd.DataFrame(lignes)


def texte_pour_llm(
    cellules, resume, table_profils, cfg: Config, max_cellules=25
) -> str:
    """Description agrégée des segments, prête à coller dans un LLM."""
    unite = unite_delta(cfg)
    total = resume.loc["Total"]
    lignes = [
        "# Contexte",
        "",
        "Crédit à la consommation. Changement de prix ("
        + cfg.sens_prix
        + " des taux) entre "
        f"la période AVANT ({cfg.periode_avant[0]} → {cfg.periode_avant[1]}) et la période "
        f"APRÈS ({cfg.periode_apres[0]} → {cfg.periode_apres[1]}).",
        f"Indicateur : {'GNoS total' if cfg.metrique == 'gnos_total' else 'GNoS par demande'}, "
        f"delta {cfg.delta} ({unite}). Delta de la population totale : {total.delta:+.1f} {unite}.",
        "Trois segments ont été formés à partir des variables : "
        + ", ".join(cfg.variables_sociodemo)
        + ". S+ = le plus sensible au prix, S = intermédiaire, S- = le moins sensible.",
        "",
    ]
    for segment in SEGMENTS:
        r = resume.loc[segment]
        lignes += [
            f"# Segment {segment}",
            "",
            f"- Poids : {r.part_population:.1%} des demandes ; {int(r.n_avant)} avant, {int(r.n_apres)} après.",
            f"- Delta : {r.delta:+.1f} {unite} (écart à la population : {r.delta - total.delta:+.1f}).",
        ]
        if "delta_taux_pp" in r:
            lignes.append(f"- Variation du taux moyen : {r.delta_taux_pp:+.2f} point.")
        lignes += [
            "",
            "Modalités par variable (part dans le segment vs population, indice) :",
        ]
        sous = table_profils.loc[table_profils.segment == segment]
        for variable in cfg.variables_sociodemo:
            v = sous.loc[sous.variable == variable].sort_values(
                "indice", ascending=False
            )
            morceaux = [
                f"{m.modalite} {m.part_segment:.0%} vs {m.part_population:.0%} (×{m.indice:.2f})"
                for m in v.itertuples()
                if m.effectif > 0
            ]
            lignes.append(f"- {variable} : " + " ; ".join(morceaux))
        cel = cellules.loc[cellules.segment == segment].sort_values(
            "n_total", ascending=False
        )
        lignes += [
            "",
            f"Combinaisons exactes de modalités ({len(cel)} au total, "
            f"les {min(max_cellules, len(cel))} plus grandes ; ordre : {', '.join(cfg.variables_sociodemo)}) :",
        ]
        for nom, c in cel.head(max_cellules).iterrows():
            lignes.append(
                f"- {nom} — {int(c.n_total)} demandes, delta {c.delta_tri:+.1f}"
            )
        lignes.append("")
    lignes += [
        "# Demande",
        "",
        "À partir de ces chiffres agrégés uniquement, décris en quelques phrases le profil type de "
        "chaque segment (S+, S, S-) : qui sont ces clients, quels traits les distinguent de la "
        "population, et des hypothèses plausibles sur leur sensibilité au prix. Distingue ce qui est "
        "observé (chiffres) de ce qui est interprété. Signale les traits peu marqués (indice proche "
        "de 1). Les descriptions sont marginales : deux traits surreprésentés ne sont pas forcément "
        "portés par les mêmes personnes.",
    ]
    return "\n".join(lignes)


def sauvegarder(dossier, cfg, cellules, resume, ic, table_profils, texte):
    """Exporte uniquement des agrégats. Refuse d'écraser un dossier existant."""
    chemin = Path(dossier)
    chemin.mkdir(parents=True, exist_ok=False)
    (chemin / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    cellules.to_csv(chemin / "cellules_et_segments.csv")
    resume.to_csv(chemin / "synthese_segments.csv")
    ic.to_csv(chemin / "intervalles_bootstrap.csv")
    table_profils.to_csv(chemin / "profils_segments.csv", index=False)
    (chemin / "description_pour_llm.md").write_text(texte, encoding="utf-8")
    return chemin


# --------------------------------------------------------------------------- #
# Données de démonstration
# --------------------------------------------------------------------------- #
def demo_deux_periodes(n=20000, graine=7) -> pd.DataFrame:
    """Données synthétiques : hausse de taux de +1 point au 1er juillet 2024.

    Les jeunes célibataires locataires réagissent fortement, les seniors
    propriétaires très peu. Une saisonnalité commune touche tout le monde.
    """
    rng = np.random.default_rng(graine)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 366, n), unit="D"
    )
    apres = np.asarray(dates >= pd.Timestamp("2024-07-01"))
    age = rng.integers(18, 80, n)
    statut = rng.choice(["celibataire", "couple", "divorce"], n, p=[0.4, 0.45, 0.15])
    logement = np.where(
        rng.random(n) < np.clip((age - 20) / 60, 0.1, 0.8), "proprietaire", "locataire"
    )
    emploi = rng.choice(["salarie", "independant", "retraite"], n, p=[0.7, 0.15, 0.15])
    emploi = np.where(
        age >= 65, "retraite", np.where(emploi == "retraite", "salarie", emploi)
    )
    sensibilite = (
        0.10
        + 0.10 * (age < 35)
        + 0.06 * (statut == "celibataire")
        + 0.06 * (logement == "locataire")
        - 0.08 * (age >= 55)
    )
    taux = 6 + rng.normal(0, 0.8, n) + apres * 1.0
    p_booking = np.clip(0.45 - apres * sensibilite * 0.7 - 0.03 * apres, 0.02, 0.95)
    booked = rng.binomial(1, p_booking)
    montant = rng.lognormal(np.log(12000), 0.4, n) * (1 - 0.05 * apres)
    return pd.DataFrame(
        {
            "application_id": [f"DEMO-{i:06}" for i in range(n)],
            "date": dates,
            "product": "renovation",
            "age": age,
            "statut_marital": statut,
            "logement": logement,
            "emploi": emploi,
            "rate_pp": taux.round(2),
            "booked": booked,
            "gnos": (booked * montant).round(0),
        }
    ).sort_values("date", ignore_index=True)
