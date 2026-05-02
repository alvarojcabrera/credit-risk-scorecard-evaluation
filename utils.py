"""
utils.py
========
Funciones de soporte para los notebooks de la prueba tecnica.

Contiene metricas estandar de credit scoring:
- KS (Kolmogorov-Smirnov)
- Gini / AUC
- PSI (Population Stability Index)
- IV / WoE (Information Value y Weight of Evidence)
- Tabla decilica (gains table) con tasa de malos por banda

Convencion: y=1 corresponde a "malo" (incumplimiento). Los scores se interpretan
de forma que MAYOR score = MAYOR probabilidad de incumplimiento (PD).
Cuando un proveedor entregue el score con sentido inverso, se invierte previo
al calculo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Metricas de discriminacion
# ---------------------------------------------------------------------------

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS = max |F_buenos(s) - F_malos(s)|.

    Mide la maxima separacion entre las CDFs de buenos y malos.
    Rango [0, 1]; en credit scoring un KS > 0.30 se considera aceptable y > 0.50 muy bueno.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Equivalente al indice de Somers' D para targets binarios."""
    auc = roc_auc_score(y_true, y_score)
    return float(2 * auc - 1)


# ---------------------------------------------------------------------------
# Estabilidad: PSI
# ---------------------------------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
        strategy: str = "quantile") -> float:
    """Population Stability Index entre dos distribuciones.

    Reglas habituales en banca:
        PSI < 0.10  -> sin cambio relevante
        0.10 - 0.25 -> cambio menor, monitorear
        > 0.25      -> cambio significativo, recalibrar / reentrenar

    Parametros
    ----------
    expected : referencia (ej. mes 1, o muestra de desarrollo)
    actual   : nueva poblacion (ej. mes 2, mes 3, OOT)
    bins     : numero de bandas
    strategy : 'quantile' usa los cuantiles de `expected` (recomendado para
               score continuo), 'uniform' usa cortes equidistantes.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if strategy == "quantile":
        cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
        cuts = np.unique(cuts)
        # bordes infinitos para que `actual` fuera de rango caiga en bandas extremas
        cuts[0], cuts[-1] = -np.inf, np.inf
    else:
        cuts = np.linspace(expected.min(), expected.max(), bins + 1)
        cuts[0], cuts[-1] = -np.inf, np.inf

    e_perc = pd.Series(pd.cut(expected, bins=cuts, include_lowest=True)).value_counts(normalize=True).sort_index()
    a_perc = pd.Series(pd.cut(actual, bins=cuts, include_lowest=True)).value_counts(normalize=True).sort_index()
    # alinear indice por si una banda esta vacia en `actual`
    a_perc = a_perc.reindex(e_perc.index, fill_value=0.0)

    # evita log(0) con un epsilon
    eps = 1e-6
    e_perc = e_perc.clip(lower=eps)
    a_perc = a_perc.clip(lower=eps)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


# ---------------------------------------------------------------------------
# IV / WoE para feature selection
# ---------------------------------------------------------------------------

def woe_iv(df: pd.DataFrame, feature: str, target: str,
           bins: int = 10, is_categorical: bool = False) -> pd.DataFrame:
    """Calcula la tabla WoE/IV para una variable.

    WoE_i = ln(%buenos_i / %malos_i)
    IV    = sum_i (%buenos_i - %malos_i) * WoE_i

    Reglas comunes (Siddiqi):
        IV < 0.02           -> no predictiva
        0.02 - 0.10         -> debil
        0.10 - 0.30         -> media
        > 0.30              -> fuerte (sospechar leakage si es muy alta)
    """
    data = df[[feature, target]].copy()
    if is_categorical or data[feature].dtype == "object":
        data["_bin"] = data[feature].astype("string").fillna("__NA__")
    else:
        try:
            data["_bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
        except ValueError:
            data["_bin"] = pd.cut(data[feature], bins=bins)
        data["_bin"] = data["_bin"].astype("string").fillna("__NA__")

    g = data.groupby("_bin", observed=True).agg(
        n=(target, "size"),
        malos=(target, "sum"),
    )
    g["buenos"] = g["n"] - g["malos"]
    g["%buenos"] = g["buenos"] / g["buenos"].sum()
    g["%malos"] = g["malos"] / g["malos"].sum()
    eps = 1e-6
    g["WoE"] = np.log((g["%buenos"].clip(lower=eps)) / (g["%malos"].clip(lower=eps)))
    g["IV_i"] = (g["%buenos"] - g["%malos"]) * g["WoE"]
    g["bad_rate"] = g["malos"] / g["n"]
    g["IV_total"] = g["IV_i"].sum()
    return g.reset_index()


def iv_summary(df: pd.DataFrame, target: str, features: list[str],
               bins: int = 10) -> pd.DataFrame:
    """Tabla resumen de IV por variable, ordenada de mayor a menor."""
    rows = []
    for f in features:
        is_cat = df[f].dtype == "object"
        try:
            tbl = woe_iv(df, f, target, bins=bins, is_categorical=is_cat)
            iv = tbl["IV_i"].sum()
        except Exception as e:
            iv = np.nan
        rows.append({"variable": f, "IV": iv,
                     "fuerza": _iv_strength(iv)})
    return pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)


def _iv_strength(iv: float) -> str:
    if pd.isna(iv):
        return "n/a"
    if iv < 0.02:
        return "no predictiva"
    if iv < 0.10:
        return "debil"
    if iv < 0.30:
        return "media"
    if iv < 0.50:
        return "fuerte"
    return "sospechosa (verificar leakage)"


# ---------------------------------------------------------------------------
# Tabla decilica (gains table) por bandas de score
# ---------------------------------------------------------------------------

def decile_table(y_true: np.ndarray, y_score: np.ndarray,
                 bins: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Construye la tabla por bandas de score (decilica por defecto).

    Por convencion ordena los scores DE MAYOR A MENOR riesgo (decil 1 = peor).
    Sirve para ver concentracion de malos, KS por banda y lift acumulado.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df = df.sort_values("score", ascending=ascending).reset_index(drop=True)
    df["banda"] = pd.qcut(df.index, q=bins, labels=range(1, bins + 1))

    g = df.groupby("banda", observed=True).agg(
        n=("y", "size"),
        malos=("y", "sum"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    )
    g["buenos"] = g["n"] - g["malos"]
    g["bad_rate"] = g["malos"] / g["n"]
    total_malos, total_buenos = g["malos"].sum(), g["buenos"].sum()
    g["%malos_acum"] = g["malos"].cumsum() / total_malos
    g["%buenos_acum"] = g["buenos"].cumsum() / total_buenos
    g["KS"] = (g["%malos_acum"] - g["%buenos_acum"]).abs()
    g["lift"] = g["bad_rate"] / (total_malos / g["n"].sum())
    return g.reset_index()


# ---------------------------------------------------------------------------
# Tabla resumen comparativa de modelos
# ---------------------------------------------------------------------------

def model_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Bloque de metricas estandar para reportar un modelo de score."""
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Gini": gini_coefficient(y_true, y_score),
        "KS": ks_statistic(y_true, y_score),
        "n": int(len(y_true)),
        "tasa_mala": float(np.mean(y_true)),
    }
