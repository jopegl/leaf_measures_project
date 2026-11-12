"""
Busca de padrões entre o erro por imagem e as características das imagens.

Só estatística simples e interpretável (sem scipy):
  - correlação de Spearman entre erro e cada característica;
  - piores X% vs. demais: probabilidade de superioridade / correlação rank-biserial;
  - características binárias: proporção nas piores vs. nas demais;
  - pastas de folha com erro sistematicamente maior;
  - viés de "regressão à média" (superestima pequenas, subestima grandes).

Cada achado relevante vira um "indício" com força (forte/moderado), texto em
português e os números que o sustentam. São CORRELAÇÕES: apontam onde olhar,
não provam causa. Muitos testes são feitos ao mesmo tempo, então exigimos
efeito mínimo E significância corrigida para múltiplos testes
(Benjamini-Hochberg, FDR 5%) para reduzir falsos positivos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leaf_measures.analysis.features import BINARY_FEATURES, FEATURE_INFO, feature_label
from leaf_measures.data.normalization import MEASURE_ORDER

MEASURE_PT = {"length": "comprimento", "width": "largura", "perimeter": "perímetro", "area": "área",
              "mean": "média das 4 medidas"}
SIZE_FEATURES = {"leaf_area", "leaf_length", "leaf_width", "leaf_perimeter"}

# limiares
MIN_N = 15
STRONG, MODERATE = 0.5, 0.3
P_MAX = 0.01   # para análises com poucos testes (pastas, viés)
Q_MAX = 0.05   # FDR (Benjamini-Hochberg) para as ~200 combinações característica × medida


@dataclass
class Clue:
    strength: str            # 'forte' | 'moderado' | 'info'
    kind: str                # 'caracteristica' | 'pasta' | 'vies' | 'anotacao'
    measure: str
    text: str
    feature: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return self.stats.get("scope") or MEASURE_PT[self.measure]

    @property
    def score(self) -> float:
        return {"forte": 2, "moderado": 1, "info": 0}[self.strength] + abs(self.stats.get("effect", 0))


# ------------------------------------------------------------------ estatística
def _p_normal(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < MIN_N or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, np.nan, n
    rho = float(x[ok].rank().corr(y[ok].rank()))
    if abs(rho) >= 1:
        return rho, 0.0, n
    # Transformação de Fisher com a variância de Fieller et al. (1957) para Spearman: bem calibrada
    # mesmo com n pequeno (a aproximação normal direta subestima o p-valor).
    z = math.atanh(rho) * math.sqrt((n - 3) / 1.06)
    return rho, _p_normal(z), n


def rank_biserial(values: pd.Series, group: pd.Series) -> tuple[float, float]:
    """
    Compara `values` entre group=True e group=False.
    Retorna (r, p): r = 2·P(valor do grupo True > valor do grupo False) − 1, em [-1, 1].
    """
    ok = values.notna()
    v, g = values[ok], group[ok].astype(bool)
    n1, n2 = int(g.sum()), int((~g).sum())
    if n1 < 3 or n2 < 3 or v.nunique() < 2:
        return np.nan, np.nan
    ranks = v.rank()
    u = ranks[g].sum() - n1 * (n1 + 1) / 2
    r = 2 * u / (n1 * n2) - 1
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return float(r), _p_normal((u - n1 * n2 / 2) / sd)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """q-values (FDR) de Benjamini-Hochberg; NaN é preservado."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / np.arange(1, pv.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _strength(effect: float, p: float) -> str | None:
    if np.isnan(effect) or np.isnan(p) or p > P_MAX:
        return None
    if abs(effect) >= STRONG:
        return "forte"
    if abs(effect) >= MODERATE:
        return "moderado"
    return None


def _fmt(v: float, feat: str) -> str:
    if feat in BINARY_FEATURES:
        return f"{100 * v:.0f}%"
    unit = FEATURE_INFO.get(feat, ("", ""))[1]
    if abs(v) >= 100:
        s = f"{v:,.0f}".replace(",", ".")
    elif abs(v) >= 1:
        s = f"{v:.2f}"
    else:
        s = f"{v:.3f}"
    if unit == "%":
        return f"{s}%"
    if unit in ("cm²", "cm", "px", "px/cm²"):
        return f"{s} {unit}"
    return s


# ------------------------------------------------------------------ análises
def numeric_features(table: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_INFO if f in table and table[f].notna().sum() >= MIN_N
            and table[f].nunique() >= 2]


def target_columns(table: pd.DataFrame) -> dict:
    """{chave da medida: coluna de RER}."""
    cols = {m: f"{m}_rer" for m in MEASURE_ORDER if f"{m}_rer" in table}
    cols["mean"] = "rer_mean"
    return cols


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, col in target_columns(table).items():
        for f in numeric_features(table):
            rho, p, n = spearman(table[f], table[col])
            row = {"medida": m, "caracteristica": f, "descricao": feature_label(f), "rho": rho, "p": p, "n": n}
            if f in SIZE_FEATURES and m != "mean":
                row["rho_erro_absoluto"] = spearman(table[f], table[f"{m}_abs_error"])[0]
            rows.append(row)
    df = pd.DataFrame(rows, columns=["medida", "caracteristica", "descricao", "rho", "p", "n", "rho_erro_absoluto"])
    if df.empty:
        return df
    return df.reindex(df["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def worst_vs_rest(table: pd.DataFrame, quantile: float) -> pd.DataFrame:
    rows = []
    for m, col in target_columns(table).items():
        worst = table[col] >= table[col].quantile(1 - quantile)
        for f in numeric_features(table):
            r, p = rank_biserial(table[f], worst)
            rows.append({
                "medida": m, "caracteristica": f, "descricao": feature_label(f),
                "piores_mediana" if f not in BINARY_FEATURES else "piores_proporcao":
                    table.loc[worst, f].median() if f not in BINARY_FEATURES else table.loc[worst, f].mean(),
                "demais_mediana" if f not in BINARY_FEATURES else "demais_proporcao":
                    table.loc[~worst, f].median() if f not in BINARY_FEATURES else table.loc[~worst, f].mean(),
                "rank_biserial": r, "p": p, "n_piores": int(worst.sum()), "n_demais": int((~worst).sum()),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["medida", "caracteristica", "rank_biserial", "p"])
    return df.reindex(df["rank_biserial"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def folders(table: pd.DataFrame, min_images: int = 3) -> pd.DataFrame:
    if "leaf_folder" not in table:
        return pd.DataFrame()
    g = table.groupby("leaf_folder")["rer_mean"]
    df = pd.DataFrame({"n_imagens": g.size(), "rer_medio": g.mean(), "rer_mediana": g.median()})
    df = df[df["n_imagens"] >= min_images]
    df["vezes_a_media_geral"] = df["rer_medio"] / table["rer_mean"].mean()
    return df.sort_values("rer_medio", ascending=False)


def bias(table: pd.DataFrame) -> pd.DataFrame:
    """Erro com sinal por quintil do valor real, por medida."""
    rows = []
    for m in MEASURE_ORDER:
        if f"{m}_real" not in table:
            continue
        real, err = table[f"{m}_real"], table[f"{m}_error"]
        rho, p, n = spearman(real, err)
        try:
            q = pd.qcut(real, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        by_q = err.groupby(q).mean()
        rows.append({"medida": m, "rho_real_vs_erro": rho, "p": p, "n": n,
                     **{f"erro_medio_q{int(k) + 1}": v for k, v in by_q.items()},
                     "erro_medio_geral": err.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ indícios
def scope_text(measures: list[str]) -> str:
    ms = [m for m in measures if m != "mean"]
    if len(ms) == len(MEASURE_ORDER):
        return "todas as medidas"
    if not ms:
        return MEASURE_PT["mean"]
    return ", ".join(MEASURE_PT[m] for m in ms)


def find_clues(table: pd.DataFrame, quantile: float = 0.1) -> tuple[list[Clue], dict]:
    """Roda todas as análises e devolve (indícios ordenados, tabelas de apoio)."""
    tables = {
        "correlacoes": correlations(table),
        "piores_vs_demais": worst_vs_rest(table, quantile),
        "pastas": folders(table),
        "vies": bias(table),
    }
    clues: list[Clue] = []
    if len(table) < MIN_N:
        print(f"[AVISO] só {len(table)} amostras: são necessárias pelo menos {MIN_N} para buscar padrões. "
              "Os rankings de piores/melhores continuam valendo.")
    _add_q_values(tables["correlacoes"], tables["piores_vs_demais"])
    clues += _feature_clues(table, tables["correlacoes"], tables["piores_vs_demais"], quantile)
    clues += _clues_from_folders(tables["pastas"], table)
    clues += _clues_from_bias(tables["vies"])
    clues += _clues_from_annotation(table)
    return sorted(clues, key=lambda c: -c.score), tables


def _add_q_values(corr: pd.DataFrame, wvr: pd.DataFrame):
    """Corrige os p-valores de todas as combinações característica × medida × método juntas."""
    p_all = np.concatenate([corr["p"].to_numpy(dtype=float), wvr["p"].to_numpy(dtype=float)])
    q_all = benjamini_hochberg(p_all)
    corr["q"] = q_all[: len(corr)]
    wvr["q"] = q_all[len(corr):]


def _strength_q(effect: float, q: float) -> str | None:
    if np.isnan(effect) or np.isnan(q) or q > Q_MAX:
        return None
    return "forte" if abs(effect) >= STRONG else ("moderado" if abs(effect) >= MODERATE else None)


def _feature_clues(table, corr: pd.DataFrame, wvr: pd.DataFrame, quantile: float) -> list[Clue]:
    """Um indício por característica, juntando todas as medidas em que ela aparece."""
    pct = int(round(quantile * 100))
    out = []
    for f in numeric_features(table):
        c = corr[corr["caracteristica"] == f].copy()
        w = wvr[wvr["caracteristica"] == f].copy()
        c["forca"] = [_strength_q(r, q) for r, q in zip(c["rho"], c["q"])]
        w["forca"] = [_strength_q(r, q) for r, q in zip(w["rank_biserial"], w["q"])]
        c_sig, w_sig = c[c["forca"].notna()], w[w["forca"].notna()]
        if c_sig.empty and w_sig.empty:
            continue

        effects = list(c_sig["rho"]) + list(w_sig["rank_biserial"])
        dominant = max(effects, key=abs)
        sign = 1 if dominant > 0 else -1
        c_sig = c_sig[np.sign(c_sig["rho"]) == sign]
        w_sig = w_sig[np.sign(w_sig["rank_biserial"]) == sign]
        measures = sorted(set(c_sig["medida"]) | set(w_sig["medida"]),
                          key=lambda m: (MEASURE_ORDER + ["mean"]).index(m))
        strength = "forte" if "forte" in set(c_sig["forca"]) | set(w_sig["forca"]) else "moderado"
        label = feature_label(f)
        scope = scope_text(measures)

        if f in BINARY_FEATURES:
            txt = f"Imagens com {label} têm erro {'maior' if sign > 0 else 'menor'}"
        else:
            txt = f"Erro maior em imagens com {'maior' if sign > 0 else 'menor'} {label}"
        if not c_sig.empty:
            lo, hi = c_sig["rho"].min(), c_sig["rho"].max()
            rng = f"ρ = {lo:+.2f}" if abs(hi - lo) < 0.005 else f"ρ de {lo:+.2f} a {hi:+.2f}"
            txt += f" ({rng}, n = {int(c_sig['n'].max())})."
        else:
            txt += "."

        ref = w[w["medida"] == "mean"]
        ref = ref.iloc[0] if not ref.empty else w.iloc[w["rank_biserial"].abs().argmax()]
        if f in BINARY_FEATURES:
            txt += (f" {100 * ref['piores_proporcao']:.0f}% das {pct}% piores imagens têm isso, contra "
                    f"{100 * ref['demais_proporcao']:.0f}% das demais.")
        else:
            txt += (f" Nas {pct}% piores, mediana {_fmt(ref['piores_mediana'], f)} contra "
                    f"{_fmt(ref['demais_mediana'], f)} nas demais.")

        if f in SIZE_FEATURES and sign < 0:
            abs_rhos = c.loc[c["rho_erro_absoluto"].notna(), "rho_erro_absoluto"] if "rho_erro_absoluto" in c else []
            if len(abs_rhos) and abs_rhos.abs().max() < MODERATE:
                txt += (" Atenção: o erro ABSOLUTO quase não muda com o tamanho; o RER é maior em folhas "
                        "pequenas principalmente porque divide por um valor menor.")
                strength = "moderado"

        out.append(Clue(strength, "caracteristica", "mean" if "mean" in measures else measures[0], txt, f,
                        {"effect": dominant, "measures": measures, "scope": scope}))
    _flag_confounded(table, out)
    return out


def _flag_confounded(table: pd.DataFrame, clues: list[Clue], threshold: float = 0.7):
    """Se duas características com indício andam juntas, avisa no indício mais fraco."""
    ordered = sorted(clues, key=lambda c: -abs(c.stats["effect"]))
    for i, weak in enumerate(ordered):
        for strong in ordered[:i]:
            rho, _, _ = spearman(table[weak.feature], table[strong.feature])
            if not np.isnan(rho) and abs(rho) >= threshold:
                weak.text += (f" ⚠️ Anda junto com {feature_label(strong.feature)} (ρ = {rho:+.2f} entre as "
                              "duas): pode ser o mesmo efeito, não dois.")
                weak.stats["confounded_with"] = strong.feature
                break


def _clues_from_folders(df: pd.DataFrame, table: pd.DataFrame) -> list[Clue]:
    out = []
    if df.empty or len(df) < 3:
        return out
    for folder, r in df.iterrows():
        if r["vezes_a_media_geral"] < 1.25:
            break
        vals = table.loc[table["leaf_folder"] == folder, "rer_mean"]
        rest = table.loc[table["leaf_folder"] != folder, "rer_mean"]
        eff, p = rank_biserial(pd.concat([vals, rest]),
                               pd.Series([True] * len(vals) + [False] * len(rest), index=list(vals.index) + list(rest.index)))
        s = _strength(eff, p) or ("moderado" if r["vezes_a_media_geral"] >= 2 and r["n_imagens"] >= 5 else None)
        if not s:
            continue
        txt = (f"A pasta '{folder}' tem RER médio de {r['rer_medio']:.2f}%, "
               f"{r['vezes_a_media_geral']:.1f}× a média geral ({int(r['n_imagens'])} imagens). "
               "Vale ver se essas fotos foram capturadas em condições diferentes.")
        out.append(Clue(s, "pasta", "mean", txt, str(folder),
                        {"effect": eff if not np.isnan(eff) else 0, "p": p, "scope": MEASURE_PT["mean"]}))
    return out


def _clues_from_bias(df: pd.DataFrame) -> list[Clue]:
    rows = []
    for _, r in df.iterrows():
        s_ = _strength(r["rho_real_vs_erro"], r["p"])
        if s_ and r["rho_real_vs_erro"] < 0:
            rows.append((r, s_))
    if not rows:
        return []
    measures = [r["medida"] for r, _ in rows]
    detail = "; ".join(f"{MEASURE_PT[r['medida']]}: {r.get('erro_medio_q1', np.nan):+.2f} nas menores, "
                       f"{r.get('erro_medio_q5', np.nan):+.2f} nas maiores" for r, _ in rows)
    strength = "forte" if any(s_ == "forte" for _, s_ in rows) else "moderado"
    eff = min(r["rho_real_vs_erro"] for r, _ in rows)
    txt = ("O modelo 'puxa para a média': superestima folhas pequenas e subestima as grandes "
           f"(erro médio com sinal — {detail}). Costuma indicar poucos exemplos nos extremos "
           "ou que o modelo ainda não captou bem a escala. Veja graficos/vies_erro_vs_real.png.")
    return [Clue(strength, "vies", measures[0], txt, "", {"effect": eff, "measures": measures,
                                                          "scope": scope_text(measures)})]


def _clues_from_annotation(table: pd.DataFrame, threshold: float = 15.0) -> list[Clue]:
    if "geom_area_rer" not in table or table["geom_area_rer"].notna().sum() < MIN_N:
        return []
    sus = table[table["geom_area_rer"] > threshold]
    if sus.empty:
        return []
    worst = table["rer_mean"] >= table["rer_mean"].quantile(0.9)
    frac_w = (table.loc[worst, "geom_area_rer"] > threshold).mean()
    frac_r = (table.loc[~worst, "geom_area_rer"] > threshold).mean()
    s = "forte" if frac_w >= 3 * max(frac_r, 0.01) and frac_w >= 0.2 else "info"
    txt = (f"{len(sus)} imagem(ns) têm a área calculada pela máscara + marcador divergindo mais de "
           f"{threshold:.0f}% da área anotada ({100 * frac_w:.0f}% entre as piores vs. "
           f"{100 * frac_r:.0f}% nas demais). Podem ter erro de anotação, de máscara ou marcador fora do "
           "plano da folha. Lista em suspeitas_anotacao.csv.")
    return [Clue(s, "anotacao", "area", txt, "geom_area_rer", {"effect": frac_w - frac_r, "scope": "anotação"})]
