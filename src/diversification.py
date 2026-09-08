"""Diversification: correlation filter and sector clustering."""

from __future__ import annotations

import pandas as pd

# Common NSE sector map (extend as needed).
SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "TITAN": "Consumer",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto",
    "DIVISLAB": "Pharma",
    "ALKEM": "Pharma",
    "TORNTPHARM": "Pharma",
    "GLAXO": "Pharma",
    "SUNPHARMA": "Pharma",
    "ABB": "Industrial",
    "SIEMENS": "Industrial",
    "HAL": "Defence",
    "DATAPATTNS": "Defence",
    "GRASIM": "Cement",
    "ULTRACEMCO": "Cement",
    "MCX": "Financial",
    "HDFCAMC": "Financial",
    "RADICO": "Consumer",
    "MANKIND": "Pharma",
    "ATUL": "Chemicals",
    "NAVINFLUOR": "Chemicals",
    "FLUOROCHEM": "Chemicals",
    "KINGFA": "Chemicals",
    "NETWEB": "IT",
    "MTARTECH": "Industrial",
    "POWERMECH": "Infra",
    "GRSE": "Defence",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "Other")


def return_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame, days: int = 20) -> float:
    if df_a is None or df_b is None or len(df_a) < days or len(df_b) < days:
        return 0.0
    ret_a = df_a["close"].tail(days).pct_change().dropna()
    ret_b = df_b["close"].tail(days).pct_change().dropna()
    aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def diversify_picks(
    scored: list[tuple],
    history: dict[str, pd.DataFrame],
    max_per_sector: int = 1,
    max_correlation: float = 0.75,
) -> list:
    """
    Greedily select picks avoiding sector clustering and high correlation.
    `scored` items: (candidate, score, signals) sorted by score desc.
    """
    selected: list = []
    for item in scored:
        candidate = item[0]
        symbol = candidate.symbol
        sector = get_sector(symbol)

        sector_count = sum(1 for s in selected if get_sector(s[0].symbol) == sector)
        if sector_count >= max_per_sector:
            continue

        too_correlated = False
        sym_df = history.get(symbol)
        for existing in selected:
            other_df = history.get(existing[0].symbol)
            corr = return_correlation(sym_df, other_df)
            if corr > max_correlation:
                too_correlated = True
                break
        if too_correlated:
            continue

        selected.append(item)
    return selected
