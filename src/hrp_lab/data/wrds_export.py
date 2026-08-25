"""Normalize browser-exported CRSP CIZ daily stock files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ALIASES: Mapping[str, tuple[str, ...]] = {
    "date": ("dlycaldt", "date", "caldt", "yyyymmdd"),
    "asset_id": ("permno", "security_id", "id"),
    "ticker": ("ticker", "tradingsymbol", "symbol"),
    "return": ("dlyret", "ret", "total_ret", "return"),
    "market_cap": ("dlycap", "market_cap", "mcap", "me"),
}


@dataclass(frozen=True)
class CRSPPanel:
    """Clean long data plus aligned daily return and capitalization panels."""

    long: pd.DataFrame
    returns: pd.DataFrame
    market_caps: pd.DataFrame
    labels: dict[int, str]
    duplicate_rows: int
    dropped_assets: tuple[int, ...]


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    }
    frame = frame.rename(columns=renamed)
    selected: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        found = next((name for name in aliases if name in frame.columns), None)
        if found is None:
            raise ValueError(
                f"Missing required {canonical!r} column; accepted names: {aliases}; "
                f"observed: {tuple(frame.columns)}"
            )
        selected[found] = canonical
    return frame.rename(columns=selected)[list(ALIASES)]


def _first_valid(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[-1] if not valid.empty else np.nan


def load_crsp_export(
    path: str | Path,
    *,
    minimum_coverage: float = 0.98,
    drop_incomplete_dates: bool = True,
) -> CRSPPanel:
    """Read a CRSP export and construct a stable PERMNO-indexed complete panel.

    The function never interpolates returns. Assets below the requested coverage
    are removed first; remaining missing cross sections are dropped by default.
    Duplicate PERMNO/date rows are collapsed only after retaining the last
    nonmissing value per field, matching the deterministic WRDS export order.
    """

    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in (0, 1]")
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    raw = pd.read_csv(data_path, compression="infer", low_memory=False)
    frame = _canonicalize_columns(raw)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["asset_id"] = pd.to_numeric(frame["asset_id"], errors="coerce")
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype("string").str.strip()
    frame = frame.dropna(subset=["date", "asset_id"])
    frame["asset_id"] = frame["asset_id"].astype(np.int64)
    frame.loc[~np.isfinite(frame["return"]), "return"] = np.nan
    frame.loc[frame["return"] < -1.0, "return"] = np.nan
    frame.loc[~np.isfinite(frame["market_cap"]) | (frame["market_cap"] <= 0), "market_cap"] = np.nan

    duplicate_rows = int(frame.duplicated(["date", "asset_id"], keep=False).sum())
    if duplicate_rows:
        duplicate_mask = frame.duplicated(["date", "asset_id"], keep=False)
        unique_rows = frame.loc[~duplicate_mask]
        collapsed_duplicates = (
            frame.loc[duplicate_mask]
            .sort_values(["date", "asset_id"])
            .groupby(["date", "asset_id"], as_index=False, sort=True)
            .agg({"ticker": _first_valid, "return": _first_valid, "market_cap": _first_valid})
        )
        frame = pd.concat([unique_rows, collapsed_duplicates], ignore_index=True)

    observation_count = frame.groupby("asset_id")["return"].count()
    max_count = int(observation_count.max()) if not observation_count.empty else 0
    keep_ids = observation_count[observation_count >= minimum_coverage * max_count].index
    dropped_assets = tuple(sorted(set(frame["asset_id"]) - set(keep_ids)))
    frame = frame[frame["asset_id"].isin(keep_ids)].copy()

    returns = frame.pivot(index="date", columns="asset_id", values="return").sort_index()
    market_caps = frame.pivot(index="date", columns="asset_id", values="market_cap").reindex_like(returns)
    if drop_incomplete_dates:
        complete = returns.notna().all(axis=1)
        returns = returns.loc[complete]
        market_caps = market_caps.loc[complete]
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Fewer than two assets remain after panel cleaning")

    ticker_map = (
        frame.dropna(subset=["ticker"])
        .sort_values("date")
        .groupby("asset_id")["ticker"]
        .last()
        .to_dict()
    )
    labels = {int(asset): str(ticker_map.get(asset, asset)) for asset in returns.columns}
    frame = frame.sort_values(["date", "asset_id"], ignore_index=True)
    return CRSPPanel(
        long=frame,
        returns=returns.astype(float),
        market_caps=market_caps.astype(float),
        labels=labels,
        duplicate_rows=duplicate_rows,
        dropped_assets=dropped_assets,
    )
