"""Connector for the Federal Reserve Economic Data (FRED) API.

This connector provides a method to download time series observations
directly from the FRED API. Each series is specified by its ``series_id``
and will be merged on the ``Date`` column. The FRED API returns values
as strings; they are coerced to numeric via :func:`coerce_numeric`.

An API key is required. It can be provided via the constructor or via
the ``FRED_API_KEY`` environment variable (see `.env.example`). The FRED
documentation lists endpoints such as ``fred/series/observations`` and
explains parameters like ``series_id``, ``observation_start`` and
``frequency``【318843565029899†L101-L110】.
"""

from __future__ import annotations

import os
from datetime import date
from typing import List

import pandas as pd
import requests

from .base import Connector, coerce_numeric


class FredConnector(Connector):
    """Fetch observations from the FRED API.

    Examples
    --------
    ```
    from datetime import date
    from mvp_pipeline.src.connectors.fred import FredConnector

    fred = FredConnector(api_key="your_key")
    df = fred.fetch(start=date(2010,1,1), end=date(2024,1,1), series=["DTWEXBGS","PPIIDC"])
    ```
    """

    #: Base endpoint for FRED observations (JSON format)
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY must be set via .env or passed to the constructor")

    def _fetch_one(self, series_id: str, start: date, end: date, frequency: str = "m") -> pd.DataFrame:
        """Fetch a single FRED time series.

        Parameters
        ----------
        series_id : str
            Identifier for the FRED series (e.g. ``DTWEXBGS``).
        start : date
            Start date for observations.
        end : date
            End date for observations.
        frequency : str, optional
            Frequency of the returned data (default ``"m"`` for monthly). See the
            FRED docs for allowed values such as ``d`` for daily or ``m`` for
            monthly.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with ``Date`` and a single column named by ``series_id``.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "frequency": frequency,
        }
        r = requests.get(self.BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json().get("observations", [])
        df = pd.DataFrame(data)[["date", "value"]]
        df = df.rename(columns={"date": "Date", "value": series_id})
        df["Date"] = pd.to_datetime(df["Date"])
        return coerce_numeric(df, keep=("Date",))

    def fetch(self, start: date, end: date, series: List[str], frequency: str = "m") -> pd.DataFrame:
        """Fetch multiple series and merge them on the ``Date`` column.

        Parameters
        ----------
        start, end : datetime.date
            Date range for the API call.
        series : list of str
            One or more FRED series identifiers.
        frequency : str, optional
            Observation frequency (default "m" for monthly).

        Returns
        -------
        pandas.DataFrame
            Merged DataFrame containing ``Date`` and a column for each series.
        """
        if not series:
            raise ValueError("At least one series identifier is required")
        frames = [self._fetch_one(sid, start, end, frequency) for sid in series]
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on="Date", how="outer")
        return merged.sort_values("Date").reset_index(drop=True)