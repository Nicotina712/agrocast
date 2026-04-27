"""Connector for the U.S. Energy Information Administration (EIA) API.

This connector targets the EIA API v2, which organizes data in a hierarchical
structure. Requests must include an API key in the URL, as documented by
the EIA: ``https://api.eia.gov/v2/API_route?api_key=xxxxxx``【501420058795718†L472-L483】. The EIA API
supports complex queries using ``facets`` and returns JSON with data rows
containing a ``period`` field (YYYY‑MM) and a ``value``. This connector
converts ``period`` into a ``Date`` column on the first of each month and
names the value column according to the requested route.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, Iterable

import pandas as pd
import requests

from .base import Connector, coerce_numeric


class EIAConnector(Connector):
    """Fetch data from the EIA API v2.

    Parameters
    ----------
    api_key : str, optional
        API key for the EIA service. If omitted, the ``EIA_API_KEY``
        environment variable is used.
    """

    BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("EIA_API_KEY must be set via .env or passed to the constructor")

    def fetch(
        self,
        route: str,
        start: date,
        end: date,
        frequency: str = "monthly",
        facets: Dict[str, Iterable[str]] | None = None,
    ) -> pd.DataFrame:
        """Fetch time series data from the EIA API.

        Parameters
        ----------
        route : str
            API route (e.g., ``"electricity/retail-sales"``).
        start, end : datetime.date
            Date range for the query.
        frequency : str, optional
            Granularity of the data (default ``"monthly"``). See EIA docs for
            other allowed values.
        facets : dict, optional
            Additional faceted filters (e.g., {"stateid": ["CO"], "sectorid": ["RES"]}).

        Returns
        -------
        pandas.DataFrame
            A DataFrame with ``Date`` and a value column. If multiple
            facets produce multiple series, they are aggregated under a
            single column named from the route.
        """
        url = f"{self.BASE_URL}/{route}/data"
        params = {
            "api_key": self.api_key,
            "frequency": frequency,
            "start": start.isoformat(),
            "end": end.isoformat(),
            # Ensure stable sorting for deterministic output
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        if facets:
            # Flatten facets into the expected parameter format
            for key, values in facets.items():
                if not isinstance(values, (list, tuple, set)):
                    values = [values]
                for idx, val in enumerate(values):
                    params[f"facets[{key}][]"] = val
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        data_rows = js.get("response", {}).get("data", [])
        if not data_rows:
            return pd.DataFrame(columns=["Date"])
        df = pd.DataFrame(data_rows)
        # Create a date column by appending day=1 to period (YYYY-MM)
        df["Date"] = pd.to_datetime(df["period"] + "-01")
        series_name = route.replace("/", "_")
        df = df[["Date", "value"]].rename(columns={"value": series_name})
        return coerce_numeric(df, keep=("Date",)).sort_values("Date").reset_index(drop=True)