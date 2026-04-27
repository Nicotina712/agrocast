"""Connector for NOAA Climate Data Online (CDO) API.

This connector provides access to NOAA's Climate Data Online (CDO) service,
which supplies weather and climate statistics such as precipitation and
temperature. The API requires a token and uses endpoints such as
``/data``, ``/datasets``, and ``/stations``. Each API call must include
the token in the ``token`` header and is rate limited (5 requests per
second, 10k per day)【734280521679970†L40-L58】【734280521679970†L74-L86】.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Iterable, List

import pandas as pd
import requests

from .base import Connector, coerce_numeric


class NOAAConnector(Connector):
    """Fetch data from NOAA CDO.

    Parameters
    ----------
    token : str, optional
        API token. If omitted, the ``NOAA_TOKEN`` environment variable is
        used instead.
    """

    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("NOAA_TOKEN")
        if not self.token:
            raise RuntimeError("NOAA_TOKEN must be set via .env or passed to the constructor")

    def fetch(
        self,
        datasetid: str,
        datatypeid: Iterable[str],
        locationid: str,
        start: date,
        end: date,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch NOAA climate data.

        Parameters
        ----------
        datasetid : str
            Identifier of the dataset (e.g., "GHCND").
        datatypeid : iterable of str
            One or more data type identifiers (e.g., ["TMAX", "TMIN"]).
        locationid : str
            Location identifier (e.g., "FIPS:28101" or station ID).
        start, end : datetime.date
            Date range for the request.
        limit : int, optional
            Maximum number of results to return (default 1000). The API has
            a limit per request; use pagination for larger queries.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``Date`` and one column per data type.
        """
        headers = {"token": self.token}
        params = {
            "datasetid": datasetid,
            "datatypeid": ",".join(datatypeid),
            "locationid": locationid,
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "units": "standard",
            "limit": limit,
        }
        r = requests.get(self.BASE_URL, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        records = js.get("results", [])
        if not records:
            return pd.DataFrame(columns=["Date"])
        df = pd.DataFrame(records)
        # NOAA returns each record with ``date``, ``datatype`` and ``value``
        df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        if {"datatype", "value"}.issubset(df.columns):
            df = df.pivot_table(index="Date", columns="datatype", values="value", aggfunc="mean").reset_index()
        return coerce_numeric(df, keep=("Date",)).sort_values("Date").reset_index(drop=True)