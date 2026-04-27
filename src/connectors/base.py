"""Base connector classes and helpers for loading API-driven data.

The connectors defined here serve as a common interface for all external data
sources used by the soybeans forecasting pipeline. Each connector must
implement a ``fetch`` method that returns a pandas ``DataFrame`` with a
``Date`` column (dtype ``datetime64[ns]``) and one or more numeric columns.

Additional helper functions are provided to coerce non-date columns to
numeric, replacing non‑numeric values with NaN for downstream cleaning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd


class Connector(ABC):
    """Abstract base class for API connectors.

    Each concrete connector should implement the :meth:`fetch` method. The
    implementation should call the relevant API, convert the response to a
    pandas ``DataFrame``, and ensure that there is a ``Date`` column and at
    least one numeric data column. Numeric coercion should be handled via
    ``coerce_numeric`` below.
    """

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Retrieve a DataFrame from the API.

        Parameters
        ----------
        **kwargs
            Connector-specific keyword arguments such as start and end dates,
            series identifiers, facets, etc.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by ``Date`` with numeric columns. Columns
            containing non‑numeric values will be converted to NaN.
        """
        raise NotImplementedError


def coerce_numeric(df: pd.DataFrame, keep: Iterable[str] | None = None) -> pd.DataFrame:
    """Ensure all columns except those in ``keep`` are numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    keep : Iterable[str], optional
        Columns that should not be coerced to numeric. Typical examples
        include ``Date`` or other string identifiers. If ``None``, only the
        ``Date`` column is preserved.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` where non‑numeric columns have been converted to
        numeric with ``pd.to_numeric(errors="coerce")``. Non‑parsable values
        become NaN.
    """
    if keep is None:
        keep = {"Date"}
    else:
        keep = set(keep)

    out = df.copy()
    for col in out.columns:
        if col not in keep:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out