"""Ordinary least squares with store-clustered standard errors, in plain NumPy.

Weekly observations from the same store are not independent (a strong store is
strong every week), so ordinary OLS standard errors would be too small. We use
cluster-robust (CR1) standard errors grouped by store, the standard fix for
panel data like this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def design_matrix(
    df: pd.DataFrame,
    numeric: list[str] = (),
    categorical: list[str] = (),
    extra: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Intercept + numeric columns + one-hot categoricals (first level dropped)."""
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    parts += [df[c].astype(float).rename(c) for c in numeric]
    for c in categorical:
        parts.append(pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float))
    for name, s in (extra or {}).items():
        parts.append(s.astype(float).rename(name))
    X = pd.concat(parts, axis=1)
    return X.loc[:, X.abs().sum() > 0]  # drop dummies for unobserved levels


@dataclass
class OLSResult:
    params: pd.DataFrame  # coef, se, t, ci_low, ci_high
    r2: float
    n: int
    n_clusters: int

    def __getitem__(self, name: str) -> pd.Series:
        return self.params.loc[name]


def ols_clustered(y: pd.Series, X: pd.DataFrame, groups: pd.Series) -> OLSResult:
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    n, k = Xv.shape
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    resid = yv - Xv @ beta

    codes, _ = pd.factorize(groups)
    g = codes.max() + 1
    scores = np.zeros((g, k))
    np.add.at(scores, codes, Xv * resid[:, None])       # sum of x_i * u_i within each cluster
    meat = scores.T @ scores
    correction = (g / (g - 1)) * ((n - 1) / (n - k))     # CR1 small-sample adjustment
    vcov = xtx_inv @ meat @ xtx_inv * correction
    se = np.sqrt(np.diag(vcov))

    tss = ((yv - yv.mean()) ** 2).sum()
    params = pd.DataFrame(
        {"coef": beta, "se": se, "t": beta / se,
         "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se},
        index=X.columns,
    )
    return OLSResult(params=params, r2=1 - (resid @ resid) / tss, n=n, n_clusters=g)
