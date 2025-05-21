import pandas as pd
import numpy as np
from typing import Tuple

def compute_threshold_move(ext_df: pd.DataFrame, hc_df: pd.DataFrame) -> float:
    """
    Given the EXT‐state inventory and
    the holding‐cost calculation (hc_df),
    return a recommended Weeks-On-Hand threshold.
    For example, pick the median EXT WOH:
    """
    # only look at SKUs present in hc_df
    common = ext_df[ext_df["SKU"].isin(hc_df["SKU"])]
    if common.empty:
        return 1.0
    # here we pick the 50th percentile of EXT WeeksOnHand
    return common["WeeksOnHand"].median()


def classify_movement(df: pd.DataFrame, quantile: float = 0.5) -> pd.DataFrame:
    """
    Label each SKU 'High' or 'Slow' based on AvgWeeklyUsage quantile.
    """
    tmp = df.copy()
    qv  = tmp["AvgWeeklyUsage"].quantile(quantile)
    tmp["MovementClass"] = np.where(tmp["AvgWeeklyUsage"] >= qv, "High", "Slow")
    return tmp


def quadrantify(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    method: str = "median"
) -> Tuple[pd.DataFrame, float, float]:
    """
    Split xcol vs ycol into four quadrants by median or mean:
      - High-High, High-Low, Low-High, Low-Low
    Returns: (df_with_Quadrant, xm, ym)
    """
    # 1) Copy & drop NaN so medians/means are correct
    df_q = df.dropna(subset=[xcol, ycol]).copy()

    # 2) Compute thresholds
    if method == "median":
        xm, ym = df_q[xcol].median(), df_q[ycol].median()
    elif method == "mean":
        xm, ym = df_q[xcol].mean(),   df_q[ycol].mean()
    else:
        raise ValueError("method must be 'median' or 'mean'")

    # 3) Boolean masks
    hi_x = df_q[xcol] >= xm
    hi_y = df_q[ycol] >= ym

    # 4) Map to the strings, then use Python concat
    left = hi_x.map({True: "High", False: "Low"})
    right = hi_y.map({True: "High", False: "Low"})
    df_q["Quadrant"] = left + "-" + right

    # 5) (Optional) ensure a consistent category order
    df_q["Quadrant"] = pd.Categorical(
        df_q["Quadrant"],
        categories=["High-High","High-Low","Low-High","Low-Low"],
        ordered=True
    )

    return df_q, xm, ym


def top_n_by_metric(
    df: pd.DataFrame,
    group: str,
    metric: str,
    n: int = 10,
    asc: bool = False
) -> pd.DataFrame:
    """
    Return top-n groups sorted by a given metric.
    """
    agg = df.groupby(group, as_index=False)[metric].sum()
    return agg.sort_values(metric, ascending=asc).head(n)
