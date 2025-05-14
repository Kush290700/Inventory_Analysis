import pandas as pd
import numpy as np
from .metrics import calculate_weeks_in_data
from .logger import logger

def compute_aging_buckets(df: pd.DataFrame, days_col: str = "DaysInStorage") -> pd.DataFrame:
    """
    Bin days in storage into standard age buckets.
    """
    bins   = [0,30,60,90,180,365,np.inf]
    labels = ["0–30","31–60","61–90","91–180","181–365","365+"]
    df["AgeBucket"] = pd.cut(df.get(days_col,0), bins=bins, labels=labels, right=False)
    return df


def abc_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ABC classes based on cumulative percent of InventoryValue.
    """
    df = df.copy().sort_values("InventoryValue", ascending=False)
    total = df["InventoryValue"].sum()
    df["CumPerc"] = df["InventoryValue"].cumsum() / total
    df["ABC"]     = pd.cut(
        df["CumPerc"],
        bins=[0,0.8,0.95,1.0],
        labels=["A","B","C"],
        include_lowest=True
    )
    return df.drop(columns=["CumPerc"])


def compute_holding_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-item holding cost components and ABC class.
    """
    today = pd.Timestamp("today").normalize()
    df = df.copy()

    df["DaysInStorage"]  = (today - df["OriginDate"].fillna(today)).dt.days.clip(lower=0)
    df["FractionOfYear"] = np.minimum(df["DaysInStorage"] / 365.0, 1.0)
    df["InventoryValue"] = df["OnHandCost"].fillna(0.0)

    total_val = df["InventoryValue"].sum()
    df["ValueFraction"]  = np.where(total_val>0, df["InventoryValue"]/total_val, 0.0)

    # cost parameters
    rc, sa = 0.05, 102055.0
    spc    = (71466*0.4 + 107128*0.7 + 48280*0.7 + 453626 + 544699*0.5)
    rr     = 0.03

    df["CapitalCost"] = df["InventoryValue"] * rc  * df["FractionOfYear"]
    df["ServiceCost"] = df["ValueFraction"]  * sa  * df["FractionOfYear"]
    df["StorageCost"] = df["ValueFraction"]  * spc * df["FractionOfYear"]
    df["RiskCost"]    = df["InventoryValue"] * rr  * df["FractionOfYear"]

    for c in ["CapitalCost","ServiceCost","StorageCost","RiskCost"]:
        df[c] = df[c].fillna(0.0)

    df["TotalHoldingCost"]   = df[
        ["CapitalCost","ServiceCost","StorageCost","RiskCost"]
    ].sum(axis=1)
    df["HoldingCostPercent"] = (
        df["TotalHoldingCost"] / df["InventoryValue"].replace({0:np.nan})
    ).fillna(0.0) * 100

    df = compute_aging_buckets(df)
    result = abc_classification(df)
    logger.info(f"Computed holding cost for {len(result)} items")
    return result
