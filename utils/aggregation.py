import pandas as pd
import numpy as np
from .metrics import calculate_weeks_in_data
from .logger import logger

def aggregate_sales_history(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize shipped pounds, orders, cost & revenue by SKU.
    """
    return (
        sales_df
        .groupby(["SKU", "Supplier", "Protein", "Description"], as_index=False)
        .agg(
            ShippedLb=("ShippedLb", "sum"),
            QuantityOrdered=("QuantityOrdered", "sum"),
            Cost=("Cost", "sum"),
            Rev=("Rev", "sum")
        )
    )

def merge_data(
    agg_sales: pd.DataFrame,
    inv_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    cost_val_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join inventory, sales and production into one flattened table.
    """
    # 1) Inventory on-hand aggregation
    inv_agg = (
        inv_df
        .groupby(["SKU", "ProductState", "ProductName"], as_index=False)
        .agg(
            OnHandWeightLb=("WeightLb", "sum"),
            OnHandCost=("CostValue", "sum")
        )
    )

    # 2) Supplier-map from SSN if present
    supplier_map = (
        inv_df.dropna(subset=["Ssn"])
             .set_index("SKU")["Ssn"]
             .to_dict()
        if "Ssn" in inv_df.columns else {}
    )

    # 3) Join sales onto inventory
    df = inv_agg.merge(agg_sales, on="SKU", how="left")
    df = df.fillna({
        "Supplier": "",
        "Protein": "",
        "Description": "",
        "ShippedLb": 0.0,
        "QuantityOrdered": 0,
        "Cost": 0.0,
        "Rev": 0.0
    })

    # 4) Backfill blank Supplier
    mask_blank = df["Supplier"].astype(str).str.strip() == ""
    df.loc[mask_blank, "Supplier"] = (
        df.loc[mask_blank, "SKU"]
          .map(supplier_map)
          .fillna("")
    )

    # 5) Production shipped
    if "ProductionShippedLb" in prod_df.columns:
        prod_df["ProductionShippedLb"] = (
            pd.to_numeric(prod_df["ProductionShippedLb"], errors="coerce")
              .fillna(0.0)
        )
    else:
        prod_df["ProductionShippedLb"] = (
            pd.to_numeric(prod_df.get("WeightLb", 0), errors="coerce")
              .fillna(0.0)
        )

    # 6) Final merge
    df = df.merge(
        prod_df[["SKU", "ProductionShippedLb"]],
        on="SKU",
        how="left"
    )
    df["ProductionShippedLb"] = df["ProductionShippedLb"].fillna(0.0)

    return df

def aggregate_final_data(df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    From a merged SKU-level DataFrame (inv + sales + production),
    compute:
      - TotalUsage = shipped + production
      - AvgWeeklyUsage = TotalUsage / weeks_span
      - WeeksOnHand      = OnHandWeightTotal / AvgWeeklyUsage  (NaN if TotalUsage == 0)
      - AnnualTurns      = (AvgWeeklyUsage * 52) / OnHandWeightTotal
    Adds WOH_Flag = "No Usage" for products with zero TotalUsage.
    Falls back to 4-week span if calculate_weeks_in_data fails.
    """
    # 1) Determine weeks of data
    try:
        weeks_span = calculate_weeks_in_data(sales_df)
        if weeks_span <= 0:
            raise ValueError(f"invalid span {weeks_span}")
    except Exception as e:
        logger.warning(f"calculate_weeks_in_data failed ({e}); defaulting to 4 weeks")
        weeks_span = 4

    # 2) Aggregate to SKU level (keeping identifiers)
    sku_stats = (
        df
        .groupby(
            ["SKU", "Supplier", "Protein", "Description", "ProductState", "ProductName"],
            as_index=False
        )
        .agg(
            OnHandWeightTotal  = ("OnHandWeightLb",      "sum"),
            OnHandCostTotal    = ("OnHandCost",          "sum"),
            TotalShippedLb     = ("ShippedLb",           "sum"),
            TotalProductionLb  = ("ProductionShippedLb", "sum"),
            TotalRevenue       = ("Rev",                 "sum"),
            TotalCost          = ("Cost",                "sum")
        )
    )

    # 3) Compute usage metrics
    sku_stats["TotalUsage"]     = sku_stats["TotalShippedLb"] + sku_stats["TotalProductionLb"]
    sku_stats["AvgWeeklyUsage"] = sku_stats["TotalUsage"] / weeks_span

    # 4) Flag zero-usage SKUs
    sku_stats["WOH_Flag"] = np.where(sku_stats["TotalUsage"] == 0, "No Usage", "")

    # 5) Compute WeeksOnHand (NaN when TotalUsage == 0)
    sku_stats["WeeksOnHand"] = (
        sku_stats["OnHandWeightTotal"]
        / sku_stats["AvgWeeklyUsage"].replace({0: np.nan})
    )
    sku_stats["WeeksOnHand"] = sku_stats["WeeksOnHand"].replace([np.inf, -np.inf], np.nan)

    # 6) Compute AnnualTurns
    sku_stats["AnnualTurns"] = (
        sku_stats["AvgWeeklyUsage"] * 52.0
        / sku_stats["OnHandWeightTotal"].replace({0: np.nan})
    )
    sku_stats["AnnualTurns"] = sku_stats["AnnualTurns"].replace([np.inf, -np.inf], 0).fillna(0)

    # 7) Build SKU_Desc
    sku_stats["SKU_Desc"] = (
        sku_stats["SKU"]
        + " – "
        + sku_stats["ProductName"].where(
            sku_stats["ProductName"].ne(""),
            sku_stats["Description"]
        )
    )

    logger.info(f"Aggregated final data: {len(sku_stats)} SKUs")
    return sku_stats
