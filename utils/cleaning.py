import pandas as pd
from .logger import logger

def preprocess_data(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    cost_val_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean SKUs, split inventory SKU into code + name, parse numerics,
    and dedupe production & cost tables.
    """
    # Trim & string‐ify all SKUs
    for df in (sales_df, inv_df, prod_df, cost_val_df):
        df["SKU"] = df["SKU"].astype(str).str.strip()

    # Split Inventory SKU → code + ProductName
    tmp = inv_df["SKU"].str.split(" - ", n=1, expand=True)
    inv_df["SKU"] = tmp[0].fillna("")
    inv_df["ProductName"] = tmp[1].fillna(inv_df["SKU"])

    # Numeric cleaning on inventory
    if "WeightLb" in inv_df.columns:
        inv_df["WeightLb"] = (
            pd.to_numeric(inv_df["WeightLb"], errors="coerce")
              .fillna(0.0)
        )
    else:
        inv_df["WeightLb"] = 0.0

    if "CostValue" in inv_df.columns:
        inv_df["CostValue"] = (
            inv_df["CostValue"].astype(str)
            .replace(r"[\$,]", "", regex=True)
            .astype(float)
            .fillna(0.0)
        )
    else:
        inv_df["CostValue"] = 0.0

    # Numeric cleaning on sales
    for col in ("Cost", "Rev", "ShippedLb"):
        if col in sales_df.columns:
            sales_df[col] = (
                sales_df[col].astype(str)
                             .replace(r"[\$,]", "", regex=True)
                             .astype(float)
                             .fillna(0.0)
            )

    # Clean production costs
    if "CostNow" in prod_df.columns:
        prod_df["CostNow"] = (
            prod_df["CostNow"].astype(str)
                             .replace(r"[\$,]", "", regex=True)
                             .astype(float)
                             .fillna(0.0)
        )

    # Drop duplicates (keep first)
    prod_df     = prod_df.drop_duplicates(subset=["SKU"], keep="first")
    cost_val_df = cost_val_df.drop_duplicates(subset=["SKU"], keep="first")

    return sales_df, inv_df, prod_df, cost_val_df


def process_inventory_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a one‐sheet inventory snapshot:
      - Normalize SKU & ProductName
      - Default ItemCount to 1
      - Parse dates & compute OnHandWeight/Cost
    """
    df = df.copy()

    # SKU & ProductName
    df["SKU"] = df.get("SKU", "").astype(str).str.strip()
    if "ProductName" in df.columns:
        df["ProductName"] = df["ProductName"].astype(str)
    elif "Product" in df.columns:
        df["ProductName"] = df["Product"].astype(str)
    else:
        df["ProductName"] = ""

    # ItemCount (default 1)
    if "ItemCount" in df.columns:
        df["ItemCount"] = (
            pd.to_numeric(df["ItemCount"], errors="coerce")
              .fillna(1)
              .astype(int)
        )
    else:
        df["ItemCount"] = 1

    # WeightLb (default 0.0)
    if "WeightLb" in df.columns:
        df["WeightLb"] = (
            pd.to_numeric(df["WeightLb"], errors="coerce")
              .fillna(0.0)
        )
    else:
        df["WeightLb"] = 0.0

    # Cost per unit (default 0.0)
    cost_col = "Cost_pr" if "Cost_pr" in df.columns else "CostValue"
    if cost_col in df.columns:
        df["Cost_pr"] = (
            pd.to_numeric(df[cost_col], errors="coerce")
              .fillna(0.0)
        )
    else:
        df["Cost_pr"] = 0.0

    # Date parsing
    if "OriginDate" in df.columns:
        df["OriginDate"] = pd.to_datetime(df["OriginDate"], errors="coerce")
    elif "CreatedAt" in df.columns:
        df["OriginDate"] = pd.to_datetime(df["CreatedAt"], errors="coerce")
    else:
        df["OriginDate"] = pd.to_datetime("today")

    # Totals & descriptor
    df["OnHandWeightLb"] = df["WeightLb"] * df["ItemCount"]
    df["OnHandCost"]     = df["Cost_pr"]  * df["ItemCount"]
    df["SKU_Desc"]       = df["SKU"] + " – " + df["ProductName"]

    return df


def process_inventory_detail1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Inventory Detail1 sheet:
      - parse timestamps
      - fill missing location/bin
      - ensure numeric ItemCount & WeightLb
    """
    df = df.copy()

    # Timestamps
    df["BinScannedAt"] = pd.to_datetime(df.get("BinScannedAt"), errors="coerce")
    df["CreatedAt"]    = pd.to_datetime(df.get("CreatedAt"),    errors="coerce")

    # ProductLocation & LastKnownBin (default "Unknown")
    if "ProductLocation" in df.columns:
        df["ProductLocation"] = df["ProductLocation"].fillna("Unknown")
    else:
        df["ProductLocation"] = "Unknown"

    if "LastKnownBin" in df.columns:
        df["LastKnownBin"] = df["LastKnownBin"].fillna("Unknown")
    else:
        df["LastKnownBin"] = "Unknown"

    # ItemCount (default 1)
    if "ItemCount" in df.columns:
        df["ItemCount"] = (
            pd.to_numeric(df["ItemCount"], errors="coerce")
              .fillna(1)
              .astype(int)
        )
    else:
        df["ItemCount"] = 1

    # WeightLb (default 0.0)
    if "WeightLb" in df.columns:
        df["WeightLb"] = (
            pd.to_numeric(df["WeightLb"], errors="coerce")
              .fillna(0.0)
        )
    else:
        df["WeightLb"] = 0.0

    return df
