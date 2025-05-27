import io
import pandas as pd
from .logger import logger

def load_sheets(raw_file) -> dict[str, pd.DataFrame]:
    """
    Read the four expected sheets from a single .xlsx upload.
    Always rewinds the upload, and uses openpyxl explicitly.
    """
    expected = [
        "Sales History",
        "Cost Value",
        "Inventory Detail",
        "Production Batch",
        "Inventory Detail1",
        "Mikuni",
        "Product Detail"
    ]

    try:
        raw_file.seek(0)
    except Exception:
        pass

    content = raw_file.read()
    if not content:
        raise ValueError("Uploaded file is empty. Please re‐upload your .xlsx")

    bio = io.BytesIO(content)
    try:
        xls = pd.ExcelFile(bio, engine="openpyxl")
    except ValueError as e:
        name = getattr(raw_file, "name", "file")
        raise ValueError(f"Could not read '{name}' as .xlsx: {e}")

    missing = [s for s in expected if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Missing sheets from workbook: {missing}")

    dfs: dict[str, pd.DataFrame] = {}
    for name in expected:
        dfs[name] = xls.parse(name, dtype={"SKU": str})
        logger.info(f"Loaded '{name}' ({len(dfs[name])} rows)")
    return dfs


def load_inventory_snapshot(file) -> pd.DataFrame:
    """
    Load a single‐sheet snapshot (CSV or XLSX) into a DataFrame.
    """
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file, dtype={"SKU": str})
    else:
        # rewind & read as Excel
        try:
            file.seek(0)
        except Exception:
            pass
        df = pd.read_excel(io.BytesIO(file.read()), dtype={"SKU": str})
    logger.info(f"Loaded snapshot ({len(df)} rows)")
    return df
