"""
Data Cleaning Module
UCI Online Retail II Dataset
Removes invalid rows and documents exclusions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the UCI Online Retail II Excel file (two sheets) and combine."""
    path = Path(filepath)
    logger.info(f"Loading raw data from {path}")

    if path.suffix in (".xlsx", ".xls"):
        df_2009 = pd.read_excel(filepath, sheet_name="Year 2009-2010", dtype=str)
        df_2010 = pd.read_excel(filepath, sheet_name="Year 2010-2011", dtype=str)
        df = pd.concat([df_2009, df_2010], ignore_index=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(filepath, dtype=str, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Raw rows loaded: {len(df):,}")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataframe.
    Returns (clean_df, audit_dict) where audit_dict documents all exclusions.
    """
    audit = {}
    original_count = len(df)

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Invoice": "InvoiceNo",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
        "Country": "Country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Cast types
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # 1. Drop rows with missing CustomerID (anonymous)
    mask_no_customer = df["CustomerID"].isna() | (df["CustomerID"].astype(str).str.strip() == "")
    n_no_customer = mask_no_customer.sum()
    df = df[~mask_no_customer]
    audit["missing_customer_id"] = int(n_no_customer)
    logger.info(f"  Removed {n_no_customer:,} rows with missing CustomerID")

    # 2. Drop rows with missing InvoiceDate
    mask_no_date = df["InvoiceDate"].isna()
    n_no_date = mask_no_date.sum()
    df = df[~mask_no_date]
    audit["missing_invoice_date"] = int(n_no_date)
    logger.info(f"  Removed {n_no_date:,} rows with missing InvoiceDate")

    # 3. Remove cancellations (InvoiceNo starts with 'C')
    mask_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    n_cancel = mask_cancel.sum()
    df = df[~mask_cancel]
    audit["cancellations"] = int(n_cancel)
    logger.info(f"  Removed {n_cancel:,} cancellation rows")

    # 4. Remove non-positive quantity
    mask_qty = df["Quantity"] <= 0
    n_qty = mask_qty.sum()
    df = df[~mask_qty]
    audit["non_positive_quantity"] = int(n_qty)
    logger.info(f"  Removed {n_qty:,} rows with non-positive quantity")

    # 5. Remove non-positive unit price
    mask_price = df["UnitPrice"] <= 0
    n_price = mask_price.sum()
    df = df[~mask_price]
    audit["non_positive_price"] = int(n_price)
    logger.info(f"  Removed {n_price:,} rows with non-positive unit price")

    # 6. Remove rows with missing/NaN Quantity or UnitPrice
    mask_nan = df["Quantity"].isna() | df["UnitPrice"].isna()
    n_nan = mask_nan.sum()
    df = df[~mask_nan]
    audit["nan_qty_or_price"] = int(n_nan)
    logger.info(f"  Removed {n_nan:,} rows with NaN quantity/price")

    # Compute line-item revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Cast CustomerID to string (strip decimals if loaded as float)
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    audit["original_rows"] = int(original_count)
    audit["clean_rows"] = int(len(df))
    audit["total_removed"] = int(original_count - len(df))
    audit["unique_customers"] = int(df["CustomerID"].nunique())
    audit["unique_invoices"] = int(df["InvoiceNo"].nunique())
    audit["date_range"] = f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}"

    logger.info(f"Cleaning complete. Remaining rows: {len(df):,}  |  Unique customers: {audit['unique_customers']:,}")
    return df.reset_index(drop=True), audit


def print_audit(audit: dict) -> None:
    """Pretty-print the audit dictionary."""
    print("\n===== DATA CLEANING AUDIT =====")
    for k, v in audit.items():
        print(f"  {k:<30}: {v}")
    print("================================\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning.py <path_to_raw_data>")
    else:
        raw = load_raw_data(sys.argv[1])
        clean, audit = clean_data(raw)
        print_audit(audit)
        out = Path("data/processed/clean_data.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out, index=False)
        print(f"Saved to {out}")
