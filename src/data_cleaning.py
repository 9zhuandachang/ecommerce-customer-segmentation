"""Canonical Online Retail II loading, cleaning, and row-level audit."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


SOURCE_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

CANONICAL_COLUMNS = list(COLUMN_MAP.values())

# Operational, fee, adjustment, test, and voucher codes. Alphanumeric product
# codes such as 15056BL and DCGS0058 are intentionally not excluded.
NON_PRODUCT_CODES = {
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "B",
    "BANK CHARGES",
    "C2",
    "C3",
    "CRUK",
    "D",
    "DOT",
    "GIFT",
    "M",
    "PADS",
    "POST",
    "S",
    "TEST001",
    "TEST002",
}


def _normalize_identifier(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized.mask(normalized.eq(""), pd.NA)


def load_transactions(paths: Sequence[Path]) -> pd.DataFrame:
    """Load and canonically type all source files with source provenance."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(
            path,
            dtype={
                "Invoice": "string",
                "StockCode": "string",
                "Description": "string",
                "Customer ID": "string",
                "Country": "string",
            },
            low_memory=False,
        )
        missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame = frame[SOURCE_COLUMNS].rename(columns=COLUMN_MAP)
        frame["source_file"] = path.name
        frame["source_row"] = pd.RangeIndex(start=2, stop=len(frame) + 2)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["invoice"] = _normalize_identifier(data["invoice"])
    data["stock_code"] = _normalize_identifier(data["stock_code"])
    data["description"] = data["description"].astype("string").str.strip()
    data["customer_id"] = _normalize_identifier(data["customer_id"])
    data["country"] = data["country"].astype("string").str.strip()
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    return data


def _non_product_mask(data: pd.DataFrame) -> pd.Series:
    code = data["stock_code"].str.upper()
    return (
        code.isin(NON_PRODUCT_CODES)
        | code.str.startswith("GIFT_", na=False)
        | code.str.startswith("TEST", na=False)
    )


def clean_transactions(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply documented rules and return clean, excluded, and audit outputs."""
    rule_masks = {
        "invalid_invoice_date": data["invoice_date"].isna(),
        "missing_invoice": data["invoice"].isna(),
        "missing_stock_code": data["stock_code"].isna(),
        "invalid_quantity": data["quantity"].isna(),
        "invalid_unit_price": data["unit_price"].isna(),
        "exact_duplicate": data.duplicated(subset=CANONICAL_COLUMNS, keep="first"),
        "missing_customer_id": data["customer_id"].isna(),
        "cancelled_invoice": data["invoice"].str.startswith("C", na=False),
        "nonpositive_quantity": data["quantity"].le(0).fillna(False),
        "nonpositive_unit_price": data["unit_price"].le(0).fillna(False),
        "non_product_entry": _non_product_mask(data),
    }

    primary_reason = pd.Series("included", index=data.index, dtype="string")
    for reason, mask in rule_masks.items():
        primary_reason = primary_reason.mask(primary_reason.eq("included") & mask, reason)

    work = data.copy()
    work["cleaning_status"] = primary_reason
    excluded = work.loc[~primary_reason.eq("included")].copy()
    clean = work.loc[primary_reason.eq("included")].drop(columns="cleaning_status").copy()
    clean["line_amount"] = clean["quantity"] * clean["unit_price"]
    clean = clean.sort_values(
        ["invoice_date", "invoice", "stock_code", "customer_id"], kind="stable"
    ).reset_index(drop=True)
    excluded = excluded.sort_values(["source_file", "source_row"], kind="stable")

    definitions = {
        "invalid_invoice_date": "InvoiceDate cannot be parsed",
        "missing_invoice": "Invoice is missing",
        "missing_stock_code": "StockCode is missing",
        "invalid_quantity": "Quantity is not numeric",
        "invalid_unit_price": "Price is not numeric",
        "exact_duplicate": "All canonical source fields match an earlier row",
        "missing_customer_id": "Customer ID is missing",
        "cancelled_invoice": "Invoice begins with C",
        "nonpositive_quantity": "Quantity is zero or negative",
        "nonpositive_unit_price": "Price is zero or negative",
        "non_product_entry": "Operational, fee, adjustment, test, or voucher code",
        "included": "Row passes all cleaning rules",
    }
    rows = []
    for reason, mask in rule_masks.items():
        rows.append(
            {
                "rule": reason,
                "matched_rows": int(mask.sum()),
                "primary_rows": int(primary_reason.eq(reason).sum()),
                "action": "exclude",
                "definition": definitions[reason],
            }
        )
    rows.append(
        {
            "rule": "included",
            "matched_rows": int(primary_reason.eq("included").sum()),
            "primary_rows": int(primary_reason.eq("included").sum()),
            "action": "keep",
            "definition": definitions["included"],
        }
    )
    audit = pd.DataFrame(rows)
    return clean, excluded, audit
