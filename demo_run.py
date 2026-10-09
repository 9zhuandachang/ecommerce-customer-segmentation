"""
Quick demo script:
Generates synthetic data resembling UCI Online Retail II and
runs the full pipeline so you can test the dashboard without
downloading the actual dataset.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

RAW_PATH = Path("data/raw/online_retail_II.xlsx")
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n_customers=2000, n_rows=80000) -> pd.DataFrame:
    """Generate synthetic transaction data with three customer archetypes."""
    logger.info(f"Generating {n_rows:,} synthetic transactions for {n_customers:,} customers...")

    customer_ids = [f"C{i:05d}" for i in range(1, n_customers + 1)]

    # Archetypes: Champions (20%), Loyal (40%), At-Risk (40%)
    archetype = np.random.choice(["champion", "loyal", "atrisk"], size=n_customers, p=[0.2, 0.4, 0.4])
    cust_df = pd.DataFrame({"CustomerID": customer_ids, "archetype": archetype})

    start = pd.Timestamp("2009-12-01")
    end = pd.Timestamp("2011-12-09")
    total_days = (end - start).days

    rows = []
    invoice_counter = 10000
    for _, cust in cust_df.iterrows():
        arch = cust["archetype"]
        if arch == "champion":
            n_inv = np.random.randint(20, 60)
        elif arch == "loyal":
            n_inv = np.random.randint(6, 20)
        else:
            n_inv = np.random.randint(1, 6)

        # Random invoice dates spread across the dataset period
        inv_days = np.sort(np.random.choice(total_days, size=n_inv, replace=False))
        for day in inv_days:
            inv_date = start + pd.Timedelta(days=int(day))
            invoice_no = f"INV{invoice_counter:08d}"
            invoice_counter += 1
            n_lines = np.random.randint(1, 8)
            for _ in range(n_lines):
                qty = np.random.randint(1, 20)
                price = round(np.random.uniform(0.5, 50.0), 2)
                rows.append({
                    "Invoice": invoice_no,
                    "StockCode": f"SC{np.random.randint(1000, 9999)}",
                    "Description": "PRODUCT",
                    "Quantity": qty,
                    "InvoiceDate": inv_date,
                    "Price": price,
                    "Customer ID": cust["CustomerID"],
                    "Country": "United Kingdom",
                })

        if len(rows) >= n_rows:
            break

    df = pd.DataFrame(rows[:n_rows])
    return df


def main():
    # 1. Generate synthetic data
    df_raw = generate_synthetic_data()

    # Save as Excel (both sheets identical for simplicity)
    logger.info(f"Saving synthetic data to {RAW_PATH} ...")
    with pd.ExcelWriter(RAW_PATH, engine="openpyxl") as writer:
        half = len(df_raw) // 2
        df_raw.iloc[:half].to_excel(writer, sheet_name="Year 2009-2010", index=False)
        df_raw.iloc[half:].to_excel(writer, sheet_name="Year 2010-2011", index=False)
    logger.info("Synthetic data saved.")

    # 2. Run training pipeline
    logger.info("Running training pipeline on synthetic data...")
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from train_pipeline import main as train_main
    train_main()


if __name__ == "__main__":
    main()
    print("\nDemo complete. Run:  streamlit run app/dashboard.py")
