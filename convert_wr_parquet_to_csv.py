import pandas as pd
import os
from pathlib import Path

# Define the base path for WR data
base_path = Path("data/raw/wr")

# Years to convert
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Convert each year's parquet file to CSV
for year in years:
    parquet_path = base_path / str(year) / "data" / "full-00000-of-00001.parquet"
    csv_path = base_path / str(year) / "data" / f"wr_{year}.csv"
    
    if parquet_path.exists():
        try:
            print(f"Converting {year}...", end=" ")
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
            print(f"✓ Success! Saved to {csv_path}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"✗ Parquet file not found: {parquet_path}")

print("\nConversion complete!")
