import numpy as np
import pandas as pd

def compute_income_segregation(df, cbg_gdf):
    """
    For each POI (i.e. each row in df), compute an income segregation score
    based on the processed visitor counts.
    
    The function maps each CBG (key in the processed dict) to an income quartile
    (using cbg_income_map, which should map CBG (as int) to a quartile in {1,2,3,4}),
    sums the visitor counts by quartile, and then calculates the segregation
    measure as:
    
         segregation = (2/3) * sum(|proportion - 0.25|)
    
    where the proportion is the fraction of visitors from each quartile.
    
    Parameters:
      df             : DataFrame that now includes 'processed_visitor_home_cbgs'
      cbg_income_map : Dictionary mapping CBG (as int) to income quartile (1-4)
    
    Returns:
      The DataFrame with an added column 'income_segregation'
    """
    income_label_to_quartile = {"low": 1, "lower_middle": 2, "upper_middle": 3, "high": 4}
    
    # Ensure CBGs in `cbg_gdf` are correctly formatted
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)

    # Create mapping: CBG → Income Quartile
    cbg_income_map = cbg_gdf.set_index("cbg")["income_quantile"].map(income_label_to_quartile).to_dict()

    def segregation_from_dict(visitor_dict):
        """Compute the segregation score for a single POI."""
        # Array to store counts per quartile (4 income groups)
        quartile_counts = np.zeros(4, dtype=float)

        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)  # Ensure CBG is an integer
            except Exception:
                continue


    def segregation_from_dict(visitor_dict):
        # Build an array for the 4 income quartiles.
        quartile_counts = np.zeros(4, dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            quartile = cbg_income_map.get(cbg_int, None)

            if quartile is not None:
                quartile_counts[quartile - 1] += count  # Store in 0-based index

        total = quartile_counts.sum()
        if total == 0:
            return np.nan  # Avoid division by zero

        proportions = quartile_counts / total
        # Compute segregation score: (2/3) * sum(|proportion - 0.25|)
        segregation = (2/3) * np.sum(np.abs(proportions - 0.25))
        return segregation

    df['income_segregation'] = df['adjusted_cbg_visitors'].apply(segregation_from_dict)
    return df
