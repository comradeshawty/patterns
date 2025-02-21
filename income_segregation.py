import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

def get_income_data():
  income_data = pd.DataFrame({
    'Income Bracket': [
        'Less than $10,000', '$10,000 to $14,999', '$15,000 to $24,999',
        '$25,000 to $34,999', '$35,000 to $49,999', '$50,000 to $74,999',
        '$75,000 to $99,999', '$100,000 to $149,999', '$150,000 to $199,999', '$200,000 or more'],
    'Percentage': [5.3, 4.6, 7.7, 8.3, 10.4, 17.2, 13.2, 15.9, 8.1, 9.3]})
  total_households = 471767
  return income_data,total_households

def impute_missing_values(gdf, columns_to_impute='median_hh_income', k=5):
    """
    Impute missing values in the GeoDataFrame using spatial nearest neighbors with an imputed_flag column.
    """
    gdf=gdf.to_crs(epsg=32616)
    coords = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))
    tree=cKDTree(coords)
    gdf['imputed_flag'] = 0
    for idx, row in gdf.iterrows():
        if pd.isna(row[columns_to_impute]):
            distances, indices = tree.query(coords[idx], k=k)
            valid_neighbors = gdf.iloc[indices][columns_to_impute].dropna()

            if not valid_neighbors.empty:
                imputed_values = valid_neighbors.mean()
                gdf.loc[idx, columns_to_impute] = imputed_values
                gdf.loc[idx, 'imputed_flag'] = 1  
    return gdf

def calculate_income_quantiles_cbsa(total_households, income_data, cbg_gdf):
    income_data['Households'] = (income_data['Percentage'] / 100) * total_households
    income_data['Cumulative Households'] = income_data['Households'].cumsum()
    quantiles = [0, total_households * 0.25, total_households * 0.5, total_households * 0.75, total_households]
    income_data['cbsa_income_quantile'] = pd.cut(income_data['Cumulative Households'], bins=quantiles,
                                            labels=['low', 'lower_middle', 'upper_middle', 'high'], include_lowest=True)
    cbg_gdf['income_quantile'] = pd.qcut(cbg_gdf['median_hh_income'], 4, labels=['low', 'lower_middle', 'upper_middle', 'high'])

    return cbg_gdf

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
