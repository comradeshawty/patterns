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

"""
Module: income_segregation
Repository: comradeshawty/patterns
Description: Data processing routines for income segregation analysis.
"""
import numpy as np
import pandas as pd

def compute_income_segregation(df, cbg_gdf):
    """
    For each POI (i.e. each row in df), compute an income segregation score
    based on the processed visitor counts, and compute experienced income segregation for each CBG.
    
    The function maps each CBG (key in the processed dict) to an income quartile
    (using cbg_income_map, which maps CBG (as int) to a quartile in {1,2,3,4}),
    sums the visitor counts by quartile per POI, and then calculates the POI segregation
    measure as:
    
         segregation = (2/3) * sum(|proportion - 0.25|)
    
    where the proportion is the fraction of visitors from each quartile at that POI.
    
    In addition, we compute the experienced income segregation for each CBG.
    For each POI (denoted by α):
      - τ₍q,α₎: the proportion of time at place α spent by income group q.
      - For each CBG b visiting that POI, τ₍b,α₎ is calculated as the count for b at α divided by
        the total visitors at α, but then normalized across all POIs (i.e. divided by the CBG's global total visits).
    Then, for each CBG, the relative exposure is:
         τ₍b,q₎ = Σ₍α visited by b₎ (τ₍b,α₎ * τ₍q,α₎)
    and the experienced income segregation measure is:
         Sᵢ = (2/3) * Σ₍q=1...4₎ |τ₍b,q₎ − 0.25|
    
    Parameters:
      df     : DataFrame that includes the 'adjusted_cbg_visitors' column containing the processed visitor counts.
      cbg_gdf: GeoDataFrame with CBG information and an 'income_quantile' column; 
               the income labels for CBGs are in {"low", "lower_middle", "upper_middle", "high"}.
    
    Returns:
      Tuple: (df with added column 'income_segregation', updated cbg_gdf with column 'experienced_income_segregation')
    """
    income_label_to_quartile = {"low": 1, "lower_middle": 2, "upper_middle": 3, "high": 4}
    
    # Ensure CBGs in cbg_gdf are correctly formatted
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    
    # Create mapping: CBG → Income Quartile
    cbg_income_map = cbg_gdf.set_index("cbg")["income_quantile"].map(income_label_to_quartile).to_dict()
    
    def segregation_from_dict(visitor_dict):
        """
        Compute the POI-level income segregation score along with the income quartile proportions.
        Returns a tuple: (POI segregation score, quartile proportions vector, total visits at the POI).
        """
        quartile_counts = np.zeros(4, dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            quartile = cbg_income_map.get(cbg_int, None)
            if quartile is not None:
                quartile_counts[quartile - 1] += count  # store in 0-based index
        
        total = quartile_counts.sum()
        if total == 0:
            return np.nan  # Avoid division by zero
        
        proportions = quartile_counts / total  # τ₍q,α₎ for each quartile at this POI
        segregation = (2/3) * np.sum(np.abs(proportions - 0.25))
        return segregation, proportions, total

    # Compute POI-level segregation scores and also prepare for CBG-level aggregation.
    poi_segregation_scores = []  # will hold segregation score for each POI
    # Dictionary to hold total visits per CBG across all POIs.
    total_visits_per_cbg = {}
    # Dictionary to accumulate exposure contributions per CBG; key: cbg, value: np.array (length 4)
    cbg_exposure = {}

    # First pass: calculate global total visits for each CBG across all POIs.
    for idx, row in df.iterrows():
        visitor_dict = row['adjusted_cbg_visitors']
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            total_visits_per_cbg[cbg_int] = total_visits_per_cbg.get(cbg_int, 0) + count
    
    # Second pass: compute each POI's quartile proportions and accumulate CBG exposure contributions.
    for idx, row in df.iterrows():
        visitor_dict = row['adjusted_cbg_visitors']
        seg_result = segregation_from_dict(visitor_dict)
        if seg_result is np.nan:
            poi_segregation_scores.append(np.nan)
            continue
        poi_segreg, quartile_proportions, total_alpha = seg_result
        poi_segregation_scores.append(poi_segreg)
        # For each CBG present in the POI, compute its weight for this POI and add its exposure contribution.
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            # τ₍b,α₎ for this POI b: fraction of the CBG's visits that occur at this POI.
            if total_visits_per_cbg.get(cbg_int, 0) == 0:
                continue
            tau_b_alpha = count / total_visits_per_cbg[cbg_int]
            # Contribution for each income quartile.
            contribution = tau_b_alpha * quartile_proportions
            if cbg_int in cbg_exposure:
                cbg_exposure[cbg_int] += contribution
            else:
                cbg_exposure[cbg_int] = np.array(contribution, dtype=float)
    
    # Add POI-level segregation scores to df.
    df['Sα'] = poi_segregation_scores
    null_rows = df[df['Sα'].isnull()]
    null_rows.to_csv('/content/drive/MyDrive/data/null_income_segregation.csv', index=False)
    df.dropna(subset=['Sα'], inplace=True, ignore_index=True)
    
    # Now compute experienced income segregation for each CBG.
    experienced_segregation = {}
    for cbg, exposure in cbg_exposure.items():
        # exposure is τ₍b,q₎: the CBG's relative exposure across the 4 income quartiles.
        segregation_value = (2/3) * np.sum(np.abs(exposure - 0.25))
        experienced_segregation[cbg] = segregation_value
    
    # Map experienced segregation back to the cbg_gdf.
    cbg_gdf['Si'] = cbg_gdf['cbg'].map(experienced_segregation)
    
    return df, cbg_gdf
