import pandas as pd
import geopandas as gpd
import numpy as np

def get_racial_data():
  racial_data=pd.DataFrame({'Race':['tot_pop','hispanic_pop','white_pop','black_pop','asian_pop','other_race_pop','two_race_pop'],
                            'Count':[1367101,78645,858252,359725,21623,6232,42624]})
  return racial_data
def agg_race_cols(cbg_gdf):
  cbg_gdf['oth_race_pop']=cbg_gdf['american_indian_pop']+cbg_gdf['hawaiian_pop']+cbg_gdf['other_race_pop']
  return cbg_gdf
from collections import defaultdict

def compute_racial_segregation_with_exposure(df, cbg_gdf):
    """
    Compute both POI-level and experienced racial segregation metrics.
    
    POI-level racial segregation is computed using a weighted mean fraction of 
    each racial group (e.g. white_pop, black_pop, hispanic_pop, asian_pop, other_race_pop) 
    available in the 'weighted_means' column of df. The segregation measure for each POI (PLACEKEY) is defined as:
    
        S_alpha_race = (n / (2*n - 2)) * sum(|τ(r, α) - 1/n|)
    
    where:
      - n is the number of racial groups (determined by the keys in weighted_means)
      - τ(r, α) is the fraction of visitors of racial group r at the POI.
      
    Experienced racial segregation is computed for each CBG as follows:
      - First, total visits per CBG are calculated from the "adjusted_cbg_visitors" column of df.
      - For each POI, and for each CBG visiting that POI, we compute the contribution as:
            tau_b_alpha = (visitor count for CBG at POI) / (total visits of that CBG)
        and accumulate the weighted racial exposure:
            exposure contribution = tau_b_alpha * weighted_means
      - The experienced racial segregation for a CBG is then:
      
            S_r_experienced = (n / (2*n - 2)) * sum(|τ(b, r) - 1/n|)
      
        where τ(b, r) is the CBG's relative exposure (accumulated from all POIs) to racial group r.
        
    Parameters:
        df : pandas.DataFrame
            DataFrame containing each POI (each row) with at least the following columns:
              - "weighted_means": a dict with keys corresponding to racial groups and values being the weighted fraction for that race.
              - "adjusted_cbg_visitors": a dict mapping CBG identifiers to visitor counts for that POI.
              
        cbg_gdf : pandas.DataFrame or geopandas.GeoDataFrame
            DataFrame containing CBG data. Must have a column "cbg" that can be mapped to the keys in the visitor counts.
            
    Returns:
        Tuple:
            - df: The input DataFrame with a new column "S_alpha_race" that stores the POI-level racial segregation score.
            - cbg_gdf: The cbg_gdf with a new column "experienced_racial_segregation" that stores the experienced racial segregation for each CBG.
    """
    # Determine number of racial groups and the constant factor in the segregation formula.
    # We assume that every row in 'weighted_means' uses the same set of keys.
    # Here, n can be 5 (e.g. white_pop, black_pop, hispanic_pop, asian_pop, other_race_pop)
    sample_weights = df['weighted_means'].iloc[0]
    racial_groups = list(sample_weights.keys())
    n = len(racial_groups)
    constant_factor = n / (2 * n - 2)
    
    # --- 1. Compute POI-level racial segregation ---
    # Convert the weighted_means dictionaries into a DataFrame (one column per racial group)
    weighted_means_df = pd.DataFrame(df['weighted_means'].tolist(), index=df.index)
    # Replace missing values with zero
    weighted_means_df = weighted_means_df.fillna(0)
    # Compute absolute differences from the ideal equal share (1/n)
    abs_diff = (weighted_means_df - (1.0 / n)).abs()
    # Compute the segregation score for each POI
    df['Sα_race'] = constant_factor * abs_diff.sum(axis=1)
    
    # --- 2. Compute experienced racial segregation per CBG ---
    
    # First, calculate total visits per CBG over all POIs.
    total_visits_per_cbg = {}
    for visitors in df["adjusted_cbg_visitors"]:
        for cbg, count in visitors.items():
            try:
                # Convert cbg to integer if possible.
                cbg_int = int(cbg)
            except Exception:
                continue
            total_visits_per_cbg[cbg_int] = total_visits_per_cbg.get(cbg_int, 0) + count
    
    # Prepare CBG exposure accumulation using a defaultdict of numpy arrays.
    # The exposure vector will have length equal to the number of racial groups.
    cbg_exposure = defaultdict(lambda: np.zeros(n, dtype=float))
    
    # Accumulate each POI's exposure contributions for each CBG.
    def accumulate_exposure(row):
        visitor_dict = row["adjusted_cbg_visitors"]
        # Ensure the weighted_means vector is in the same order as racial_groups.
        proportions = np.array([row["weighted_means"].get(race, 0) for race in racial_groups], dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            total_for_cbg = total_visits_per_cbg.get(cbg_int, 0)
            if total_for_cbg > 0:
                tau_b_alpha = count / total_for_cbg
                cbg_exposure[cbg_int] += tau_b_alpha * proportions
    
    df.apply(accumulate_exposure, axis=1)
    
    # Compute experienced segregation for each CBG.
    experienced_segregation = {}
    for cbg, exposure in cbg_exposure.items():
        experienced_segregation[cbg] = constant_factor * np.sum(np.abs(exposure - (1.0 / n)))
    
    # Map experienced segregation back to the cbg_gdf.
    # Ensure the "cbg" field is of the same type as the keys in experienced_segregation.
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    cbg_gdf["Si_race"] = cbg_gdf["cbg"].map(experienced_segregation)
    
    return df, cbg_gdf
