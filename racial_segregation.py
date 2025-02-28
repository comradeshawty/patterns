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


def compute_racial_segregation_with_cbsa_baseline(df, cbg_gdf):
    """
    Compute both POI-level and experienced racial segregation metrics using the overall CBSA racial distribution as baseline.
    
    The weighted_means column in df is expected to have keys:
      ['white_pop_frac', 'black_pop_frac', 'asian_pop_frac', 'oth_race_pop_frac', 'two_race_pop_frac', 'hispanic_pop_frac']
      
    We map these keys to the corresponding race names in the CBSA data:
      white_pop_frac      -> white_pop
      black_pop_frac      -> black_pop
      asian_pop_frac      -> asian_pop
      oth_race_pop_frac   -> other_race_pop
      two_race_pop_frac   -> two_race_pop
      hispanic_pop_frac   -> hispanic_pop
      
    For each POI, the racial segregation measure is:
        S_alpha_race = (n / (2*n - 2)) * sum(|τ(r,α) - baseline(r)|)
    where:
      - n is the number of racial groups (here 6),
      - τ(r,α) is the weighted fraction for race r at the POI,
      - baseline(r) is the overall racial fraction from CBSA census data.
    
    For each CBG, the experienced segregation is:
        S_r_experienced = (n / (2*n - 2)) * sum(|τ(b, r) - baseline(r)|)
    where τ(b, r) is the accumulated exposure for race r from all POIs visited by the CBG.
    
    A sanity check ensures that each weighted_means dictionary has the expected keys.
    
    Parameters:
      df : pandas.DataFrame
          DataFrame with each POI as a row. Must have:
            - "weighted_means": dict with keys like ['white_pop_frac', ... 'hispanic_pop_frac'] and corresponding fractions.
            - "adjusted_cbg_visitors": dict mapping CBG identifiers to visitor counts.
      cbg_gdf : pandas.DataFrame or geopandas.GeoDataFrame
          DataFrame with CBG data. Must have a "cbg" column that maps to the keys in visitor counts.
    
    Returns:
      tuple: 
         - df: Input DataFrame with an added "S_alpha_race" column for POI-level segregation.
         - cbg_gdf: The CBG GeoDataFrame with an added "experienced_racial_segregation" column.
    """
    # Mapping from weighted_means keys to CBSA race names.
    mapping = {
        "white_pop_frac": "white_pop",
        "black_pop_frac": "black_pop",
        "asian_pop_frac": "asian_pop",
        "oth_race_pop_frac": "other_race_pop",
        "two_race_pop_frac": "two_race_pop",
        "hispanic_pop_frac": "hispanic_pop"
    }
    
    expected_keys = set(mapping.keys())
    
    # Sanity check on the first row of weighted_means.
    sample_weights = df['weighted_means'].iloc[0]
    provided_keys = set(sample_weights.keys())
    if provided_keys != expected_keys:
        raise ValueError(f"Expected weighted_means keys: {expected_keys}, but got: {provided_keys}")

    # Define the order of races for vectorized operations.
    race_order = [mapping[k] for k in sorted(mapping.keys())]
    weighted_keys_order = sorted(mapping.keys())  # Sorting keys to enforce consistent order.
    n = len(race_order)  # should be 6
    
    # Constant factor in segregation formula.
    constant_factor = n / (2 * n - 2)
    
    # Obtain overall CBSA racial distribution from census data.
    racial_data = get_racial_data()
    tot_pop = racial_data.loc[racial_data['Race'] == 'tot_pop', 'Count'].values[0]
    
    # Build the baseline vector based on the mapping order.
    baseline = []
    for race in race_order:
        race_count_series = racial_data.loc[racial_data['Race'] == race, 'Count']
        if not race_count_series.empty:
            baseline.append(float(race_count_series.values[0]) / tot_pop)
        else:
            baseline.append(0.0)
    baseline = np.array(baseline, dtype=float)
    
    # --- 1. Compute POI-level racial segregation ---
    # Convert weighted_means dictionaries into a DataFrame with columns ordered by weighted_keys_order.
    def convert_weighted_means(weighted_means):
        # Rearrange the values into the order of weighted_keys_order.
        return [weighted_means[k] for k in weighted_keys_order]
    
    weighted_means_df = pd.DataFrame(df['weighted_means'].apply(convert_weighted_means).tolist(),
                                     index=df.index,
                                     columns=weighted_keys_order)
    weighted_means_df = weighted_means_df.fillna(0)
    
    # For POI-level segregation, create an array representing the weighted means in the order matching baseline.
    # However, our baseline is built over race_order, so we must reorder weighted_means columns accordingly.
    # Create a mapping from weighted key to its corresponding CBSA race.
    col_order = []
    for k in weighted_keys_order:
        col_order.append(mapping[k])
    # Create a DataFrame with columns renamed to the CBSA race names.
    weighted_means_df = weighted_means_df.rename(columns=mapping)
    # Reorder columns to match race_order.
    weighted_means_df = weighted_means_df[race_order]
    
    # Compute absolute differences from the baseline.
    abs_diff = (weighted_means_df - baseline).abs()
    df['S_alpha_race_v2'] = constant_factor * abs_diff.sum(axis=1)
    
    # --- 2. Compute experienced racial segregation for each CBG ---
    # Calculate total visits per CBG across all POIs.
    total_visits_per_cbg = {}
    for visitors in df['adjusted_cbg_visitors']:
        for cbg, count in visitors.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            total_visits_per_cbg[cbg_int] = total_visits_per_cbg.get(cbg_int, 0) + count

    # Accumulate each CBG's exposure using a defaultdict.
    cbg_exposure = defaultdict(lambda: np.zeros(n, dtype=float))
    
    # Function to accumulate exposure contributions across POIs.
    def accumulate_exposure(row):
        visitor_dict = row['adjusted_cbg_visitors']
        # Convert weighted_means for this row into a consistent numpy array aligned with race_order.
        proportions = np.array([row["weighted_means"][k] for k in weighted_keys_order], dtype=float)
        # Rename array to match CBSA race names.
        # Build a dictionary mapping CBSA race -> value.
        prop_dict = {mapping[k]: v for k, v in zip(weighted_keys_order, proportions)}
        # Create the exposure vector in the order of race_order.
        proportions_ordered = np.array([prop_dict[race] for race in race_order], dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            total_for_cbg = total_visits_per_cbg.get(cbg_int, 0)
            if total_for_cbg > 0:
                tau_b_alpha = count / total_for_cbg
                cbg_exposure[cbg_int] += tau_b_alpha * proportions_ordered
                
    df.apply(accumulate_exposure, axis=1)
    
    # Compute experienced segregation for each CBG.
    experienced_segregation = {}
    for cbg, exposure in cbg_exposure.items():
        experienced_segregation[cbg] = constant_factor * np.sum(np.abs(exposure - baseline))
    
    # Map experienced segregation back to cbg_gdf.
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    cbg_gdf["Si_race_v2"] = cbg_gdf["cbg"].map(experienced_segregation)
    
    return df, cbg_gdf
