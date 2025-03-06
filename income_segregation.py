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

def calculate_income_quintiles_cbsa(total_households, income_data, cbg_gdf):
    income_data['Households'] = (income_data['Percentage'] / 100) * total_households
    income_data['Cumulative Households'] = income_data['Households'].cumsum()
    quintiles = [0, total_households * 0.2, total_households * 0.4,total_households * 0.6, total_households * 0.8, total_households]
    income_data['cbsa_income_quintile'] = pd.cut(income_data['Cumulative Households'], bins=quintiles,
                                            labels=['low', 'lower_middle','middle', 'upper_middle', 'high'], include_lowest=True)
    cbg_gdf['income_quintile'] = pd.qcut(cbg_gdf['median_hh_income'], 5, labels=['low', 'lower_middle','middle', 'upper_middle', 'high'])

    return cbg_gdf

"""
Module: income_segregation
Repository: comradeshawty/patterns
Description: Data processing routines for income segregation analysis.
"""
import numpy as np
import pandas as pd
def compute_residential_income_segregation(cbg_gdf):
    """
    Computes S_res (residential income segregation) for each CBG using
    fixed bracket definitions for 'low', 'lower_middle', 'upper_middle', 'high'.
    
    Each row of cbg_gdf is expected to have columns representing the
    number of households in these brackets:
      'less_than_10k', '10k_15k', '15k_to_20k', '20k_to_25k', '25k_to_30k',
      '30k_to_35k', '35k_to_40k', '40k_to_45k', '45k_to_50k', '50k_to_60k',
      '60k_to_75k', '75k_to_100k', '100k_to_125k', '125k_to_150k',
      '150k_to_200k', '200k_or_more'.

    The category definitions (from bracket to income group) are:
      low =  { 'less_than_10k', '10k_15k', '15k_to_20k' }
      lower_middle = { '20k_to_25k', '25k_to_30k', '30k_to_35k', '35k_to_40k', '40k_to_45k', '45k_to_50k' }
      upper_middle = { '50k_to_60k', '60k_to_75k', '75k_to_100k' }
      high = { '100k_to_125k', '125k_to_150k', '150k_to_200k', '200k_or_more' }

    We define the segregation measure using four quartiles:
      S_res = (2/3) * sum( | proportion_in_quartile - 0.25 | ) over all quartiles.

    Parameters
    ----------
    cbg_gdf : DataFrame (or GeoDataFrame)
        Must have the columns for each bracket listed above.
    
    Returns
    -------
    cbg_gdf : DataFrame (copy)
        A modified copy of the original with an added column "S_res" that holds
        the computed segregation measure per CBG.
    """

    # Mapping from bracket columns to quartile category
    bracket_map = {
        'low': ['less_than_10k', '10k_15k', '15k_to_20k'],
        'lower_middle': ['20k_to_25k', '25k_to_30k', '30k_to_35k','35k_to_40k', '40k_to_45k', '45k_to_50k'],
        'upper_middle': ['50k_to_60k', '60k_to_75k', '75k_to_100k'],
        'high': ['100k_to_125k', '125k_to_150k','150k_to_200k', '200k_or_more']}

    def compute_s_res_for_row(row):
        q_pops = []
        total_pop = 0.0
        for category in ['low', 'lower_middle', 'upper_middle', 'high']:
            cat_sum = 0.0
            for bracket_col in bracket_map[category]:
                cat_sum += float(row.get(bracket_col, 0.0))
            q_pops.append(cat_sum)
            total_pop += cat_sum

        if total_pop == 0:
            return np.nan

        proportions = [pop / total_pop for pop in q_pops]
        s_res = (2.0 / 3.0) * sum(abs(p - 0.25) for p in proportions)
        return s_res

    new_gdf = cbg_gdf.copy()
    new_gdf["S_res"] = new_gdf.apply(compute_s_res_for_row, axis=1)
    new_gdf.dropna(subset=['Si','S_res'],inplace=True,ignore_index=True)
    return new_gdf
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
    
    Additionally, this function adds a column to df named 'quartile_proportions'
    which contains, for each POI, a dictionary with the proportions of visitors from each income quartile.
    The dictionary is formatted as: {'low': prop, 'lower_middle': prop, 'upper_middle': prop, 'high': prop}
    
    Parameters:
      df     : DataFrame that includes the 'adjusted_cbg_visitors' column containing the processed visitor counts.
      cbg_gdf: GeoDataFrame with CBG information and an 'income_quantile' column; 
               the income labels for CBGs are in {"low", "lower_middle", "upper_middle", "high"}.
    
    Returns:
      Tuple: (df with added columns 'income_segregation' and 'quartile_proportions', 
              updated cbg_gdf with column 'experienced_income_segregation')
    """
    income_label_to_quartile = {"low": 1, "lower_middle": 2, "upper_middle": 3, "high": 4}
    
    # Ensure CBGs in cbg_gdf are correctly formatted.
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    
    # Create mapping: CBG → Income Quartile.
    cbg_income_map = cbg_gdf.set_index("cbg")["income_quantile"].map(income_label_to_quartile).to_dict()
    
    def segregation_from_dict(visitor_dict):
        """
        Compute the POI-level income segregation score along with the distribution of visitor proportions by quartile.
        Returns a tuple: (segregation score, proportions array, proportions dictionary, total visitors at the POI).
        """
        quartile_counts = np.zeros(4, dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            quartile = cbg_income_map.get(cbg_int, None)
            if quartile is not None:
                quartile_counts[quartile - 1] += count  # store in 0-based index.
        
        total = quartile_counts.sum()
        if total == 0:
            default_proportions = {"low": 0, "lower_middle": 0, "upper_middle": 0, "high": 0}
            return np.nan, None, default_proportions, total
        
        proportions = quartile_counts / total  # Fraction for each quartile.
        segregation = (2/3) * np.sum(np.abs(proportions - 0.25))
        proportions_dict = {
            "low": proportions[0],
            "lower_middle": proportions[1],
            "upper_middle": proportions[2],
            "high": proportions[3]
        }
        return segregation, proportions, proportions_dict, total

    # Compute POI-level segregation scores and prepare for CBG-level aggregation.
    poi_segregation_scores = []         # Holds segregation score for each POI.
    quartile_proportions_list = []        # Holds the proportions dictionary for each POI.
    
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
        segregation_value, proportions, proportions_dict, total_alpha = segregation_from_dict(visitor_dict)
        
        poi_segregation_scores.append(segregation_value)
        quartile_proportions_list.append(proportions_dict)
        
        # For each CBG present in the POI, compute its weight for this POI and add its exposure contribution.
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            global_total = total_visits_per_cbg.get(cbg_int, 0)
            if global_total == 0 or proportions is None:
                continue
            # τ₍b,α₎: fraction of the CBG's visits that occur at this POI.
            tau_b_alpha = count / global_total
            contribution = tau_b_alpha * proportions
            if cbg_int in cbg_exposure:
                cbg_exposure[cbg_int] += contribution
            else:
                cbg_exposure[cbg_int] = np.array(contribution, dtype=float)
                
    # Add the computed POI-level income segregation scores and quartile proportions as new columns in df.
    df = df.copy()
    df['Sα'] = poi_segregation_scores
    df['quartile_proportions'] = quartile_proportions_list

    # Compute experienced income segregation for each CBG and add it to the cbg_gdf.
    experienced_income_segregation = {}
    for cbg, exposure_array in cbg_exposure.items():
        exposure_sum = exposure_array.sum()
        if exposure_sum == 0:
            experienced_income_segregation[cbg] = np.nan
        else:
            normalized_exposure = exposure_array / exposure_sum
            experienced_income_segregation[cbg] = (2/3) * np.sum(np.abs(normalized_exposure - 0.25))
            
    cbg_gdf = cbg_gdf.copy()
    cbg_gdf['Si'] = cbg_gdf['cbg'].map(experienced_income_segregation)
    
    return df, cbg_gdf


def compute_quintile_income_segregation(df, cbg_gdf):
    """
    For each POI (i.e. each row in df), compute an income segregation score
    based on the processed visitor counts, and compute experienced income segregation for each CBG.
    
    The function maps each CBG (key in the processed dict) to an income quintile
    (using cbg_income_map, which maps CBG (as int) to a quartile in {1,2,3,4,5}),
    sums the visitor counts by quartile per POI, and then calculates the POI segregation
    measure as:
    
         segregation = (5/8) * sum(|proportion - 0.2|)
    
    where the proportion is the fraction of visitors from each quartile at that POI.
    
    In addition, we compute the experienced income segregation for each CBG.
    For each POI (denoted by α):
      - τ₍q,α₎: the proportion of time at place α spent by income group q.
      - For each CBG b visiting that POI, τ₍b,α₎ is calculated as the count for b at α divided by
        the total visitors at α, but then normalized across all POIs (i.e. divided by the CBG's global total visits).
    Then, for each CBG, the relative exposure is:
         τ₍b,q₎ = Σ₍α visited by b₎ (τ₍b,α₎ * τ₍q,α₎)
    and the experienced income segregation measure is:
         Sᵢ = (5/8) * Σ₍q=1...5₎ |τ₍b,q₎ − 0.2|
    
    Additionally, this function adds a column to df named 'quintile_proportions'
    which contains, for each POI, a dictionary with the proportions of visitors from each income quintile.
    The dictionary is formatted as: {'low': prop, 'lower_middle': prop, 'middle':prop,'upper_middle': prop, 'high': prop}
    
    Parameters:
      df     : DataFrame that includes the 'adjusted_cbg_visitors' column containing the processed visitor counts.
      cbg_gdf: GeoDataFrame with CBG information and an 'income_quintile' column; 
               the income labels for CBGs are in {"low", "lower_middle", "middle","upper_middle", "high"}.
    
    Returns:
      Tuple: (df with added columns 'quintile_income_segregation' and 'quintile_proportions', 
              updated cbg_gdf with column 'experienced_income_segregation')
    """
    income_label_to_quintile = {"low": 1, "lower_middle": 2,"middle":3, "upper_middle": 4, "high": 5}
    
    # Ensure CBGs in cbg_gdf are correctly formatted.
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    
    # Create mapping: CBG → Income Quartile.
    cbg_income_map = cbg_gdf.set_index("cbg")["income_quintile"].map(income_label_to_quintile).to_dict()
    
    def segregation_from_dict(visitor_dict):
        """
        Compute the POI-level income segregation score along with the distribution of visitor proportions by quartile.
        Returns a tuple: (segregation score, proportions array, proportions dictionary, total visitors at the POI).
        """
        quintile_counts = np.zeros(5, dtype=float)
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            quintile = cbg_income_map.get(cbg_int, None)
            if quintile is not None:
                quintile_counts[quintile - 1] += count  # store in 0-based index.
        
        total = quintile_counts.sum()
        if total == 0:
            default_proportions = {"low": 0, "lower_middle": 0, "middle":0,"upper_middle": 0, "high": 0}
            return np.nan, None, default_proportions, total
        
        proportions = quintile_counts / total  # Fraction for each quartile.
        segregation = (5/8) * np.sum(np.abs(proportions - 0.2))
        proportions_dict = {
            "low": proportions[0],
            "lower_middle": proportions[1],
            "middle":proportions[2],
            "upper_middle": proportions[3],
            "high": proportions[4]
        }
        return segregation, proportions, proportions_dict, total

    # Compute POI-level segregation scores and prepare for CBG-level aggregation.
    poi_segregation_scores = []         # Holds segregation score for each POI.
    quintile_proportions_list = []        # Holds the proportions dictionary for each POI.
    
    # Dictionary to hold total visits per CBG across all POIs.
    total_visits_per_cbg = {}
    # Dictionary to accumulate exposure contributions per CBG; key: cbg, value: np.array (length 5)
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
        segregation_value, proportions, proportions_dict, total_alpha = segregation_from_dict(visitor_dict)
        
        poi_segregation_scores.append(segregation_value)
        quintile_proportions_list.append(proportions_dict)
        
        # For each CBG present in the POI, compute its weight for this POI and add its exposure contribution.
        for cbg, count in visitor_dict.items():
            try:
                cbg_int = int(cbg)
            except Exception:
                continue
            global_total = total_visits_per_cbg.get(cbg_int, 0)
            if global_total == 0 or proportions is None:
                continue
            # τ₍b,α₎: fraction of the CBG's visits that occur at this POI.
            tau_b_alpha = count / global_total
            contribution = tau_b_alpha * proportions
            if cbg_int in cbg_exposure:
                cbg_exposure[cbg_int] += contribution
            else:
                cbg_exposure[cbg_int] = np.array(contribution, dtype=float)
                
    # Add the computed POI-level income segregation scores and quartile proportions as new columns in df.
    df = df.copy()
    df['Sα_q'] = poi_segregation_scores
    df['quintile_proportions'] = quintile_proportions_list

    # Compute experienced income segregation for each CBG and add it to the cbg_gdf.
    experienced_income_segregation = {}
    for cbg, exposure_array in cbg_exposure.items():
        exposure_sum = exposure_array.sum()
        if exposure_sum == 0:
            experienced_income_segregation[cbg] = np.nan
        else:
            normalized_exposure = exposure_array / exposure_sum
            experienced_income_segregation[cbg] = (5/8) * np.sum(np.abs(normalized_exposure - 0.2))
            
    cbg_gdf = cbg_gdf.copy()
    cbg_gdf['Si_q'] = cbg_gdf['cbg'].map(experienced_income_segregation)
    
    return df, cbg_gdf
