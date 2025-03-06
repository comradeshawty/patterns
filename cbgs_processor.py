import json
import pandas as pd
import geopandas as gpd
import numpy as np
from collections import Counter

def cbg_to_str(df,column):
    df.rename(columns={column:'cbg'},inplace=True)
    df['cbg']=df['cbg'].astype(str).str.lstrip('0')
    return df
    
def fix_malformed_dict_str(s):
    """Fixes malformed dictionary strings by ensuring they end with a closing brace."""
    if not s.endswith("}"):
        last_comma_index = s.rfind(",")
        if last_comma_index != -1:
            s = s[:last_comma_index] + "}"
    return s

def extract_valid_cbgs(visitor_home_cbgs, valid_cbgs):
    """
    Extracts home CBGs from VISITOR_HOME_CBGS, filtering out CBGs not present in valid_cbgs.
    """
    global unparsed_count
    try:
        fixed_str = fix_malformed_dict_str(visitor_home_cbgs)
        visitor_dict = json.loads(fixed_str)  
        cbg_list_visitor = [(int(cbg), int(count)) for cbg, count in visitor_dict.items() if int(cbg) in valid_cbgs]
        return cbg_list_visitor
    except Exception:
        unparsed_count += 1  
        return []  

def adjust_for_visitor_loss(cbg_list_visitor, raw_visitor):
    """
    Adjusts for visitor loss by redistributing unknown visitors across known CBGs.
    """
    sum_known_visitors = sum([x for _, x in cbg_list_visitor])
    if sum_known_visitors == 0:
        return cbg_list_visitor     
    unknown_visitors = raw_visitor - sum_known_visitors
    assigned_cbg_list_visitor = [(cbg, no + no * unknown_visitors / sum_known_visitors) for cbg, no in cbg_list_visitor]
    return assigned_cbg_list_visitor

def adjust_home_cbg_counts_for_coverage(cbg_counter, cbg_coverage, median_coverage, max_upweighting_factor=100):
    """
    Adjusts CBG counts for SafeGraph's differential sampling across CBGs.
    """
    had_to_guess_coverage_value = False
    if not cbg_counter:
        return cbg_counter, had_to_guess_coverage_value   
    new_counter = Counter()
    for cbg, count in cbg_counter.items():
        if cbg not in cbg_coverage or np.isnan(cbg_coverage[cbg]):
            upweighting_factor = 1 / median_coverage
            had_to_guess_coverage_value = True
        else:
            upweighting_factor = 1 / cbg_coverage[cbg]
            if upweighting_factor > max_upweighting_factor:
                upweighting_factor = 1 / median_coverage
                had_to_guess_coverage_value = True        
        new_counter[cbg] = count * upweighting_factor    
    return new_counter, had_to_guess_coverage_value
def process_cbg_data_v2(df_m, cbg_gdf, raw_visitor_col, visitor_home_cbgs_col):
    global unparsed_count
    unparsed_count = 0
    cbg_gdf["cbg"] = cbg_gdf["cbg"].astype(str).str.lstrip("0").astype(int)
    valid_cbgs_set = set(cbg_gdf["cbg"])
    total_population = cbg_gdf["tot_pop"].to_numpy()
    num_devices = cbg_gdf["number_devices_residing"].to_numpy()
    cbg_coverage = num_devices / total_population
    median_coverage = np.nanmedian(cbg_coverage)
    cbg_coverage_dict = dict(zip(cbg_gdf["cbg"], cbg_coverage))

    def process_row(row):
        raw_visitor = row[raw_visitor_col]
        visitor_home_cbgs = row[visitor_home_cbgs_col]
        cbg_list_visitor = extract_valid_cbgs(visitor_home_cbgs, valid_cbgs_set)
        assigned_cbg_list_visitor = adjust_for_visitor_loss(cbg_list_visitor, raw_visitor)
        cbg_counter = Counter(dict(assigned_cbg_list_visitor))
        adjusted_cbg_counter, _ = adjust_home_cbg_counts_for_coverage(cbg_counter, cbg_coverage_dict, median_coverage)
        adjusted_cbg_counter = {cbg: count for cbg, count in adjusted_cbg_counter.items() if cbg in valid_cbgs_set}
        return adjusted_cbg_counter
    df_m["adjusted_cbg_visitors"] = df_m.apply(process_row, axis=1)
    print(f"Number of rows not able to be parsed: {unparsed_count}")
    return df_m

import ast
import numpy as np
import geopandas as gpd
from collections import Counter
from scipy.spatial import cKDTree
from shapely.geometry import Point

def compute_weighted_and_simple_median_distance(mp, cbg_gdf):
    cbg_gdf=cbg_gdf.to_crs(epsg=32616)
    mp=gpd.GeoDataFrame(mp,geometry=gpd.points_from_xy(mp.LONGITUDE,mp.LATITUDE),crs='epsg:4236')
    mp=mp.to_crs(epsg=32616)
    cbg_gdf = cbg_gdf.set_index('cbg')
    cbg_keys = np.array(cbg_gdf.index)
    cbg_centroids = np.array([(geom.centroid.x, geom.centroid.y) for geom in cbg_gdf['geometry']])
    ckdtree = cKDTree(cbg_centroids)
    def weighted_median(values, weights):
        """ Compute the weighted median of a list of values given corresponding weights. """
        sorted_indices = np.argsort(values)
        values_sorted = np.array(values)[sorted_indices]
        weights_sorted = np.array(weights)[sorted_indices]
        cumulative_weight = np.cumsum(weights_sorted)
        cutoff = cumulative_weight[-1] / 2.0
        return values_sorted[np.searchsorted(cumulative_weight, cutoff)]
    def process_row(row):
        """ Process each row to calculate weighted median distance using cKDTree. """
        if isinstance(row['adjusted_cbg_visitors'], str):
            try:
                cbg_visitors = ast.literal_eval(row['adjusted_cbg_visitors'])
            except Exception:
                return None
        else:
            cbg_visitors = row['adjusted_cbg_visitors']
        if not isinstance(cbg_visitors, dict) or not isinstance(row['geometry'], Point):
            return None
        cbg_counter = Counter(cbg_visitors)
        distances = []
        weights = []
        poi_coords = np.array([row['geometry'].x, row['geometry'].y])
        cbg_indices = [np.where(cbg_keys == cbg_key)[0][0] for cbg_key in cbg_counter.keys() if cbg_key in cbg_keys]

        if not cbg_indices:
            return None
        cbg_centroid_coords = cbg_centroids[cbg_indices]
        dists = np.linalg.norm(cbg_centroid_coords - poi_coords, axis=1)

        for idx, cbg_key in enumerate(cbg_counter.keys()):
            if cbg_key in cbg_keys:
                distances.append(dists[idx])
                weights.append(cbg_counter[cbg_key])
        weighted_med = weighted_median(distances, weights) if distances and weights else None

        simple_med = np.median(distances) if distances else None
        return weighted_med, simple_med
        
    mp[['weighted_median_distance_from_home', 'median_distance_from_home']] = mp.apply(lambda row: process_row(row), axis=1, result_type="expand")
    return mp

def normalize_cbg_data(cbg_gdf):
    normalize_by_tot_pop = ["white_age", "black_age", "american_indian_age", "asian_age", "other_race_age","two_race_age", "white_alone_age", "hispanic_age",
                            "hispanic_pop", "non_hispanic_pop","white_pop", "black_pop", "american_indian_pop", "asian_pop", "hawaiian_pop","oth_race_pop",
        "other_race_pop", "two_race_pop", "tot_18_to_65", "below_poverty", "with_disability","working_walked", "working_transit", "commuting_pop","working_pop"]
    normalize_by_pop_in_hh = ["no_veh_renter", "no_veh_owner"]
    for col in normalize_by_tot_pop:
        if col in cbg_gdf.columns:
            cbg_gdf[col + "_frac"] = cbg_gdf[col] / cbg_gdf["tot_pop"].replace(0, pd.NA)
    for col in normalize_by_pop_in_hh:
        if col in cbg_gdf.columns:
            cbg_gdf[col + "_frac"] = cbg_gdf[col] / cbg_gdf["pop_in_hh"].replace(0, pd.NA)
    return cbg_gdf

def compute_weighted_mean(df_m, cbg_gdf, columns,weighted_means_col, adjusted_cbg_visitors_col='adjusted_cbg_visitors'):
    cbg_dict = cbg_gdf.set_index('cbg')[columns].to_dict(orient='index')

    def get_weighted_mean(adjusted_cbg_counter, column):
        weighted_sum = sum(cbg_dict[cbg][column] * count for cbg, count in adjusted_cbg_counter.items() if cbg in cbg_dict)
        total_visitors = sum(adjusted_cbg_counter.values())
        return weighted_sum / total_visitors if total_visitors > 0 else np.nan

    def process_row(row):
        adjusted_cbg_counter = row[adjusted_cbg_visitors_col]
        weighted_means = {}
        for column in columns:
            weighted_means[column] = get_weighted_mean(adjusted_cbg_counter, column)
        return weighted_means

    df_m[weighted_means_col] = df_m.apply(process_row, axis=1)
    return df_m

def compute_racial_weighted_mean(df_m, cbg_gdf, columns = ['white_pop_frac', 'black_pop_frac', 'asian_pop_frac', 'oth_race_pop_frac', 'two_race_pop_frac','hispanic_pop_frac'], adjusted_cbg_visitors_col='adjusted_cbg_visitors',weighted_means_col='racial_weighted_means'):
    cbg_dict = cbg_gdf.set_index('cbg')[columns].to_dict(orient='index')

    def get_weighted_mean(adjusted_cbg_counter, column):
        weighted_sum = sum(cbg_dict[cbg][column] * count for cbg, count in adjusted_cbg_counter.items() if cbg in cbg_dict)
        total_visitors = sum(adjusted_cbg_counter.values())
        return weighted_sum / total_visitors if total_visitors > 0 else np.nan

    def process_row(row):
        adjusted_cbg_counter = row[adjusted_cbg_visitors_col]
        weighted_means = {}
        for column in columns:
            weighted_means[column] = get_weighted_mean(adjusted_cbg_counter, column)
        return weighted_means

    df_m[weighted_means_col] = df_m.apply(process_row, axis=1)
    return df_m
    
def compute_income_weighted_mean(df_m, cbg_gdf, columns=['less_than_10k', '10k_15k', '15k_to_20k', '20k_to_25k', '25k_to_30k', '30k_to_35k', '35k_to_40k', '40k_to_45k', '45k_to_50k', '50k_to_60k', '60k_to_75k', '75k_to_100k', '100k_to_125k', '125k_to_150k', '150k_to_200k', '200k_or_more'],
                                 adjusted_cbg_visitors_col='adjusted_cbg_visitors',weighted_means_col='income_weighted_means'):
    cbg_dict = cbg_gdf.set_index('cbg')[columns].to_dict(orient='index')

    def get_weighted_mean(adjusted_cbg_counter, column):
        weighted_sum = sum(cbg_dict[cbg][column] * count for cbg, count in adjusted_cbg_counter.items() if cbg in cbg_dict)
        total_visitors = sum(adjusted_cbg_counter.values())
        return weighted_sum / total_visitors if total_visitors > 0 else np.nan

    def process_row(row):
        adjusted_cbg_counter = row[adjusted_cbg_visitors_col]
        weighted_means = {}
        for column in columns:
            weighted_means[column] = get_weighted_mean(adjusted_cbg_counter, column)
        return weighted_means

    df_m[weighted_means_col] = df_m.apply(process_row, axis=1)
    return df_m

def compute_exact_visitor_counts(df_m, weighted_means_col, raw_visitor_col, demographic_col, new_col_name):
    df_m[new_col_name] = df_m.apply(lambda row: np.ceil(row[weighted_means_col].get(demographic_col, 0) * row[raw_visitor_col]).astype(int),axis=1)
    return df_m

def compute_racial_visitor_counts(df_m, weighted_means_col, raw_visitor_col):
    racial_columns = ['white_pop_frac', 'black_pop_frac',
                      'asian_pop_frac', 'oth_race_pop_frac',
                      'two_race_pop_frac','hispanic_pop_frac']
    for col in racial_columns:
        new_col_name = col.replace('pop_frac', 'visitors')
        df_m[new_col_name] = df_m.apply(lambda row: np.ceil(row[weighted_means_col].get(col, 0) * row[raw_visitor_col]).astype(int),axis=1)
    return df_m

def build_cbg_race_dict(cbg_gdf):
    """
    Build a dictionary mapping CBG id to an array of racial fractions.
    The racial fraction order is:
      [white, black, american_indian, asian, hispanic, hawaiian, other_race, two_race]
    
    Parameters:
        cbg_gdf (pd.DataFrame): DataFrame with CBG-level racial data.
    
    Returns:
        dict: Mapping from cbg id (string) to numpy array of racial fractions.
    """
    race_cols = ['white_pop_frac', 'black_pop_frac', 'american_indian_pop_frac', 
                 'asian_pop_frac', 'hispanic_pop_frac', 'hawaiian_pop_frac', 
                 'other_race_pop_frac', 'two_race_pop_frac']
    cbg_race_dict = {}
    for row in cbg_gdf.itertuples(index=False):
        cbg_id = str(row.cbg)
        fractions = np.array([getattr(row, col) for col in race_cols], dtype=float)
        cbg_race_dict[cbg_id] = fractions
    return cbg_race_dict
