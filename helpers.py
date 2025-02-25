import pandas as pd
import geopandas as gpd
import numpy as np
import json
from cbgs_processor import process_cbg_data_v2,compute_weighted_and_simple_median_distance

def load_data():
    mp=pd.read_csv('/content/drive/MyDrive/data/mp.csv')
    mp=mp[mp['PLACEKEY']!='222-222@8gk-tdk-q2k']
    cbg_gdf=gpd.read_file('/content/drive/MyDrive/data/brh_cbg.geojson')
    cbg_gdf['cbg'] = cbg_gdf['cbg'].astype(str).str.zfill(12).astype(int)
    mp=process_cbg_data_v2(mp, cbg_gdf, 'RAW_VISITOR_COUNTS', 'VISITOR_HOME_CBGS')
    mp=get_time_buckets(mp)
    mp=add_raw_visit_counts(mp)
    mp = propagate_month_adjustments_to_daily_counts(mp)
    mp,removed_df=merge_duplicate_pois(mp)  
    return mp, cbg_gdf
def add_raw_visit_counts(mp):
    mp.loc[:, 'popularity_by_hour_sum'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(literal_eval(x)) if isinstance(x, str) else sum(x))
    mp.loc[:, 'visits_by_day_sum'] = mp['VISITS_BY_DAY'].apply(lambda x: sum(literal_eval(x)) if isinstance(x, str) else sum(x))
    mp.loc[:,'visitor_counts_cbg_scaled'] = mp["adjusted_cbg_visitors"].apply(lambda x: sum(x.values()))
    mp['visits_per_visitor']=mp['RAW_VISIT_COUNTS'] / mp['RAW_VISITOR_COUNTS']
    mp['visit_counts_cbg_scaled'] = mp['visitor_counts_cbg_scaled'] * mp['visits_per_visitor']
    return mp

def propagate_month_adjustments_to_daily_counts(mp):    
    def adjust_visits(visits, scale_factor):
        if isinstance(visits, str):
            visits = literal_eval(visits)
        return [round(v * scale_factor) for v in visits]
    scale_factors = mp.apply(lambda row: row["visit_counts_cbg_scaled"] / row["RAW_VISIT_COUNTS"] 
                             if row["RAW_VISIT_COUNTS"] > 0 else 1, axis=1)
    mp["adjusted_visits_by_day"] = [adjust_visits(v, s) for v, s in zip(mp["VISITS_BY_DAY"], scale_factors)]
    mp["adjusted_popularity_by_day"] = [adjust_visits(v, s) for v, s in zip(mp["POPULARITY_BY_HOUR"], scale_factors)]
    mp["adjusted_raw_visit_counts"] = mp["adjusted_visits_by_day"].apply(sum)    
    return mp

def label_outliers_iqr(df, column_to_filter='adjusted_raw_visit_counts', k=2, outlier_col='outlier', verbose=False):
  # Copied from: https://colab.research.google.com/drive/1LwQNJp9qI0abUzd5jYwT_xJTHJ98iZsD?authuser=1#scrollTo=awAIJqIMxAQk
  df = df.copy()
  quartiles = df[column_to_filter].quantile([.25, .75])
  iqr = np.abs(quartiles.iloc[1] - quartiles.iloc[0])
  tolerable_range = pd.Series([quartiles.iloc[0] - k*iqr - 0.01, quartiles.iloc[1] + k*iqr + 0.01 ])
  df[outlier_col] = (df[column_to_filter].isna() | df[column_to_filter].clip(*tolerable_range).isin(tolerable_range)) # clip sets values outside the range to be equal to the boundaries

  if(verbose): 
    num_nullvalues = df[column_to_filter].isna().sum()
    num_outliers = df[outlier_col].sum() - num_nullvalues
    print("Found {0} null values and {1} outliers out of {2} records for the column {3}.".format(num_nullvalues, num_outliers, df.shape[0],column_to_filter))
  return(df)

def drop_outliers_by_row(df, column_to_filter, k=1.5, verbose=True):
    if(verbose): print("Running drop_outliers()")
    df_out = df.copy()
    df_out = label_outliers_iqr(df_out, column_to_filter=column_to_filter, k=k, verbose=verbose)
    df_out = df_out[~df_out['outlier']]
    return(df_out)

def merge_duplicate_pois(mp, save_path="/content/drive/MyDrive/data/removed_duplicate_pois.csv"):
    category_list=['Accounting, Tax Preparation, Bookkeeping, and Payroll Services',
       'Depository Credit Intermediation',
       'Agencies, Brokerages, and Other Insurance Related Activities',
       'Insurance Carriers', 'Legal Services',
       'Activities Related to Real Estate',
       'Activities Related to Credit Intermediation',
       'Offices of Real Estate Agents and Brokers',
       'Nondepository Credit Intermediation',
       'Other Financial Investment Activities','Offices of Other Health Practitioners', 'Offices of Physicians','General Medical and Surgical Hospitals',
      'Outpatient Care Centers', 'Medical and Diagnostic Laboratories',
       'Specialty (except Psychiatric and Substance Abuse) Hospitals',
       'Other Ambulatory Health Care Services', 'Offices of Dentists',
       'All Other Ambulatory Health Care Services']
    mp["POLYGON_ID"] = mp["PLACEKEY"].str.split("@").str[1]
    mp["VISITOR_HOME_CBGS_STR"] = mp["VISITOR_HOME_CBGS"].astype(str)
    mp=mp.sort_values(by='RAW_VISIT_COUNTS',ascending=False)
    grouped = mp.groupby(["POLYGON_ID", "VISITOR_HOME_CBGS_STR"])
    merged_rows = []
    removed_rows = []
    for (visit_count, home_cbgs), group in grouped:
        if len(group) >= 2:  # Only process sequences with 5+ duplicates
            first_row = group.iloc[0].copy()  # Keep the first row's values

            # Determine new LOCATION_NAME (Most common TOP_CATEGORY - Most common STREET ADDRESS)
            most_common_top_category = Counter(group["TOP_CATEGORY"]).most_common(1)[0][0]
            most_common_sub_category = Counter(group["SUB_CATEGORY"]).most_common(1)[0][0]
            most_common_tag_category = Counter(group["CATEGORY_TAGS"]).most_common(1)[0][0]

            most_common_address = Counter(group["STREET_ADDRESS"]).most_common(1)[0][0]
            first_row["LOCATION_NAME"] = f"{most_common_sub_category} - {most_common_address}"
            first_row["TOP_CATEGORY"]=most_common_top_category
            first_row["SUB_CATEGORY"]=most_common_sub_category
            first_row["CATEGORY_TAGS"]=most_common_tag_category
            # Keep only the first row, discard the rest
            merged_rows.append(first_row)
            removed_rows.append(group.iloc[1:])  # Store dropped rows
    if removed_rows:
        removed_df = pd.concat(removed_rows)
        removed_df.to_csv(save_path, index=False)
    else:
        removed_df = pd.DataFrame()
    cleaned_mp = mp[~mp.index.isin(removed_df.index)]
    cleaned_mp = pd.concat([cleaned_mp, pd.DataFrame(merged_rows)], ignore_index=True)
    cleaned_mp.drop(columns=["POLYGON_ID", "VISITOR_HOME_CBGS_STR"], inplace=True)

    return cleaned_mp, removed_df
