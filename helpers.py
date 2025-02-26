import pandas as pd
import ast
import numpy as np
import geopandas as gpd
from collections import Counter
from scipy.spatial import cKDTree
from ast import literal_eval
from shapely.geometry import Point
import numpy as np
import json
from cbgs_processor import process_cbg_data_v2,compute_weighted_and_simple_median_distance
from recategorize_patterns import SUB_CATEGORY_MAPPING,sub_categories_to_pretty_names,preprocess_mp,update_mp_from_w,merge_duplicate_pois,update_mp,update_category,assign_place_category_and_subcategory,assign_specific_subcategories
def load_data():
    mp=pd.read_csv('/content/drive/MyDrive/data/mp.csv')
    brh_np=pd.read_csv('/content/drive/MyDrive/data/brh_np.csv')

    cbg_gdf=gpd.read_file('/content/drive/MyDrive/data/brh_cbg.geojson')
    #cbg_gdf['cbg'] = cbg_gdf['cbg'].astype(str).str.zfill(12).astype(int)
    mp=preprocess_mp(mp)
    mp=update_mp_from_w(mp)
    mp=process_cbg_data_v2(mp, cbg_gdf, 'RAW_VISITOR_COUNTS', 'VISITOR_HOME_CBGS')
    mp=get_time_buckets(mp)
    mp=add_raw_visit_counts(mp)
    mp = propagate_month_adjustments_to_daily_counts(mp)
    mp,removed_df=merge_duplicate_pois(mp)
    mp=assign_place_category_and_subcategory(mp, SUB_CATEGORY_MAPPING, sub_categories_to_pretty_names)
    mp=assign_specific_subcategories(mp)
    mp=update_mp(mp)
    mp = compute_weighted_and_simple_median_distance(mp, cbg_gdf)
    return mp, cbg_gdf,brh_np
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

def get_time_buckets(mp):
    mp['POPULARITY_BY_HOUR'] = mp['POPULARITY_BY_HOUR'].apply(ast.literal_eval)
    mp['early_morning_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[0:6]))  # 0 AM - 5 AM
    mp['breakfast_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[6:10]))    # 6 AM - 9 AM
    mp['morning_work_hours_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[10:12]))  # 10 AM - 11 AM
    mp['lunch_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[12:14]))      # 12 PM - 1 PM
    mp['afternoon_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[14:17]))  # 2 PM - 4 PM
    mp['dinner_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[17:20]))     # 5 PM - 7 PM
    mp['nighttime_visits'] = mp['POPULARITY_BY_HOUR'].apply(lambda x: sum(x[20:24]))  # 8 PM - 11 PM
    def calculate_work_hours_visitors(row):
      work_hours = list(range(9, 17))  # 7:30 AM (index 7) to 5:30 PM (index 17)
      work_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
      if isinstance(row['POPULARITY_BY_DAY'], str):
          try:
              popularity_by_day = ast.literal_eval(row['POPULARITY_BY_DAY'])
          except Exception:
              return None
      else:
          popularity_by_day = row['POPULARITY_BY_DAY']
      if isinstance(row['POPULARITY_BY_HOUR'], str):
          try:
              popularity_by_hour = ast.literal_eval(row['POPULARITY_BY_HOUR'])
          except Exception:
              return None
      else:
          popularity_by_hour = row['POPULARITY_BY_HOUR']
      if not isinstance(popularity_by_day, dict) or not isinstance(popularity_by_hour, list):
          return None
      work_day_visits = sum([popularity_by_day.get(day, 0) for day in work_days])
      if len(popularity_by_hour) == 24:
          work_hours_visits = sum([popularity_by_hour[hour] for hour in work_hours])
      else:
          return None
      total_weekly_visits = sum(popularity_by_day.values()) if popularity_by_day else 0
      if total_weekly_visits == 0 or sum(popularity_by_hour) == 0:
          return 0
      workday_proportion = work_day_visits / total_weekly_visits
      work_hours_proportion = work_hours_visits / sum(popularity_by_hour)
      work_hours_visitors = work_day_visits * work_hours_proportion
      return work_hours_visitors
    mp['work_hours_visits'] = mp.apply(calculate_work_hours_visitors, axis=1)
    return mp
