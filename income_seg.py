from os import remove
import pandas as pd
from datetime import datetime
import collections, functools, operator
from geopy.distance import geodesic
from shapely.geometry import Polygon
import json
from shapely.wkt import loads
pd.set_option('display.max_columns', None)
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from ast import literal_eval
import ast
from collections import Counter
import geopandas as gpd
import h5py

def calculate_income_quantiles_cbsa(total_households, income_data, cbg_gdf):
    """
    Calculates income quantiles for a CBSA using total households and income distribution percentages.

    Parameters:
    - total_households (int): Total number of households in the CBSA.
    - income_data (DataFrame): DataFrame with columns 'Income Bracket' and 'Percentage'.

    Returns:
    - DataFrame with income brackets, household counts, cumulative distribution, and income quantiles.
    """
    income_data['Households'] = (income_data['Percentage'] / 100) * total_households
    income_data['Cumulative Households'] = income_data['Households'].cumsum()
    quantiles = [0, total_households * 0.25, total_households * 0.5, total_households * 0.75, total_households]
    income_data['cbsa_income_quantile'] = pd.cut(income_data['Cumulative Households'], bins=quantiles,
                                            labels=['low', 'lower_middle', 'upper_middle', 'high'], include_lowest=True)
    cbg_gdf['income_quantile'] = pd.qcut(cbg_gdf['median_hh_income'], 4, labels=['low', 'lower_middle', 'upper_middle', 'high'])

    return cbg_gdf
def impute_missing_values(gdf, columns_to_impute, tree, coords, k=5):
    """
    Impute missing values in the GeoDataFrame using spatial nearest neighbors with an imputed_flag column.
    """
    # Add an imputed_flag column initialized to 0
    gdf['imputed_flag'] = 0

    for idx, row in gdf.iterrows():
        if pd.isna(row[columns_to_impute]):  # Check if any column needs imputation
            distances, indices = tree.query(coords[idx], k=k)
            valid_neighbors = gdf.iloc[indices][columns_to_impute].dropna()

            if not valid_neighbors.empty:
                imputed_values = valid_neighbors.mean()
                gdf.loc[idx, columns_to_impute] = imputed_values
                gdf.loc[idx, 'imputed_flag'] = 1  # Mark as imputed

    return gdf
