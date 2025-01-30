def process_mp(
    df,
    placekeys_to_drop,
    categories_to_drop,
    malls_subcategory_column="SUB_CATEGORY",
    placekey_column="PLACEKEY",
    parent_place_column="PARENT_PLACEKEY",
    location_name_column="LOCATION_NAME",
    city_column="CITY"
):
    """
    Processes a DataFrame by filtering and cleaning POI (Point of Interest) data to prepare for analysis.

    The function performs the following steps:
    1. Filters POIs to include only those in the specified valid cities.
    2. Drops POIs based on predefined placekeys and category exclusions.
    3. Replaces latitude and longitude coordinates for specific placekeys.
    4. Removes POIs with visit counts below the 25th percentile.
    5. Identifies parent placekeys and assigns flags for parent and shared polygon POIs.
    6. Maps parent location names from parent POIs.
    7. Creates group identifiers for POIs with shared parent locations and similar visit counts.
    8. Assigns a historical flag for POIs that have closed based on `CLOSED_ON` values.
    9. Sorts the final dataset by visit counts and resets indices.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing POI data.
        placekeys_to_drop (list): A list of PLACEKEYs to remove from the DataFrame.
        categories_to_drop (list): A list of category values in the malls subcategory column to exclude.
        malls_subcategory_column (str, optional): The column name for POI subcategories. Defaults to "SUB_CATEGORY".
        placekey_column (str, optional): The column name for unique POI identifiers. Defaults to "PLACEKEY".
        parent_place_column (str, optional): The column name for parent POIs. Defaults to "PARENT_PLACEKEY".
        location_name_column (str, optional): The column name for POI location names. Defaults to "LOCATION_NAME".
        city_column (str, optional): The column name for city names. Defaults to "CITY".

    Returns:
        tuple:
            - df_filtered (pd.DataFrame): The cleaned and filtered POI dataset.
            - parent_placekey_dfs (pd.DataFrame): Subset of parent POIs identified in the dataset.
            - df_unidentified_parent_locations (pd.DataFrame): POIs where parent locations could not be determined.
    """
    # Step 1: Check if each row is a unique placekey
  #  print("Note: each row is a unique Point of Interest (POI) \nand every POI has a unique safegraph_place_id.\n")
  #  print("number of rows:")
  #  print(df.shape[0]) 
  #  print("number of unique PLACEKEYs:")
  #  print(df.PLACEKEY.unique().shape[0]) 
    print(f"every row is a unique PLACEKEY:{df.shape[0] == df.PLACEKEY.unique().shape[0]}")
    print('\n')
    # Step 2: Ensure df contains relevant columns
    mp_cols = ["PLACEKEY", "PARENT_PLACEKEY", "SAFEGRAPH_BRAND_IDS", "LOCATION_NAME", "BRANDS",
      "STORE_ID", "TOP_CATEGORY", "SUB_CATEGORY", "NAICS_CODE", "LATITUDE", "LONGITUDE",
      "OPEN_HOURS", "CATEGORY_TAGS", "VISITOR_HOME_CBGS", "BUCKETED_DWELL_TIMES",
      "RELATED_SAME_DAY_BRAND", "RELATED_SAME_MONTH_BRAND", "POPULARITY_BY_DAY",
      "POPULARITY_BY_HOUR", "Weighted Median DISTANCE_FROM_HOME",
      "Weighted Median MEDIAN_DWELL", "Median RAW_VISIT_COUNTS", "Median RAW_VISITOR_COUNTS",
      "STREET_ADDRESS","CITY","REGION","POSTAL_CODE","TRACTCE","GEOID","OPENED_ON","CLOSED_ON","TRACKING_CLOSED_SINCE","geometry"]
    df=df[mp_cols].copy()
    # Step 3: Filter rows where CITY is in the valid cities
    print("Filtering POIs outside BJCTA service area...")
    valid_cities = ["Birmingham", "Bessemer", "Homewood", "Mountain Brook","Midfield", "Center Point", "Hoover", "Vestavia Hills","Tarrant", "Fairfield"]
    df_filtered = df[df[city_column].isin(valid_cities)]
    print(f"Number of PLACEKEYs after removing cities outside of service area:{df_filtered['PLACEKEY'].nunique()}")

    # Step 4: Drop rows where PLACEKEY is in the placekeys_to_drop list
    print("Dropping placekeys in placekeys_to_drop...")
    df_filtered = df_filtered[~df_filtered[placekey_column].isin(placekeys_to_drop)]
    print(f"Number of PLACEKEYs after removing placekeys in placekeys_to_remove:{df_filtered['PLACEKEY'].nunique()}")

    # Step 5: Drop rows where PLACEKEY category is in the categories_to_drop list
    print("Dropping POIs in categories_to_drop...")
    df_filtered = df_filtered[~df_filtered[malls_subcategory_column].isin(categories_to_drop)]
    print(f"Number of PLACEKEYs after removing placekeys in categories_to_drop:{df_filtered['PLACEKEY'].nunique()}")

    # Step 6: Replace the coordinates of PLACEKEYs in placekeys_to_replace with corrected coordinates
    placekeys_to_replace = {
      "zzy-222@8gk-t8n-9mk": (33.421484937331066, -86.68563073547514),
      "zzy-223@8gk-tv9-vvf": (33.506653664916534, -86.80302204440463),
      "zzy-224@8gk-tv9-vvf": (33.508096928154565, -86.80029315769171),
      "zzy-225@8gk-tv9-vvf": (33.506221877183414, -86.80464203018646),
      "zzy-22d@8gk-tv9-xdv": (33.504135788629874, -86.80153328062198),
      "zzy-22f@8gk-tv9-xdv": (33.5082618758929, -86.80030021854165),
      "zzy-22g@8gk-tv9-xdv": (33.50481534223161, -86.79950904977868),
      "zzy-22m@8gk-tv9-xdv": (33.49666168526395, -86.80949494737243),
      "zzy-22n@8gk-tv9-xdv": (33.508796356399344, -86.79861553572496)}

    print("Correcting lat long coords of placekeys in placekeys_to_replace...")
    for placekey, (lat, lon) in placekeys_to_replace.items():
        df_filtered.loc[df_filtered["PLACEKEY"] == placekey, ["LATITUDE", "LONGITUDE"]] = lat, lon
    
    # Step 7: Remove placekeys with visit counts below the 25th percentile
    print("Filtering POIs with visit counts below the 25th percentile:")
    initial_row_count = df_filtered.shape[0]
    threshold = df_filtered['Median RAW_VISITOR_COUNTS'].quantile(0.25)
    df_filtered = df_filtered[df_filtered['Median RAW_VISIT_COUNTS'] >= threshold]
    df_filtered=df_filtered.reset_index(drop=True)
    final_row_count = df_filtered.shape[0]
    removed_rows=initial_row_count - final_row_count
    print(f"Number of rows removed after outlier filtering: {removed_rows}")
    print(f"Final number of POIs in dataset: {df_filtered['PLACEKEY'].nunique()}")

    # Step 8: Create parent_placekeys_df
    print("Identifying parent placekeys...")
    parent_placekeys_set = set(df_filtered["PARENT_PLACEKEY"].dropna().unique())
    parent_placekey_dfs = df_filtered[df_filtered["PLACEKEY"].isin(parent_placekeys_set)].copy()
    parent_placekey_set = set(parent_placekey_dfs["PLACEKEY"])
    df_filtered['parent_flag'] = df_filtered['PLACEKEY'].apply(lambda pk: 1 if pk in parent_placekey_set else 0)
    parent_counts=(len(df_filtered[df_filtered['parent_flag'] == 1])/len(df_filtered))*100
    print(f"Percentage of parent placekeys:{parent_counts}")

    print("Creating flag columns...")
    # Step 9: Create flag columns
    # a) Create shared_polygon_flag column
    df_filtered['shared_polygon_flag'] = df_filtered['PARENT_PLACEKEY'].notnull().astype(int)
    shared_polygon_count = df_filtered['shared_polygon_flag'].sum()
    print(f"Percentage of POIs with shared polygons:{(shared_polygon_count/df_filtered['PLACEKEY'].nunique())*100}")

    # b) Create parent_location_name column by referencing parent_placekeys_df
    parent_placekeys_mapping = parent_placekey_dfs.set_index('PLACEKEY')['LOCATION_NAME']
    df_filtered['parent_location_name'] = df_filtered['PARENT_PLACEKEY'].map(parent_placekeys_mapping)
    df_with_parent_location = df_filtered[df_filtered['parent_location_name'].notnull()]
    df_unidentified_parent_locations = df_filtered[(df_filtered['parent_location_name'].isnull()) & (df_filtered['shared_polygon_flag'] == 1)]
    identified_parent_locations = df_filtered[(df_filtered['parent_location_name'].notnull()) & (df_filtered['shared_polygon_flag'] == 1)].shape[0]
    if shared_polygon_count > 0:
        parent_location_percentage = (identified_parent_locations / shared_polygon_count) * 100
    else:
        parent_location_percentage = 0  
    print(f"Percentage of Identified Parent Locations: {parent_location_percentage:.2f}%")
    print(f"Number of Unidentified parent locations:{df_unidentified_parent_locations['PLACEKEY'].nunique()}")

    print("Sample of identified location names:")
    print(df_with_parent_location[['PLACEKEY','parent_location_name']].sample(3))

    # c) Create group_flag column
    df_filtered['group_flag'] = (df_filtered.groupby(['PARENT_PLACEKEY', 'Median RAW_VISIT_COUNTS']).ngroup())
    df_filtered['group_flag'] = df_filtered['group_flag'].duplicated(keep=False).astype(int)
    # d) Create `group_id` column
    def create_group_id(group):
        if len(group) > 1:
            first_index = group.index.min()
            last_index = group.index.max()
            parent_location_name = group['parent_location_name'].iloc[0]
            return f"{parent_location_name}_{first_index}_{last_index}"
        return None
    group_ids = df_filtered.groupby(['PARENT_PLACEKEY', 'Median RAW_VISIT_COUNTS']).apply(create_group_id)
    df_filtered['group_id'] = df_filtered.apply(lambda row: group_ids.get((row['PARENT_PLACEKEY'], row['Median RAW_VISIT_COUNTS'])), axis=1)
    num_groups = df_filtered['group_id'].nunique()
    print(f"Number of groups identified: {num_groups}")
    print("Group id sample:")
    print(df_filtered[['PLACEKEY','parent_location_name','group_id','Median RAW_VISIT_COUNTS']].sample(5))

    # e) Create historical flag column
    historical_closed_on_vals = ["1900-01", "2023-05", "2023-08", "2023-02", "2023-03", "2023-10",
            "2023-07", "2023-06", "2023-01", "2023-09", "2023-04", "2023-11",
            "2022-12", "2023-12", "2022-03", "2022-02", "2021-03", "2022-10",
            "2022-08", "2022-06", "2020-01", "2022-09", "2022-04", "2021-07",
            "2020-02", "2021-05", "2021-12", "2019-09", "2021-01", "2020-09","2021-11"]
    mask = df_filtered['CLOSED_ON'].isin(historical_closed_on_vals)
    df_filtered['historical_flag'] = 0
    df_filtered.loc[mask, 'historical_flag'] = 1
    closed_count = df_filtered['historical_flag'].sum()
    print(f"Percentage of closed POIs:{(closed_count/df_filtered['PLACEKEY'].nunique())*100}")
    # Step 10: Sort and reset index
    df_filtered = df_filtered.sort_values(by="Median RAW_VISIT_COUNTS", ascending=False)
    parent_placekey_dfs = parent_placekey_dfs.sort_values(by="Median RAW_VISIT_COUNTS", ascending=False)
    parent_placekey_dfs=parent_placekey_dfs.reset_index(drop=True)
    df_filtered=df_filtered.reset_index(drop=True)
    print("Monthly patterns processing done!")

    return df_filtered,parent_placekey_dfs,df_unidentified_parent_locations
