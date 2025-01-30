import pandas as pd
import placekey as pk

def flag_placekey_geometry_mismatches_with_placekey(df, placekey_column="PLACEKEY", parent_placekey_column="PARENT_PLACEKEY",
                                                    distance_threshold=100, verbose=False):
    """
    Flags mismatches between PLACEKEY and geometry of shared polygon POIs using the Placekey package.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing PLACEKEY and PARENT_PLACEKEY.
        placekey_column (str): Column name for PLACEKEY.
        parent_placekey_column (str): Column name for PARENT_PLACEKEY.
        distance_threshold (float): Maximum allowed distance (in meters) for shared polygon POIs.
        verbose (bool): Whether to print additional information for debugging.

    Returns:
        pd.DataFrame: A DataFrame with flagged mismatches and distance calculations.
    """
    # Extract the `where` parts from PLACEKEY
    df["where_part"] = df[placekey_column].str.split("@").str[1]

    # Create a self-join on PLACEKEY's `where` part to compare distances
    joined = df.merge(
        df,
        on="where_part",
        suffixes=("_child", "_parent")
    )

    # Calculate Placekey distances using the Placekey library
    def calculate_distance(row):
        try:
            return pk.placekey_distance(row[f"{placekey_column}_child"], row[f"{placekey_column}_parent"])
        except Exception as e:
            if verbose:
                print(f"Error calculating distance for {row[f'{placekey_column}_child']} and {row[f'{placekey_column}_parent']}: {e}")
            return None

    joined["placekey_distance"] = joined.apply(calculate_distance, axis=1)

    # Flag mismatches where PLACEKEYs share the same `where` part but are farther apart than the threshold
    joined["mismatch_flag"] = ((joined["placekey_distance"] > distance_threshold) &(joined[f"{placekey_column}_child"] != joined[f"{placekey_column}_parent"]))

    # Filter mismatched rows
    mismatches = joined[joined["mismatch_flag"]].copy()

    # Keep relevant columns and drop duplicates
    mismatches = mismatches[[
        f"{placekey_column}_child", f"{parent_placekey_column}_child",
        f"{placekey_column}_parent", f"{parent_placekey_column}_parent",
        "placekey_distance", "mismatch_flag"
    ]].drop_duplicates()

    return mismatches

import placekey as pk
import geopandas as gpd
import numpy as np
import folium
def draw_placekeys(placekey_values, zoom_start=18, folium_map=None, hex_color='lightblue', weight=2, labels=False):
    """
    :param placekey_values: A list of Placekey strings
    :param zoom_start: Folium zoom level. 18 is suitable for neighboring resolution 10 H3s.
    :folium_map: A Folium map object to add the Placekeys to
    :labels: Whether or not to add labels for Placekeys
    :return: a Folium map object

    """
    geos = [pk.placekey_to_geo(p) for p in placekey_values]
    hexagons = [pk.placekey_to_hex_boundary(p) for p in placekey_values]

    if folium_map is None:
        centroid = np.mean(geos, axis=0)
        folium_map = folium.Map((centroid[0], centroid[1]), zoom_start=zoom_start, tiles='cartodbpositron')

    for h in hexagons:
        folium.Polygon(
            locations=h,
            weight=weight,
            color=hex_color
        ).add_to(folium_map)

    if labels:
        for p, g in zip(placekey_values, geos):
            icon = folium.features.DivIcon(
                icon_size=(120, 36),
                icon_anchor=(60, 15),
                html='<div style="align: center; font-size: 12pt; background-color: lightblue; border-radius: 5px; padding: 2px">{}</div>'.format(p),
            )

            folium.map.Marker(
                [g[0], g[1]],
                icon=icon
            ).add_to(folium_map)

    return folium_map

def process_buildings(
    df,
    valid_cities,
    placekeys_to_drop,
    building_category_mapping,
    placekeys_to_update,
    new_building_value,
    categories_to_drop,
  #  parent_placekeys_df,
    malls_subcategory_column="SUB_CATEGO",
    malls_subcategory_value="Malls",
    building_column="building",
    placekey_column="PLACEKEY",
    parent_place_column="PARENT_PLA",
    location_name_column="LOCATION_N",
    city_column="CITY"
):
    """
    Processes a DataFrame by filtering rows based on valid cities, dropping rows with specific placekeys,
    and assigning building categories based on predefined mappings. Additionally, sets the building column
    to LOCATION_N if building is None and PARENT_PLA is not None.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        valid_cities (list): A list of valid cities to filter rows.
        placekeys_to_drop (list): A list of placekeys to drop from the DataFrame.
        building_category_mapping (dict): A dictionary mapping building categories to building names.
        malls_subcategory_column (str): The column indicating subcategories (default is "SUB_CATEGO").
        malls_subcategory_value (str): The value in the subcategory column representing malls (default is "Malls").
        building_column (str): The column with building names (default is "building").
        placekey_column (str): The column with placekeys (default is "PLACEKEY").
        parent_place_column (str): The column with parent placekeys (default is "PARENT_PLA").
        location_name_column (str): The column with location names (default is "LOCATION_N").
        city_column (str): The column with city names (default is "CITY").

    Returns:
        pd.DataFrame: A processed DataFrame with the building categories assigned.
    """
    # Step 1: Filter rows where CITY is in the valid cities
    df_filtered = df[df[city_column].isin(valid_cities)]

    # Step 2: Drop rows where PLACEKEY is in the placekeys_to_drop list
    df_filtered = df_filtered[~df_filtered[placekey_column].isin(placekeys_to_drop)]
    df_filtered = df_filtered[~df_filtered[malls_subcategory_column].isin(categories_to_drop)]

    # Step 3: Reset index after dropping rows
    df_filtered = df_filtered.reset_index(drop=True)

    # Step 4: Create or ensure a building_category column exists
    if "building_category" not in df_filtered.columns:
        df_filtered["building_category"] = None

    # Step 5: Assign building categories based on the building_category_mapping
    def assign_building_category(building_name):
        for category, buildings in building_category_mapping.items():
            if building_name in buildings:
                return category
        return "Other"  # Default if no match

    df_filtered["building_category"] = df_filtered[building_column].apply(assign_building_category)

    # Step 6: Assign "Shopping Center/Mall" where SUB_CATEGO equals "Malls"
    df_filtered.loc[
        df_filtered[malls_subcategory_column] == malls_subcategory_value, "building_category"
    ] = "Shopping Center/Mall"

    # Step 7: Update the building column for the specified PLACEKEY values
    df_filtered.loc[df_filtered[placekey_column].isin(placekeys_to_update), building_column] = new_building_value
    parent_placekeys_set = set(df_filtered["PARENT_PLA"].dropna().unique())

    # Step 2: Filter rows where PLACEKEY is in parent_placekeys_set
# Step to replace 'building' with concatenated 'LOCATION_N' and 'STORE_ID' if 'STORE_ID' is not NaN
    parent_placekey_dfs = df_filtered[df_filtered["PLACEKEY"].isin(parent_placekeys_set)].copy()

    # Replace 'building' with concatenated 'LOCATION_N' and 'STORE_ID' if 'STORE_ID' is not NaN
    parent_placekey_dfs.loc[
        parent_placekey_dfs['building'].isna(),
        'building'
    ] = parent_placekey_dfs.loc[
        parent_placekey_dfs['building'].isna(),
        ['LOCATION_N', 'STORE_ID']
    ].apply(lambda row: f"{row['LOCATION_N']}_{int(row['STORE_ID'])}" if pd.notna(row['STORE_ID']) else row['LOCATION_N'], axis=1)

    # Step 8: Set 'building' to 'LOCATION_N' of the PARENT_PLA if building is None
    df_filtered.loc[
        df_filtered[building_column].isna() & df_filtered[parent_place_column].notna(),
        building_column
    ] = df_filtered.loc[
        df_filtered[building_column].isna() & df_filtered[parent_place_column].notna(),
        parent_place_column
    ].map(parent_placekey_dfs.set_index(placekey_column)[location_name_column])

    # Step 9: Set PARENT_PLA to PLACEKEY of the building if building is not None and PARENT_PLA is None
    # Handle duplicate indices by creating a multi-index on parent_placekeys_df
 #   parent_placekeys_mapping = parent_placekeys_df.drop_duplicates(subset=building_column).set_index(building_column)[
 #       placekey_column
#    ]
##    df_filtered.loc[
  #      df_filtered[building_column].notna() & df_filtered[parent_place_column].isna(),
  #      parent_place_column,
 #   ] = df_filtered.loc[
 #       df_filtered[building_column].notna() & df_filtered[parent_place_column].isna(),
#        building_column,
   # ].map(parent_placekeys_mapping)

    # Step 10: Sort by "Median RAW" column (if available)
    if "Median RAW" in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by="Median RAW", ascending=False)
    if "Median RAW" in parent_placekey_dfs.columns:
        parent_placekey_dfs = parent_placekey_dfs.sort_values(by="Median RAW", ascending=False)
    parent_placekey_dfs.drop(columns='OPEN_HOURS',inplace=True)
    df_filtered.drop(columns='OPEN_HOURS',inplace=True)

    parent_placekey_dfs=parent_placekey_dfs.reset_index(drop=True)
    df_filtered=df_filtered.reset_index(drop=True)

    return df_filtered,parent_placekey_dfs
