import pandas as pd
from collections import Counter

def category_processing(df_filtered,dir_path):
    """
    Perform EDA on the given dataframe, including analyzing brands, categories, and shared polygons.

    Parameters:
        df_filtered (pd.DataFrame): The filtered dataframe to analyze.
        dir_path (str): The directory path where the output Excel file will be saved.

    Returns:
        category_df (pd.DataFrame): A pivot table with a multi-index of category type and category, including 
        computed metrics like category count, proportions, parent counts, shared polygon counts, and top category tags.
    """
    category_types = ['TOP_CATEGORY', 'SUB_CATEGORY', 'BRANDS']
    
    all_data = []
    
    for category_type in category_types:
        unique_categories = df_filtered[category_type].dropna().unique()
        
        for category in unique_categories:
            df_cat = df_filtered[df_filtered[category_type] == category]
            total_pois = len(df_filtered)

            # Category count and proportion
            category_count = len(df_cat)
            category_proportion = category_count / total_pois if total_pois else 0

            # Parent count and proportion
            parent_count = df_cat[df_cat['parent_flag'] == 1].shape[0]
            parent_proportion = parent_count / category_count if category_count else 0

            # Shared polygon count and proportion
            shared_polygon_count = df_cat[df_cat['shared_polygon_flag'] == 1].shape[0]
            shared_polygon_proportion = shared_polygon_count / category_count if category_count else 0

            # Extract top 5 category tags
            category_tags = df_cat['CATEGORY_TAGS'].dropna().str.split(',').explode()
            top_category_tags = category_tags.value_counts().head(5).index.tolist()
            top_category_tags = ', '.join(top_category_tags) if top_category_tags else 'NaN'

            # Compute statistics for numerical columns
            numeric_columns = {
                "Median RAW_VISIT_COUNTS": "median_raw_visit_counts",
                "Weighted Median DISTANCE_FROM_HOME": "weighted_median_distance_from_home",
                "Weighted Median MEDIAN_DWELL": "weighted_median_dwell",
                "Median RAW_VISITOR_COUNTS": "median_raw_visitor_counts"
            }
            
            category_stats = df_cat[list(numeric_columns.keys())].describe(percentiles=[0.25, 0.75]).T[['min', '25%', 'mean', '75%', 'max']]

            # Rename columns to the new format
            category_stats.index = [numeric_columns[col] for col in category_stats.index]
            category_stats = category_stats.rename(columns={'min': 'min', '25%': 'p25', 'mean': 'mean', '75%': 'p75', 'max': 'max'})

            # Flatten statistics into a single row
            stats_values = category_stats.values.flatten()

            # Append row data
            row_data = [category_type, category, category_count, category_proportion,
                        parent_count, parent_proportion, shared_polygon_count, shared_polygon_proportion, top_category_tags]
            row_data.extend(stats_values)

            all_data.append(row_data)

    # Define column names
    stats_columns = [f"{col}_{stat}" for col in numeric_columns.values() for stat in ['min', 'p25', 'mean', 'p75', 'max']]
    column_names = ['Category_Type', 'Category', 'Category_Count', 'Category_Proportion',
                    'Parent_Count', 'Parent_Proportion', 'Shared_Polygon_Count', 'Shared_Polygon_Proportion',
                    'Top_Category_Tags'] + stats_columns

    # Convert to DataFrame
    category_df = pd.DataFrame(all_data, columns=column_names)

    # Set multi-index
    category_df.set_index(['Category_Type', 'Category'], inplace=True)
    # Save the DataFrame as an Excel file in the specified directory
    output_path = os.path.join(dir_path, "category_pivot.xlsx")
    category_df.to_excel(output_path, engine="openpyxl")
    print(f"File saved successfully at: {output_path}")
    return category_df

def parent_pivot_table(df,parent_placekey_dfs,dir_path):

  filtered_df = df[df["PARENT_PLACEKEY"].notnull()]
  pivot_data = []
  for _, parent_row in parent_placekey_dfs.iterrows():
      # Get the PLACEKEY and building from the current row in parent_placekey_dfs
      parent_placekey = parent_row["PLACEKEY"]
      parent_building = parent_row["LOCATION_NAME"]
      parent_count = parent_row["Median RAW_VISIT_COUNTS"]

      # Filter rows from filtered_df where PARENT_PLA matches the parent PLACEKEY
      child_rows = filtered_df[filtered_df["PARENT_PLACEKEY"] == parent_placekey]

      # Add data to the pivot table structure
      for _, child_row in child_rows.iterrows():
          pivot_data.append({
              "Parent Placekey":parent_placekey,
              "Parent Location": parent_building,
              "Parent Visit Counts":parent_count,
              "Child Location Name": child_row["LOCATION_NAME"],  # Use LOCATION_N from child rows
              **child_row.to_dict()  # Include the rest of the columns in filtered_df
          })

  # Step 5: Create a DataFrame from the pivot data
  pivot_df = pd.DataFrame(pivot_data)

  # Step 6: Set a multi-index with Building and Location Name
  pivot_df.set_index(["Parent Placekey","Parent Location","Parent Visit Counts", "Child Location Name"], inplace=True)
  output_path = os.path.join(dir_path, "parent_pivot.xlsx")
  pivot_df.to_excel(output_path, engine="openpyxl")
  print(f"File saved successfully at: {output_path}")

  shared_df=pd.concat([parent_placekey_dfs,filtered_df],copy=True)
  shared_df.drop_duplicates(subset='PLACEKEY',inplace=True,ignore_index=True)

  output_path = os.path.join(dir_path, "parent_child_df.csv")
  shared_df.to_csv(output_path,index=False)
  print(f"File saved successfully at: {output_path}")

  output_path = os.path.join(dir_path, "shared_placekeys_no_parent.csv")
  filtered_df.to_csv(output_path,index=False)
  print(f"File saved successfully at: {output_path}")

  return pivot_df,filtered_df,shared_df
