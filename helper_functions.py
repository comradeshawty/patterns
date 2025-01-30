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
