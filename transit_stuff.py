import datetime  # Use only 'import datetime' instead of 'from datetime import datetime'
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
import gtfs_kit as gk

def process_pois_and_stops(
    mp,
    stops_gdf,
    stop_stats,
    dates,         # List of YYYYMMDD date strings, e.g. ["20231201","20231202"] – used only with stop_stats
    stop_times_gdf, # DataFrame/GeoDataFrame with columns like ['geometry','stop_id','trip_id','arrival_time','departure_time'] etc.
    routes_gdf     # GeoDataFrame with routes (with 'route_id' and 'geometry' columns)
):
    """
    Fixes references to stop_times_gdf so that we don't rely on a 'date' column.
    This function:
      1) Reprojects the input DataFrames (mp, stops_gdf) to EPSG:32616 if needed.
      2) Uses cKDTree to find stops within 500m of each point in mp.
      3) Creates a subset (stops_mp) of mp for rows that have at least one nearby stop.
      4) Builds day-based usage stats from stop_stats (if available).
      5) Computes a transit_service_period from stop_stats (earliest start_time, latest end_time).
      6) Computes:
         a) stops_by_hour: total stops per hour using arrival_time in stop_times_gdf,
         b) avg_time_between_stops: average of consecutive arrival_time differences (in seconds),
         c) stop_frequency: median of consecutive arrival_time differences (in seconds),
         d) median_headway: per-stop median headway (in minutes), ignoring date-based grouping
            (each stop’s times are considered as a single day’s worth of arrivals),
         e) route_ids: route lookup from routes_gdf if provided.
    """
    # ----------------------------------------------------------------
    # 1) Reproject geometry if needed
    # ----------------------------------------------------------------
    if mp.crs.to_string() != "EPSG:32616":
        mp = mp.to_crs(epsg=32616)
    if stops_gdf.crs.to_string() != "EPSG:32616":
        stops_gdf = stops_gdf.to_crs(epsg=32616)

    # ----------------------------------------------------------------
    # 2) Build cKDTree for stops, find stops within 500m
    # ----------------------------------------------------------------
    stops_coords = np.array([(geom.x, geom.y) for geom in stops_gdf.geometry])
    poi_coords = np.array([(geom.x, geom.y) for geom in mp.geometry])
    tree = cKDTree(stops_coords)
    radius = 500.0
    result_indices = tree.query_ball_point(poi_coords, r=radius)

    # Make sure stop_id is string
    stops_gdf["stop_id"] = stops_gdf["stop_id"].astype(str)
    idx_to_stop_id = stops_gdf["stop_id"].reset_index(drop=True)

    mp["nearby_stop_ids"] = [
        [idx_to_stop_id[i] for i in idx_list] if len(idx_list) > 0 else []
        for idx_list in result_indices
    ]
    mp["num_nearby_stops"] = mp["nearby_stop_ids"].apply(lambda x: len(x) if x else np.nan)

    # ----------------------------------------------------------------
    # 3) Create subset with valid rows
    # ----------------------------------------------------------------
    condition = mp["num_nearby_stops"].notna() & (mp["num_nearby_stops"] > 0)
    essential_cols = [
        "PLACEKEY", "PARENT_PLACEKEY", "LOCATION_NAME", "address", "place_category", "place_subcategory",
        "CATEGORY_TAGS", "MEDIAN_DWELL", "weighted_median_distance_from_home", "num_nearby_stops",
        "nearby_stop_ids", "POI_CBG", "visitor_counts_cbg_scaled", "adjusted_visits_by_day", "geometry"
    ]
    existing_cols = [col for col in essential_cols if col in mp.columns]
    stops_mp = mp.loc[condition, existing_cols].copy()

    # ----------------------------------------------------------------
    # 4) Summaries from stop_stats (still uses 'date' if present)
    # ----------------------------------------------------------------
    stop_stats["stop_id"] = stop_stats["stop_id"].astype(str)
    stop_stats["date"] = stop_stats["date"].astype(str)
    dp = stop_stats.pivot_table(
        index="stop_id", columns="date", values="num_trips", aggfunc="sum", fill_value=0
    )

    if dates is None:
        # fallback if user doesn't supply dates
        dates = list(dp.columns)

    # day-of-week mapping
    def safe_dow(d_str):
        try:
            dt_ = datetime.datetime.strptime(d_str, "%Y%m%d")
            return dt_.strftime("%A").lower()
        except:
            return "unknown"

    date2dow = {d_: safe_dow(d_) for d_ in dp.columns}

    def get_stops_by_day(nearby_ids):
        if dp.empty or not nearby_ids:
            return [0]*len(dates)
        sub = dp.loc[dp.index.intersection(nearby_ids)]
        if sub.empty:
            return [0]*len(dates)
        sums = sub.sum(axis=0).reindex(dates, fill_value=0)
        return sums.tolist()

    def get_stops_by_day_of_week(nearby_ids):
        if dp.empty or not nearby_ids:
            return {}
        sub = dp.loc[dp.index.intersection(nearby_ids)]
        if sub.empty:
            return {}
        sums = sub.sum(axis=0)
        output = {}
        for d_ in sums.index:
            dw = date2dow.get(d_, "unknown")
            output[dw] = output.get(dw, 0) + sums[d_]
        return output

    stops_mp["stops_by_day"] = stops_mp["nearby_stop_ids"].apply(get_stops_by_day)
    stops_mp["stops_by_day_of_week"] = stops_mp["nearby_stop_ids"].apply(get_stops_by_day_of_week)

    # ----------------------------------------------------------------
    # 5) transit_service_period from stop_stats
    # ----------------------------------------------------------------
    def get_service_period(nearby_ids):
        if not nearby_ids:
            return {}
        sub = stop_stats[stop_stats["stop_id"].isin(nearby_ids)]
        if sub.empty:
            return {}

        local_map = {}
        for row in sub.itertuples():
            date_str = getattr(row, "date", None)
            local_map[date_str] = safe_dow(date_str)

        output = {}
        for row in sub.itertuples():
            date_str = getattr(row, "date", None)
            dow = local_map.get(date_str, "unknown")
            st = getattr(row, "start_time", None)
            et = getattr(row, "end_time", None)
            if not isinstance(st, str) or not isinstance(et, str):
                continue
            try:
                st_obj = datetime.datetime.strptime(st, "%H:%M:%S").time()
                et_obj = datetime.datetime.strptime(et, "%H:%M:%S").time()
            except:
                st_obj = datetime.time(0, 0, 0)
                et_obj = datetime.time(0, 0, 0)

            if dow not in output:
                output[dow] = [st_obj, et_obj]
            else:
                curr_st, curr_et = output[dow]
                output[dow] = [min(curr_st, st_obj), max(curr_et, et_obj)]
        # format final
        for k, val in output.items():
            st_obj, et_obj = val
            output[k] = [st_obj.strftime("%H:%M:%S"), et_obj.strftime("%H:%M:%S")]
        return output

    stops_mp["transit_service_period"] = stops_mp["nearby_stop_ids"].apply(get_service_period)

    # ----------------------------------------------------------------
    # 6) Additional metrics from stop_times_gdf (if provided)
    # ----------------------------------------------------------------
    if stop_times_gdf is not None:
        stop_times_gdf = stop_times_gdf.copy()

        # 6a) stops_by_hour
        if "arrival_time" not in stop_times_gdf.columns:
            # If arrival_time is missing, fill placeholders
            stops_mp["stops_by_hour"] = [[] for _ in range(len(stops_mp))]
            stops_mp["avg_time_between_stops"] = np.nan
            stops_mp["stop_frequency"] = np.nan
            stops_mp["median_headway"] = [[] for _ in range(len(stops_mp))]
            return stops_mp

        # Convert arrival_time to hour
        stop_times_gdf["arrival_hour"] = pd.to_datetime(
            stop_times_gdf["arrival_time"], format="%H:%M:%S", errors="coerce"
        ).dt.hour

        # We group by stop_id, arrival_hour
        by_hour = stop_times_gdf.groupby(["stop_id", "arrival_hour"]).size().unstack(fill_value=0)
        # fill missing hours
        for hr in range(24):
            if hr not in by_hour.columns:
                by_hour[hr] = 0
        by_hour = by_hour[sorted(by_hour.columns)]

        def get_stops_by_hour_fn(nearby_ids):
            if not nearby_ids:
                return [0]*24
            sub_df = by_hour.loc[by_hour.index.intersection(nearby_ids)]
            if sub_df.empty:
                return [0]*24
            hour_sums = sub_df.sum(axis=0).reindex(range(24), fill_value=0)
            return hour_sums.tolist()

        stops_mp["stops_by_hour"] = stops_mp["nearby_stop_ids"].apply(get_stops_by_hour_fn)

        # 6b) avg_time_between_stops (seconds)
        def compute_avg_time_between(nearby_ids):
            if not nearby_ids:
                return np.nan
            sub = stop_times_gdf[stop_times_gdf["stop_id"].isin(nearby_ids)].dropna(subset=["arrival_time"])
            if sub.empty:
                return np.nan
            # Convert to seconds from midnight
            sub["arrival_sec"] = pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.hour * 3600 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.minute * 60 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.second
            arr_unique = np.sort(sub["arrival_sec"].dropna().unique())
            if len(arr_unique) < 2:
                return np.nan
            diffs = np.diff(arr_unique)
            return float(np.mean(diffs))

        stops_mp["avg_time_between_stops"] = stops_mp["nearby_stop_ids"].apply(compute_avg_time_between)

        # 6c) stop_frequency (median headway in seconds)
        def compute_stop_frequency(nearby_ids):
            if not nearby_ids:
                return np.nan
            sub = stop_times_gdf[stop_times_gdf["stop_id"].isin(nearby_ids)].dropna(subset=["arrival_time"])
            if sub.empty:
                return np.nan
            sub["arrival_sec"] = pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.hour * 3600 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.minute * 60 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.second
            arr_unique = np.sort(sub["arrival_sec"].dropna().unique())
            if len(arr_unique) < 2:
                return np.nan
            diffs = np.diff(arr_unique)
            return float(np.median(diffs))

        stops_mp["stop_frequency"] = stops_mp["nearby_stop_ids"].apply(compute_stop_frequency)

        # 6d) median_headway per-stop (minutes), ignoring date
        def compute_median_headway_per_stop(stop_id):
            sub = stop_times_gdf[stop_times_gdf["stop_id"] == stop_id].dropna(subset=["arrival_time"])
            if sub.empty:
                return np.nan

            # Convert to seconds
            sub["arrival_sec"] = pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.hour * 3600 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.minute * 60 + pd.to_datetime(
                sub["arrival_time"], format="%H:%M:%S", errors="coerce"
            ).dt.second

            # Sort arrival times and get diffs
            arr_unique = np.sort(sub["arrival_sec"].dropna().unique())
            if len(arr_unique) < 2:
                return np.nan
            diffs = np.diff(arr_unique)
            return float(np.median(diffs) / 60.0)  # convert seconds -> minutes

        def compute_median_headways(nearby_ids):
            if not nearby_ids:
                return []
            return [compute_median_headway_per_stop(s) for s in nearby_ids]

        stops_mp["median_headway"] = stops_mp["nearby_stop_ids"].apply(compute_median_headways)

    # ----------------------------------------------------------------
    # If routes_gdf is provided, find route_ids for each record
    # ----------------------------------------------------------------
    if routes_gdf is not None:
        if routes_gdf.crs.to_string() != mp.crs.to_string():
            routes_gdf = routes_gdf.to_crs(mp.crs)

        # For quick lookup, we use centroid
        routes_gdf = routes_gdf.copy()
        routes_gdf["centroid"] = routes_gdf.geometry.centroid
        route_coords = np.array([(c.x, c.y) for c in routes_gdf["centroid"]])
        route_tree = cKDTree(route_coords)

        def get_nearby_route_ids(point):
            coord = (point.x, point.y)
            indices = route_tree.query_ball_point(coord, r=500)
            if not indices:
                return []
            return list(pd.unique(routes_gdf.iloc[indices]["route_id"]))

        stops_mp["route_ids"] = stops_mp["geometry"].apply(get_nearby_route_ids)

    return stops_mp
