def remove_nearby_duplicate_offices(mp_gdf, placekeys_to_drop_path, distance_threshold=20, fuzz_threshold=65):

    try:
        placekeys_to_drop = pd.read_csv(placekeys_to_drop_path)
    except FileNotFoundError:
        placekeys_to_drop = pd.DataFrame(columns=['PLACEKEY'])

    excluded_brands = {'Walmart', 'Winn Dixie', 'Walgreens', 'CVS','Publix'}
    excluded_categories = {'Child Day Care Services', 'Elementary and Secondary Schools','Child and Youth Services'}
    mp_filtered = mp_gdf[(~mp_gdf['BRANDS'].isin(excluded_brands)) &(~mp_gdf['LOCATION_NAME'].str.contains("Emergency", case=False, na=False)) &(~mp_gdf['LOCATION_NAME'].str.contains("Walmart|Winn Dixie|Walgreens|CVS|Publix", case=False, na=False)) &(~mp_gdf['TOP_CATEGORY'].isin(excluded_categories))].copy()
    mp_filtered = gpd.GeoDataFrame(mp_filtered.copy(), geometry=gpd.points_from_xy(mp_filtered.LONGITUDE, mp_filtered.LATITUDE), crs="EPSG:4326").to_crs(epsg=32616)
    mp_filtered=mp_filtered[mp_filtered['POLYGON_CLASS']=='SHARED_POLYGON']
    mp_filtered = mp_filtered.reset_index(drop=True)
    coords = np.array(list(zip(mp_filtered.geometry.x, mp_filtered.geometry.y)))
    tree = cKDTree(coords)

    to_remove = set()
    new_removed_rows = []
    new_placekeys_to_drop = []

    index_mapping = dict(zip(range(len(mp_filtered)), mp_filtered.index))

    for idx, coord in zip(mp_filtered.index, coords):
        if idx in to_remove:
            continue

        nearby_indices = [mp_filtered.index[i] for i in tree.query_ball_point(coord, distance_threshold)]
        current_name = mp_filtered.at[idx, 'LOCATION_NAME']
        current_address = mp_filtered.at[idx, 'address']

        duplicates = [idx]

        for i in nearby_indices:
            if i == idx or i in to_remove:
                continue

            nearby_name = mp_filtered.at[i, 'LOCATION_NAME']
            nearby_address = mp_filtered.at[i, 'address']

            name_similarity = fuzz.ratio(current_name, nearby_name)
            address_similarity = fuzz.ratio(current_address, nearby_address)

            if name_similarity >= fuzz_threshold and address_similarity >= fuzz_threshold:
                duplicates.append(i)

        if len(duplicates) > 1:
            duplicate_rows = mp_filtered.loc[duplicates]

            parent_rows = duplicate_rows[duplicate_rows['parent_flag'] == 1]

            if not parent_rows.empty:
                # Keep one of the parent_flag=1 rows with the highest visit count
                keep_idx = parent_rows['RAW_VISIT_COUNTS'].idxmax()
            else:
                # No parent_flag=1, so keep the row with the highest visit count
                keep_idx = duplicate_rows['RAW_VISIT_COUNTS'].idxmax()

            remove_idxs = [i for i in duplicates if i != keep_idx]

            # Handle case where neither removal condition is met (same visit counts & parent flags)
            if all(mp_filtered.loc[i, 'parent_flag'] == mp_filtered.loc[keep_idx, 'parent_flag'] for i in remove_idxs) and \
               all(mp_filtered.loc[i, 'RAW_VISIT_COUNTS'] == mp_filtered.loc[keep_idx, 'RAW_VISIT_COUNTS'] for i in remove_idxs):
                remove_idxs = remove_idxs[:1]

            # If both have parent_flag=1, update PARENT_PLACEKEY references
            if len(parent_rows) > 1:
                kept_placekey = mp_filtered.loc[keep_idx, 'PLACEKEY']
                for remove_idx in remove_idxs:
                    removed_placekey = mp_filtered.loc[remove_idx, 'PLACEKEY']
                    mp_gdf.loc[mp_gdf['PARENT_PLACEKEY'] == removed_placekey, 'PARENT_PLACEKEY'] = kept_placekey
                    print(f"**Updated PARENT_PLACEKEY:** Replaced {removed_placekey} → {kept_placekey}")

            print(f"\n**Keeping:** {mp_filtered.loc[keep_idx, 'PLACEKEY']} | {mp_filtered.loc[keep_idx, 'LOCATION_NAME']} | {mp_filtered.loc[keep_idx, 'address']}")
            print("**Removing:**")
            for i in remove_idxs:
                print(f"   - {mp_filtered.loc[i, 'PLACEKEY']} | {mp_filtered.loc[i, 'LOCATION_NAME']} | {mp_filtered.loc[i, 'address']} (Matched)")

            new_removed_rows.extend(mp_filtered.loc[remove_idxs].to_dict('records'))
            new_placekeys_to_drop.extend(mp_filtered.loc[remove_idxs, 'PLACEKEY'].tolist())
            to_remove.update(remove_idxs)

    new_removed_df = pd.DataFrame(new_removed_rows)
    mp_gdf_cleaned = mp_gdf.drop(index=mp_gdf.index.intersection(to_remove)).reset_index(drop=True)
    new_placekeys_df = pd.DataFrame({'PLACEKEY': new_placekeys_to_drop})
    placekeys_to_drop = pd.concat([placekeys_to_drop, new_placekeys_df], ignore_index=True).drop_duplicates()
    placekeys_to_drop.to_csv(placekeys_to_drop_path, index=False)
    print(f"\nAdded {len(new_placekeys_to_drop)} PLACEKEYs to `{placekeys_to_drop_path}`.")
    return mp_gdf_cleaned

def calculate_polygon_diameter(mp):
    diameters = []

    for geom in mp['POLYGON_WKT']:
        try:
            # Only apply loads() if it's still a string
            if isinstance(geom, str):
                polygon = loads(geom)  # Convert WKT string to Polygon
            else:# isinstance(geom, Polygon):
                polygon = geom  # Already a Polygon, no need to convert
            #else:
                #raise ValueError("Invalid data type for POLYGON_WKT")

            # Get bounding box (min_x, min_y, max_x, max_y)
            min_x, min_y, max_x, max_y = polygon.bounds

            # Compute geodesic distance between diagonal corners
            diameter = geodesic((min_y, min_x), (max_y, max_x)).meters
            diameters.append(diameter)

        except Exception as e:
            #print(f"Error processing geometry: {geom}, Error: {e}")
            diameters.append(None)

    mp['POLYGON_DIAMETER'] = diameters
    return mp

def remove_nearby_duplicate_offices_no_address(mp_gdf, placekeys_to_drop_path, fuzz_threshold=75):

    try:
        placekeys_to_drop = pd.read_csv(placekeys_to_drop_path)
    except FileNotFoundError:
        placekeys_to_drop = pd.DataFrame(columns=['PLACEKEY'])
    mp_gdf = calculate_polygon_diameter(mp_gdf)
    excluded_brands = {'Walmart', 'Winn Dixie', 'Walgreens', 'CVS','Publix',"Walmart Photo Center","Walmart Vision Center","Walmart Auto Care Center","Walmart Pharmacy","Woodforest National Bank","Jackson Hewitt Tax Service"}
    excluded_categories = {'Child Day Care Services', 'Elementary and Secondary Schools','Child and Youth Services'}
    mp_filtered = mp_gdf[
        (~mp_gdf['BRANDS'].isin(excluded_brands)) &
        (~mp_gdf['LOCATION_NAME'].str.contains("Emergency", case=False, na=False)) &
        #(~mp_gdf['LOCATION_NAME'].str.contains("Walmart|Winn Dixie|Walgreens|CVS|Publix", case=False, na=False)) &
        (~mp_gdf['TOP_CATEGORY'].isin(excluded_categories)) &
        (~mp_gdf['LOCATION_NAME'].isin(excluded_brands))  # **NEW EXCLUSION**
    ].copy()
    mp_filtered = gpd.GeoDataFrame(mp_filtered.copy(), geometry=gpd.points_from_xy(mp_filtered.LONGITUDE, mp_filtered.LATITUDE), crs="EPSG:4326").to_crs(epsg=32616)
    #mp_filtered=mp_filtered[mp_filtered['POLYGON_CLASS']=='SHARED_POLYGON']
    mp_filtered = mp_filtered.reset_index(drop=True)
    coords = np.array(list(zip(mp_filtered.geometry.x, mp_filtered.geometry.y)))
    tree = cKDTree(coords)

    to_remove = set()
    new_removed_rows = []
    new_placekeys_to_drop = []

    index_mapping = dict(zip(range(len(mp_filtered)), mp_filtered.index))

    for idx, coord in zip(mp_filtered.index, coords):
        if idx in to_remove:
            continue
        distance_threshold = mp_filtered.at[idx, 'POLYGON_DIAMETER']  # Use row-specific POLYGON_DIAMETER
        if pd.isna(distance_threshold):
            distance_threshold = 100  # Default fallback if missing

        nearby_indices = [mp_filtered.index[i] for i in tree.query_ball_point(coord, distance_threshold)]
        current_name = mp_filtered.at[idx, 'LOCATION_NAME']
        current_category = mp_filtered.at[idx, 'TOP_CATEGORY']
        duplicates = [idx]

        for i in nearby_indices:
            if i == idx or i in to_remove:
                continue

            nearby_name = mp_filtered.at[i, 'LOCATION_NAME']
            nearby_category = mp_filtered.at[i, 'TOP_CATEGORY']

            name_similarity = fuzz.ratio(current_name, nearby_name)
            #address_similarity = fuzz.ratio(current_address, nearby_address)
            if (
                ("Religious Organization" in {current_category, nearby_category}) and
                ("Child Day Care Services" in {current_category, nearby_category} or
                 any(keyword in current_name.lower() or keyword in nearby_name.lower() for keyword in ["childcare", "daycare", "child"]))
            ):
                continue  # Skip removing this pair

            if name_similarity >= fuzz_threshold:# and address_similarity >= fuzz_threshold:
                duplicates.append(i)

        if len(duplicates) > 1:
            duplicate_rows = mp_filtered.loc[duplicates]

            parent_rows = duplicate_rows[duplicate_rows['parent_flag'] == 1]

            if not parent_rows.empty:
                # Keep one of the parent_flag=1 rows with the highest visit count
                keep_idx = parent_rows['RAW_VISIT_COUNTS'].idxmax()
            else:
                # No parent_flag=1, so keep the row with the highest visit count
                keep_idx = duplicate_rows['RAW_VISIT_COUNTS'].idxmax()

            remove_idxs = [i for i in duplicates if i != keep_idx]

            # Handle case where neither removal condition is met (same visit counts & parent flags)
            if all(mp_filtered.loc[i, 'parent_flag'] == mp_filtered.loc[keep_idx, 'parent_flag'] for i in remove_idxs) and \
               all(mp_filtered.loc[i, 'RAW_VISIT_COUNTS'] == mp_filtered.loc[keep_idx, 'RAW_VISIT_COUNTS'] for i in remove_idxs):
                remove_idxs = remove_idxs[:1]

            # If both have parent_flag=1, update PARENT_PLACEKEY references
            if len(parent_rows) > 1:
                kept_placekey = mp_filtered.loc[keep_idx, 'PLACEKEY']
                for remove_idx in remove_idxs:
                    removed_placekey = mp_filtered.loc[remove_idx, 'PLACEKEY']
                    mp_gdf.loc[mp_gdf['PARENT_PLACEKEY'] == removed_placekey, 'PARENT_PLACEKEY'] = kept_placekey
                    print(f"**Updated PARENT_PLACEKEY:** Replaced {removed_placekey} → {kept_placekey}")

            print(f"\n**Keeping:** {mp_filtered.loc[keep_idx, 'PLACEKEY']} | {mp_filtered.loc[keep_idx, 'LOCATION_NAME']} | {mp_filtered.loc[keep_idx, 'address']}")
            print("**Removing:**")
            for i in remove_idxs:
                print(f"   - {mp_filtered.loc[i, 'PLACEKEY']} | {mp_filtered.loc[i, 'LOCATION_NAME']} | {mp_filtered.loc[i, 'address']} (Matched)")

            new_removed_rows.extend(mp_filtered.loc[remove_idxs].to_dict('records'))
            new_placekeys_to_drop.extend(mp_filtered.loc[remove_idxs, 'PLACEKEY'].tolist())
            to_remove.update(remove_idxs)

    new_removed_df = pd.DataFrame(new_removed_rows)
    mp_gdf_cleaned = mp_gdf.drop(index=mp_gdf.index.intersection(to_remove)).reset_index(drop=True)
    new_placekeys_df = pd.DataFrame({'PLACEKEY': new_placekeys_to_drop})
    placekeys_to_drop = pd.concat([placekeys_to_drop, new_placekeys_df], ignore_index=True).drop_duplicates()
    placekeys_to_drop.to_csv(placekeys_to_drop_path, index=False)
    print(f"\nAdded {len(new_placekeys_to_drop)} PLACEKEYs to `{placekeys_to_drop_path}`.")
    return mp_gdf_cleaned
def parent_childs(df_filtered):
    parent_placekeys_set = set(df_filtered["PARENT_PLACEKEY"].dropna().unique())
    parent_placekey_dfs = df_filtered.loc[df_filtered["PLACEKEY"].isin(parent_placekeys_set)].copy()
    parent_placekey_set = set(parent_placekey_dfs["PLACEKEY"])
    df_filtered = df_filtered.copy()
    df_filtered.loc[:, 'parent_flag'] = df_filtered['PLACEKEY'].apply(lambda pk: 1 if pk in parent_placekey_set else 0)
    parent_counts = (len(df_filtered.loc[df_filtered['parent_flag'] == 1]) / len(df_filtered)) * 100
    print(f"Percentage of parent placekeys: {parent_counts}")
    return df_filtered
def view_category(mp, category):
    return mp[mp['TOP_CATEGORY'].isin(category)]
def view_sub_category(mp, sub_category):
    return mp[mp['SUB_CATEGORY'].isin(sub_category)]
def view_brands(mp, brands):
    return mp[mp['BRANDS'].isin(brands)]
def view_catlabel(mp,catlabel):
  return mp[mp['three_cat_label'].isin(catlabel)]
def update_placekey_info(mp, placekeys, new_location_name=None,new_top_category=None, new_subcategory=None, new_naics_code=None, new_category_tags=None):
    if isinstance(placekeys, str):
        placekeys = [placekeys]

    mask = mp['PLACEKEY'].isin(placekeys)
    if not mask.any():
        print(f"No matching PLACEKEYs found in mp for {placekeys}. Skipping update.")
        return mp
    if new_location_name:
        mp.loc[mask, 'LOCATION_NAME'] = new_location_name
    if new_top_category:
        mp.loc[mask, 'TOP_CATEGORY'] = new_top_category
    if new_subcategory:
        mp.loc[mask, 'SUB_CATEGORY'] = new_subcategory
    if new_naics_code:
        mp.loc[mask, 'NAICS_CODE'] = new_naics_code
    if new_category_tags:
        mp.loc[mask, 'CATEGORY_TAGS'] = mp.loc[mask, 'CATEGORY_TAGS'].fillna('') + \
                                         (', ' if mp.loc[mask, 'CATEGORY_TAGS'].notna().all() else '') + \
                                         new_category_tags
    print(f"Updated {mask.sum()} rows for PLACEKEYs: {placekeys[:5]}{'...' if len(placekeys) > 5 else ''}")
    return mp

def update_legal_services(mp):
    mask = mp['LOCATION_NAME'].str.contains(r'Law|Atty|Attorney|Law Firm', case=False, na=False, regex=True)
    correct_classification = ((mp['TOP_CATEGORY'] == 'Legal Services') &(mp['SUB_CATEGORY'] == 'Offices of Lawyers') &(mp['NAICS_CODE'] == '541110'))
    mp.loc[mask & ~correct_classification, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Legal Services', 'Offices of Lawyers', '541110']
    return mp
def update_theater_companies(mp):
    mask = mp['LOCATION_NAME'].str.contains(r'Theater|Theatre', case=False, na=False)
    mp.loc[mask, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Performing Arts Companies', 'Theater Companies and Dinner Theaters', '711110']
    return mp
def update_non_court_services(mp):
    subcategory_mask = mp['SUB_CATEGORY'] == 'Courts'
    location_mask = ~mp['LOCATION_NAME'].str.contains(r'Court|Courthouse', case=False, na=False)
    target_rows = subcategory_mask & location_mask
    mp.loc[target_rows, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Legal Services', 'Offices of Lawyers', '541110']
    return mp

def remove_wholesalers(mp):
    mp=mp.copy()
    mask = mp['TOP_CATEGORY'].str.contains('Wholesalers', na=False) | \
           mp['SUB_CATEGORY'].str.contains('Wholesalers', na=False)
    mp = mp.loc[~mask].reset_index(drop=True)
    return mp

def update_religious_organizations(mp_gdf):
    religious_terms = ['Church', 'Temple', 'Synagogue', 'Mosque', 'Chapel', 'Cathedral','Basilica', 'Shrine', 'Monastery', 'Gurdwara', 'Tabernacle', 'Missionary','Worship Center', 'Bible Camp', 'Parish','Ministry','Ministries']
    religious_pattern = r'\b(?:' + '|'.join(religious_terms) + r')\b'
    religious_mask = mp_gdf['LOCATION_NAME'].str.contains(religious_pattern, flags=re.IGNORECASE, regex=True, na=False)
    mp_gdf.loc[religious_mask, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Religious Organizations', 'Religious Organizations', '813110']
    return mp_gdf

def update_real_estate_info(mp):
    mp['NAICS_CODE'] = mp['NAICS_CODE'].astype(str)
    mp.loc[mp['NAICS_CODE'] == '531190', 'NAICS_CODE'] = '531120'
    mp.loc[mp['SUB_CATEGORY'] == 'Lessors of Other Real Estate Property','SUB_CATEGORY'] = 'Lessors of Nonresidential Buildings (except Miniwarehouses)'
    mp.loc[mp['SUB_CATEGORY'].isin(['Malls', 'Shopping Centers']), 'NAICS_CODE'] = '531120'
    return mp

def preprocess_mp(mp):

  mp.drop_duplicates(subset='PLACEKEY', inplace=True, ignore_index=True)
  mp.dropna(subset='PLACEKEY', inplace=True, ignore_index=True)
  mp.dropna(subset='VISITOR_HOME_CBGS', inplace=True, ignore_index=True)
  mp.dropna(subset='RAW_VISIT_COUNTS', inplace=True, ignore_index=True)
  mp.dropna(subset='LOCATION_NAME', inplace=True, ignore_index=True)
  mp['TOP_CATEGORY']=mp['TOP_CATEGORY'].astype('str')
  mp['POSTAL_CODE']=mp['POSTAL_CODE'].astype('Int64').astype('str')
  mp['NAICS_CODE']=mp['NAICS_CODE'].astype('Int64').astype('str')
  mp['POI_CBG']=mp['POI_CBG'].astype('Int64').astype('str')
  mp.drop_duplicates(subset=['LOCATION_NAME','STREET_ADDRESS'], inplace=True, ignore_index=True)
  mp.loc[(mp['SUB_CATEGORY'] == 'Malls') & (mp['RAW_VISIT_COUNTS'] < 20000),'SUB_CATEGORY'] = 'Shopping Centers'
  categories_to_drop=['Household Appliance Manufacturing','Warehousing and Storage','Other Miscellaneous Manufacturing','General Warehousing and Storage','Machinery, Equipment, and Supplies Merchant Wholesalers',
  'Building Finishing Contractors','Building Equipment Contractors','Investigation and Security Services','Machinery, Equipment, and Supplies Merchant Wholesalers','Electrical Equipment Manufacturing',
                      'Residential Building Construction','Waste Treatment and Disposal','Waste Management and Remediation Services','Other Specialty Trade Contractors','Motor Vehicle Manufacturing','Miscellaneous Durable Goods Merchant Wholesalers',
                      'Steel Product Manufacturing from Purchased Steel','Glass and Glass Product Manufacturing','Professional and Commercial Equipment and Supplies Merchant Wholesalers','Securities and Commodity Contracts Intermediation and Brokerage',
  'Chemical and Allied Products Merchant Wholesalers','Commercial and Industrial Machinery and Equipment Rental and Leasing','Foundation, Structure, and Building Exterior Contractors','Freight Transportation Arrangement',
  'Lumber and Other Construction Materials Merchant Wholesalers','Specialized Freight Trucking','Business Support Services','Waste Management and Remediation Services','Glass and Glass Product Manufacturing','Data Processing, Hosting, and Related Services',
                      'Coating, Engraving, Heat Treating, and Allied Activities',
                      'Apparel Accessories and Other Apparel Manufacturing']
  sub_categories_to_remove=['Septic Tank and Related Services','Outdoor Power Equipment Stores',
                            'Cable and Other Subscription Programming','Refrigeration Equipment and Supplies Merchant Wholesalers',
                            'Packing and Crating','Other Electronic Parts and Equipment Merchant Wholesalers','Plumbing and Heating Equipment and Supplies (Hydronics) Merchant Wholesalers',
                            'All Other Support Services']
  mp = mp[~mp['TOP_CATEGORY'].isin(categories_to_drop)]
  mp = mp[~mp['SUB_CATEGORY'].isin(sub_categories_to_remove)]
  mp = mp.replace(r'\t', '', regex=True)
  mp = mp[~((mp['SUB_CATEGORY'] == 'Couriers and Express Delivery Services') & (mp['BRANDS'] != 'FedEx'))]
  mp=mp[mp['PLACEKEY']!='zzw-222@8gk-tv9-wrk']
  mp=mp[mp['IS_SYNTHETIC']==False]
  mp = mp.reset_index(drop=True)
  return mp
def process_pois_and_stops(mp, stops, radius=250):
    #stops = stops.loc[stops['TOP_CATEGORY'] == 'Urban Transit Systems'].copy()
    #stops.loc[:, 'LOCATION_NAME'] = stops['LOCATION_NAME'].str.replace(r'^Birmingham Jefferson County Transit Authority\s*', '', regex=True)

    stops.loc[:, 'geometry'] = stops.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
    stops_gdf = gpd.GeoDataFrame(stops, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)
    mp=mp.copy()
    mp.loc[:, 'geometry'] = mp.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
    pois_gdf = gpd.GeoDataFrame(mp, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)

    stop_coords = list(zip(stops_gdf.geometry.x, stops_gdf.geometry.y))
    poi_coords = list(zip(pois_gdf.geometry.x, pois_gdf.geometry.y))
    stop_tree = cKDTree(stop_coords)

    results = []
    for idx, poi_coord in enumerate(poi_coords):
        stop_indices = stop_tree.query_ball_point(poi_coord, radius)
        valid_distances = [np.sqrt((poi_coord[0] - stop_coords[i][0])**2 + (poi_coord[1] - stop_coords[i][1])**2) for i in stop_indices]
        valid_stop_indices = [i for i, dist in zip(stop_indices, valid_distances) if dist < radius]
        nearby_stop_names = [stops_gdf.iloc[i]['stop_name'] for i in valid_stop_indices]
        nearby_stop_ids = [stops_gdf.iloc[i]['stop_id'] for i in valid_stop_indices]

        results.append({
            'PLACEKEY': pois_gdf.iloc[idx]['PLACEKEY'],
            'nearby_stops': nearby_stop_names,
            'nearby_stop_ids':nearby_stop_ids,
            'nearby_stop_distances': valid_distances})

    nearby_pois = pd.DataFrame(results)
    nearby_pois = nearby_pois[(nearby_pois['nearby_stops'].apply(lambda x: len(x) > 0)) &
                              (nearby_pois['nearby_stop_distances'].apply(lambda x: len(x) > 0))].reset_index(drop=True)

    mp = mp.merge(nearby_pois, on="PLACEKEY", how="left")
    mp.loc[:, 'nearby_stops'] = mp['nearby_stops'].fillna('[]').apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    mp.loc[:, 'nearby_stop_distances'] = mp['nearby_stop_distances'].fillna('[]')
    mp.loc[:, 'num_nearby_stops'] = mp['nearby_stops'].apply(len)

    mp.drop_duplicates(subset="PLACEKEY", inplace=True, ignore_index=True)
    mp.dropna(subset="PLACEKEY", inplace=True, ignore_index=True)

    return mp
def extract_visit_counts_by_day(mp):
    def parse_popularity_by_day(value):
        if isinstance(value, str):
            try:
                parsed_dict = ast.literal_eval(value)  # Convert string to dictionary
                return parsed_dict if isinstance(parsed_dict, dict) else {}
            except (SyntaxError, ValueError):
                return {}
        return value if isinstance(value, dict) else {}
    mp['POPULARITY_BY_DAY'] = mp['POPULARITY_BY_DAY'].apply(parse_popularity_by_day)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in weekdays:
        mp[f'visit_count_{day.lower()}'] = mp['POPULARITY_BY_DAY'].apply(lambda x: x.get(day, 0))
    return mp
def drop_duplicates_with_priority(mp, save_path='/content/drive/MyDrive/data/removed_children.csv'):

    mp['VISITOR_HOME_CBGS_STR'] = mp['VISITOR_HOME_CBGS'].astype(str)
    mp_sorted = mp[mp['PARENT_PLACEKEY'].notna()].copy()
    mp_sorted = mp_sorted.sort_values(by=['PARENT_PLACEKEY', 'VISITOR_HOME_CBGS_STR', 'RAW_VISIT_COUNTS'], ascending=[True, True, False])
    cleaned_mp = mp_sorted.drop_duplicates(subset=['PARENT_PLACEKEY', 'VISITOR_HOME_CBGS'], keep='first')
    dropped_rows = mp_sorted[~mp_sorted.index.isin(cleaned_mp.index)]
    dropped_rows.to_csv(save_path, index=False)
    cleaned_mp = cleaned_mp.drop(columns=['VISITOR_HOME_CBGS_STR'])
    cleaned_mp.sort_values(by='RAW_VISIT_COUNTS', ascending=False, inplace=True)
    cleaned_mp.reset_index(drop=True, inplace=True)

    return cleaned_mp, dropped_rows

def three_cat_label(mp):
  mp['three_cat_label'] = mp.apply(lambda row: (
          row['TOP_CATEGORY']  # If both SUB_CATEGORY and CATEGORY_TAGS are NaN
          if pd.isna(row['SUB_CATEGORY']) and pd.isna(row['CATEGORY_TAGS'])
          else f"{row['TOP_CATEGORY']}-{row['SUB_CATEGORY']}"  # If only CATEGORY_TAGS is NaN
          if pd.isna(row['CATEGORY_TAGS'])
          else f"{row['TOP_CATEGORY']}-{row['SUB_CATEGORY']}-{row['CATEGORY_TAGS']}"),axis=1)
  return mp

def convert_placekey_to_stop(mp, placekey, stop_name):
  mp.loc[mp['PLACEKEY'] == placekey, ['LOCATION_NAME', 'TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE','CATEGORY_TAGS']] = [stop_name,'Urban Transit Systems','Bus and Other Motor Vehicle Transit Systems','485113','Bus Station,Buses']
  return mp
  
def update_mp_from_w(mp, w, columns_to_update):

    w_lookup = {col: w.set_index("PLACEKEY")[col].to_dict() for col in columns_to_update}

    # Ensure PLACEKEY is indexed for faster lookup in mp
    if "PLACEKEY" not in mp.columns:
        raise ValueError("PLACEKEY column is missing in mp")

    # Efficiently update only differing values
    for col in columns_to_update:
        if col in mp.columns:
            # Get the values from w where PLACEKEY matches
            w_values = mp["PLACEKEY"].map(w_lookup[col])

            # Find where values are different and update only those
            mask = (w_values.notna()) & (mp[col] != w_values)
            mp.loc[mask, col] = w_values[mask]

    return mp

def merge_duplicate_pois(mp, save_path="/content/drive/MyDrive/data/removed_duplicate_pois.csv"):
    """
    Identifies and merges sequences of 5+ rows with identical visit counts, VISITOR_HOME_CBGS,
    and the same polygon suffix. Replaces them with a single row and saves the removed rows to a CSV.

    Args:
        mp (pd.DataFrame): The input dataframe.
        save_path (str): File path to save removed rows.

    Returns:
        pd.DataFrame: Cleaned dataframe with merged POIs.
        pd.DataFrame: Dropped duplicate rows.
    """

    # Extract polygon identifier from PLACEKEY (everything after @)
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

    # Convert VISITOR_HOME_CBGS to a string for exact comparison
    mp["VISITOR_HOME_CBGS_STR"] = mp["VISITOR_HOME_CBGS"].astype(str)
    mp=mp.sort_values(by='RAW_VISIT_COUNTS',ascending=False)
    # Group by polygon, RAW_VISIT_COUNTS, and VISITOR_HOME_CBGS
    grouped = mp.groupby(["POLYGON_ID", "VISITOR_HOME_CBGS_STR","TOP_CATEGORY"])

    merged_rows = []
    removed_rows = []

    for (_, visit_count, home_cbgs), group in grouped:
        if len(group) >= 6:  # Only process sequences with 5+ duplicates
            first_row = group.iloc[0].copy()  # Keep the first row's values

            # Determine new LOCATION_NAME (Most common TOP_CATEGORY - Most common STREET ADDRESS)
            most_common_top_category = Counter(group["TOP_CATEGORY"]).most_common(1)[0][0]
            most_common_sub_category = Counter(group["SUB_CATEGORY"]).most_common(1)[0][0]
            most_common_tag_category = Counter(group["CATEGORY_TAGS"]).most_common(1)[0][0]

            most_common_address = Counter(group["STREET_ADDRESS"]).most_common(1)[0][0]
            first_row["LOCATION_NAME"] = f"{most_common_top_category} - {most_common_address}"
            first_row["TOP_CATEGORY"]=most_common_top_category
            first_row["SUB_CATEGORY"]=most_common_sub_category
            first_row["CATEGORY_TAGS"]=most_common_tag_category
            # Keep only the first row, discard the rest
            merged_rows.append(first_row)
            removed_rows.append(group.iloc[1:])  # Store dropped rows

    # Create a DataFrame for removed rows and save them
    if removed_rows:
        removed_df = pd.concat(removed_rows)
        removed_df.to_csv(save_path, index=False)
    else:
        removed_df = pd.DataFrame()

    # Create final cleaned dataframe
    cleaned_mp = mp[~mp.index.isin(removed_df.index)]
    cleaned_mp = pd.concat([cleaned_mp, pd.DataFrame(merged_rows)], ignore_index=True)

    # Drop temporary columns
    cleaned_mp.drop(columns=["POLYGON_ID", "VISITOR_HOME_CBGS_STR"], inplace=True)

    return cleaned_mp, removed_df

def update_category(mp, placekeys, new_category):
    if isinstance(placekeys, str):
        placekeys = [placekeys]
    mp.loc[mp["PLACEKEY"].isin(placekeys), "place_category"] = new_category

    return mp
def assign_place_category_and_subcategory(mp, sub_category_mapping, sub_categories_to_pretty_names):
    # Step 1: Assign categories using the sub_category_mapping
    category_lookup = {
        subcategory: category
        for category, subcategories in sub_category_mapping.items()
        for subcategory in subcategories
    }

    mp["place_category"] = mp["SUB_CATEGORY"].map(category_lookup).fillna(
        mp["TOP_CATEGORY"].map(category_lookup)
    ).fillna("Other")  # Default to "Other"
    def map_subcategory(row):
        top_category = row["SUB_CATEGORY"]
        place_category = row["place_category"]
        for main_category, subcategories in sub_categories_to_pretty_names.items():
            for subcategory, values in subcategories.items():
                if top_category in values:
                    return subcategory
        return f"Other {place_category}"  # If no match is found, return 'Other {place_category}'
    mp["place_subcategory"] = mp.apply(map_subcategory, axis=1)


 
    # Step 2: Define keyword-based categories with exact matches
    category_keywords = {
        "Schools": ["School", "Schools", "Academy", "Sch", "Montessori", "Summer Camp"],
        "City/Outdoors": ["Recreation Center", "City", "Playground", "Hiking", "Trail", "Courthouse"],
        "Arts and Culture": ["Mural", "Museum", "Artist",'Arts',"Cultural", "Dance", "Ballroom", "Exhibit"],
        "Entertainment": ["Trampoline Park", "Happy Hour", "Beer", "Beer Garden", "Mini Golf", "Topgolf",
                          "Pool and Billiards", "Axe Throwing", "Arcade", "Casino", "Go Kart", "Laser Tag",
                          "Escape Room", "Nightclub", "Comedy Club", "Event Space", "Speakeasy", "Theme Park",
                          "Water Park", "Winery", "Resort"],
        "Sports and Exercise": ["Tennis", "Gym", "Gymnastics", "Golf", "Yoga", "Rock Climbing", "Swimming",
                                "Baseball Field", "Athletics & Sports", "Basketball", "Climbing Gym",
                                "Hockey", "Skating Rink", "Soccer", "Boxing", "Squash", "Stable", "Volleyball"],
        "Work": ["Office Park", "Corporate Offices", "Corporate Office", "Business Center", "Conference",
                 "Coworking Space", "Meeting Room", "Industrial", "Non-Profit", "Tech Startup", "Warehouse"],
        "Personal Services": ["Barber", "Massage Therapy", "Spa", "Eyebrow", "Waxing", "Tattoo", "Medical Spa",
                              "Skin Care", "Contractors", "Handyman"],
        "Transportation": ["Greyhound", "Amtrak", "Airport", "Bus Station", "Train Station",
                           "Parking", "Taxi", "Terminal", "Travel", "Tunnel"],
        'Coffee Shops, Snacks & Bakeries':['Coffee','Bakery','Treats','Creamery','Smoothie','Donuts',"Jeni's Splendid Ice Creams",
                                           'Yogurt','Doughnuts','Tea','Teahouse','Ice Creams','Ice Cream','Crumbl Cookies','Frutta Bowls']
    }
    coffee_keywords=['Coffee','Bakery','Treats','Creamery','Smoothie','Donuts',"Jeni's Splendid Ice Creams",'Yogurt','Doughnuts','Tea','Teahouse','Ice Creams','Ice Cream','Crumbl Cookies','Frutta Bowls']

    # Step 3: Standardize LOCATION_NAME and CATEGORY_TAGS
    mp["LOCATION_NAME"] = mp["LOCATION_NAME"].str.strip()
    if "CATEGORY_TAGS" in mp.columns:
        mp["CATEGORY_TAGS"] = mp["CATEGORY_TAGS"].str.strip()

    # Step 4: Assign categories only if at least one match occurs
    for category, keywords in category_keywords.items():
        name_match = mp["LOCATION_NAME"].isin(keywords)  # Exact match for LOCATION_NAME

        if "CATEGORY_TAGS" in mp.columns:
            tag_match = mp["CATEGORY_TAGS"].isin(keywords)  # Exact match for CATEGORY_TAGS
        else:
            tag_match = False  # If CATEGORY_TAGS doesn't exist, skip this condition

        # Apply update only if at least one of the conditions is True
        update_mask = (name_match | tag_match) & (mp["place_category"] == "Other")
        mp.loc[update_mask, "place_category"] = category

    # Step 5: Special case for 'Pharmacy' exact match
    mp.loc[mp['LOCATION_NAME'].str.contains('Pharmacy', case=True, na=False), 'place_category'] = 'Retail for Basic Necessities'
    mp.loc[mp['LOCATION_NAME'].str.contains('Recreation Center', case=True, na=False), 'place_category'] = 'City/Outdoors'
    mp.loc[mp["LOCATION_NAME"].str.contains("|".join(coffee_keywords), case=True, na=False),'place_category']=='Coffee Shops, Snacks & Bakeries'
    return mp

def assign_specific_subcategories(mp): 
    # Define exact case-sensitive matches
    keyword_mappings = {
        'Coffee Shop': ['Starbucks',"Ohenry's Coffees", 'Costa Coffee','Revelator Coffee'],
        'Donuts': ["Dunkin'",'Krispy Kreme Doughnuts','Shipley Donuts','Daylight Donuts'],
        'Bakery': ['Cinnaholic','Insomnia Cookies','Great American Cookies', 'Nothing Bundt Cakes'],
        'Ice Cream & Frozen Yogurt': ['Cold Stone Creamery', "Freddy's Frozen Custard",'Yogurt Mountain', 
                                      'Baskin Robbins', 'TCBY', 'Orange Julius','Marble Slab Creamery',
                                      "Bruster's Ice Cream"],
        'Smoothie & Juice Bar': ['Tropical Smoothie Café','Clean Juice','Jamba','Planet Smoothie']
    }

    # Function to check if any of the keywords exist in the specified columns
    def match_subcategory(row):
        if row["place_category"] != "Coffee Shops, Snacks & Bakeries":
            return row["place_subcategory"]  # If not in the category, retain existing value
        
        for subcategory, keywords in keyword_mappings.items():
            if row["BRANDS"] in keywords or row["CATEGORY_TAGS"] in keywords or row["LOCATION_NAME"] in keywords:
                return subcategory
        return row["place_subcategory"]  # Keep existing value if no match

    # Apply the function to update place_subcategory only for relevant rows
    mp["place_subcategory"] = mp.apply(match_subcategory, axis=1)

    return mp

