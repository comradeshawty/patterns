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
