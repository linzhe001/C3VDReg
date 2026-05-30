import pandas as pd
import os
import numpy as np
import json

# --- Configuration ---
BASE_RESULTS_PATH = "/home/linzhe/PCLR_compare/src/unified_testing/test_results"
ANGLES_TO_SEARCH = [50, 60, 70, 80, 90]
RESULTS_JSON_PATH = "/home/linzhe/PCLR_compare/experiments/results.json"

ALGORITHMS_ORDER = ["Mamba3D", "PointNetLK_revisited", "ICP", "DCP", "PointNetLK"]

# --- Part 1: Determine Ground-Truth Rankings from results.json ---
print("--- Parsing results.json to determine target rankings ---")
target_rankings = {}
try:
    with open(RESULTS_JSON_PATH, 'r') as f:
        avg_results = json.load(f)
    
    for angle in ANGLES_TO_SEARCH:
        angle_str = str(angle)
        c3vd_data = avg_results.get("C3VD", {})
        
        rot_errors = {}
        for algo in ALGORITHMS_ORDER:
            if algo in c3vd_data and angle_str in c3vd_data[algo]:
                rot_errors[algo] = c3vd_data[algo][angle_str]['rotation']['mean']
        
        if len(rot_errors) == len(ALGORITHMS_ORDER):
            sorted_rot = sorted(rot_errors.items(), key=lambda item: item[1])
            rot_rank = [item[0] for item in sorted_rot]
            target_rankings[angle] = rot_rank
            print(f"Angle {angle}° Target Rank (Rotation): {' > '.join(rot_rank)}")
        else:
            print(f"Warning: Missing data for angle {angle} in results.json")

except FileNotFoundError:
    print(f"FATAL: {RESULTS_JSON_PATH} not found.")
    exit()
print("---------------------------------------------------------\n")


# --- Part 2: Search for samples matching the target ranking ---
all_matching_candidates = []

for angle in ANGLES_TO_SEARCH:
    target_rank = target_rankings.get(angle)
    if not target_rank:
        continue

    print(f"\n{'='*20} Searching Angle: {angle} {'='*20}")
    
    # Find CSV files
    csv_files = {}
    ALGO_FOLDERS = {
        "Mamba3D": "pointnetlk_mamba3d_c3vd_need",
        "ICP": "icp_c3vd",
        "DCP": "dcp_c3vd_trained",
        "PointNetLK": "pointnetlk_c3vd",
        "PointNetLK_revisited": "pointnetlk_revisited_c3vd_from_scratch_all_angles",
    }
    for name in ALGORITHMS_ORDER:
        folder = ALGO_FOLDERS[name]
        path = os.path.join(BASE_RESULTS_PATH, folder, f"results_angle_{angle}_corrected.csv")
        if not os.path.exists(path):
            path = os.path.join(BASE_RESULTS_PATH, folder, f"results_angle_{angle}.csv")
        
        if os.path.exists(path):
            csv_files[name] = path

    if len(csv_files) < len(ALGORITHMS_ORDER):
        print(f"Skipping angle {angle}, not all algorithm CSVs were found.")
        continue

    # Load and merge, keeping translation error for Mamba3D
    dfs = []
    for name, path in csv_files.items():
        cols_to_load = ['pert_idx', 'rotation_error_deg']
        if name == 'Mamba3D':
            cols_to_load.append('translation_error_m')
        
        df = pd.read_csv(path, usecols=cols_to_load)
        rename_dict = {'rotation_error_deg': name}
        if name == 'Mamba3D':
            rename_dict['translation_error_m'] = 'trans_Mamba3D'
        df = df.rename(columns=rename_dict)
        dfs.append(df)

    merged_df = dfs[0]
    for df_to_merge in dfs[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on='pert_idx', how='inner')

    if merged_df.empty:
        print(f"No common pert_idx found for angle {angle}.")
        continue

    print(f"Found {len(merged_df)} common pert_idx entries.")

    # Determine actual ranking for each row
    def get_row_ranking(row, cols):
        return list(row[cols].sort_values().index)

    merged_df['actual_rank'] = merged_df.apply(lambda row: get_row_ranking(row, ALGORITHMS_ORDER), axis=1)

    # Find perfect matches
    perfect_matches = merged_df[merged_df['actual_rank'].apply(lambda x: x == target_rank)].copy()
    
    if not perfect_matches.empty:
        print(f"Found {len(perfect_matches)} samples with perfect ranking match!")
        perfect_matches['rot_gap'] = perfect_matches[target_rank[-1]] - perfect_matches[target_rank[0]]
        perfect_matches['angle'] = angle
        all_matching_candidates.append(perfect_matches)
    else:
        print("No samples found with a perfect ranking match.")

# --- Part 3: Report the best find ---
if all_matching_candidates:
    final_df = pd.concat(all_matching_candidates)
    # Sort by Mamba3D translation error (ASC), then by rotation gap (DESC)
    final_df = final_df.sort_values(by=['trans_Mamba3D', 'rot_gap'], ascending=[True, False])

    print("\n" + "="*80)
    print("Top Candidates with Perfect Ranking Match (Sorted by Mamba3D Translation Error)")
    print("="*80)
    
    pd.set_option('display.width', 200)
    cols_to_show = ['angle', 'pert_idx', 'trans_Mamba3D', 'rot_gap'] + ALGORITHMS_ORDER
    print(final_df.head(15)[cols_to_show].round(4))
else:
    print("\nCould not find any sample that perfectly matches the average ranking at any angle.")