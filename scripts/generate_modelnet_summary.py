
import pandas as pd
import numpy as np
import json
import ijson
from tqdm import tqdm

def flatten_json_data(file_path):
    """
    Stream-reads and flattens the nested JSON data into a list of records.
    """
    print("Streaming and flattening JSON data...")
    records = []
    with open(file_path, 'rb') as f:
        # Using tqdm to show progress, assuming the top-level object is iterable
        parser = ijson.parse(f)
        
        current_record = {}
        path_stack = []

        for prefix, event, value in parser:
            if event == 'map_key':
                path_stack.append(value)
            elif event == 'start_map':
                if len(path_stack) == 4: # dataset, algorithm, angle, sample_id
                    current_record = {
                        'dataset': path_stack[0],
                        'algorithm': path_stack[1],
                        'angle': int(path_stack[2]),
                        'sample_id': path_stack[3]
                    }
            elif event == 'end_map':
                if len(path_stack) == 4:
                    records.append(current_record)
                
                if path_stack:
                    path_stack.pop()
            
            if event not in ('start_map', 'end_map', 'map_key', 'start_array', 'end_array') and current_record:
                key = path_stack[-1]
                current_record[key] = value
                path_stack.pop()

    print(f"Flattened {len(records)} records.")
    return records

def filter_outliers_iqr(df, group_cols, metric_cols):
    """
    Filters outliers from a DataFrame based on the IQR method for specified metrics,
    grouped by specified columns.
    """
    print("Filtering outliers using IQR method...")
    
    # Create a boolean mask with the same index as the df, default to True
    mask = pd.Series(True, index=df.index)

    # Group data and apply filtering
    grouped = df.groupby(group_cols)
    
    for name, group in tqdm(grouped, desc="Filtering groups"):
        for col in metric_cols:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            # Update the mask: keep if NOT an outlier
            mask &= ~((df.index.isin(group.index)) & (df[col] > upper_bound))
            
    return df[mask]

def aggregate_results(df):
    """
    Aggregates the cleaned data to calculate summary statistics.
    """
    print("Aggregating results...")
    
    # Define aggregation functions
    agg_funcs = {
        'rotation_error_deg': [('rmse', lambda x: np.sqrt(np.mean(x**2))), ('median', 'median')],
        'translation_error_m': [('rmse', lambda x: np.sqrt(np.mean(x**2))), ('median', 'median')],
        'c2c_rmse': [('mean', 'mean')],
        'chamfer': [('mean', 'mean')],
        'hausdorff': [('mean', 'mean')]
    }
    
    # Group and aggregate
    grouped = df.groupby(['dataset', 'algorithm', 'angle'])
    aggregated_data = grouped.agg(agg_funcs).reset_index()

    # Flatten MultiIndex columns
    aggregated_data.columns = ['_'.join(col).strip() for col in aggregated_data.columns.values]
    
    # Restructure into the desired nested dictionary format
    final_results = {}
    for _, row in tqdm(aggregated_data.iterrows(), total=len(aggregated_data), desc="Restructuring data"):
        dataset = row['dataset_']
        algo = row['algorithm_']
        angle = str(row['angle_'])

        if dataset not in final_results:
            final_results[dataset] = {}
        if algo not in final_results[dataset]:
            final_results[dataset][algo] = {}
        
        final_results[dataset][algo][angle] = {
            'rotation': {
                'rmse': row['rotation_error_deg_rmse'],
                'median': row['rotation_error_deg_median']
            },
            'translation': {
                'rmse': row['translation_error_m_rmse'],
                'median': row['translation_error_m_median']
            },
            'registration': {
                'c2c_rmse': row['c2c_rmse_mean'],
                'chamfer': row['chamfer_mean'],
                'hausdorff': row['hausdorff_mean']
            }
        }
        
    return final_results

ALGO_ORDER = ['ICP', 'DCP_transformer', 'PointNetLK', 'PointNetLK_Revisited', 'PointNetLK_c3vd_mamba3d_v1']
ALGO_DISPLAY_NAMES = {
    'ICP': 'ICP',
    'DCP_transformer': 'DCP',
    'PointNetLK': 'PointNetLK',
    'PointNetLK_Revisited': 'PointNetLK_revisited',
    'PointNetLK_c3vd_mamba3d_v1': 'Mamba3D'
}

def create_markdown_table(data):
    """
    Creates a markdown table from the aggregated results.
    """
    print("Generating Markdown table...")
    lines = []
    angles_of_interest = [20, 40, 60, 80]

    for dataset_name, algorithms in data.items():
        lines.append(f"### {dataset_name} Dataset (Filtered)")
        header = "| Algorithm | Angle | Rotation RMSE (deg) | Rotation Median (deg) | Translation RMSE (m) | Translation Median (m) | C2C RMSE | Chamfer | Hausdorff |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for original_algo_name in ALGO_ORDER:
            if original_algo_name in algorithms:
                results = algorithms[original_algo_name]
                display_name = ALGO_DISPLAY_NAMES.get(original_algo_name, original_algo_name)

                for angle in angles_of_interest:
                    angle_str = str(angle)
                    if angle_str in results:
                        metrics = results[angle_str]
                        reg_metrics = metrics.get('registration', {})
                        row = (f"| {display_name} | {angle} | "
                               f"{metrics['rotation']['rmse']:.4f} | {metrics['rotation']['median']:.4f} | "
                               f"{metrics['translation']['rmse']:.4f} | {metrics['translation']['median']:.4f} | "
                               f"{reg_metrics.get('c2c_rmse', -1):.4f} | {reg_metrics.get('chamfer', -1):.4f} | {reg_metrics.get('hausdorff', -1):.4f} |")
                        lines.append(row)
        lines.append("") 
    
    return "\n".join(lines)

def main():
    input_json = 'modelnet_results_fixed.json'
    output_md = 'modelnet_summary.md'
    
    # 1. Flatten data from large JSON
    records = flatten_json_data(input_json)
    if not records:
        print("No records found. Exiting.")
        return
        
    df = pd.DataFrame(records)

    # 2. Clean and prepare data
    metric_cols = ['rotation_error_deg', 'translation_error_m', 'c2c_rmse', 'chamfer', 'hausdorff']
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid/placeholder data before filtering
    df.dropna(subset=metric_cols, inplace=True)
    df = df[(df['chamfer'] >= 0) & (df['hausdorff'] >= 0)]

    # 3. Filter outliers
    filtered_df = filter_outliers_iqr(df, ['algorithm', 'angle'], ['chamfer', 'hausdorff'])
    
    # 4. Aggregate results
    aggregated_data = aggregate_results(filtered_df)


    # 5. Create and save Markdown table
    markdown_content = create_markdown_table(aggregated_data)
    
    try:
        with open(output_md, 'w') as f:
            f.write(markdown_content)
        print(f"Successfully created summary table at {output_md}")
    except IOError as e:
        print(f"Error writing to {output_md}: {e}")

if __name__ == "__main__":
    main()
