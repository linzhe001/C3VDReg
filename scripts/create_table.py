import json

def create_markdown_table(data):
    """Creates a markdown table from the results and returns it as a string."""
    lines = []
    angles_of_interest = [20, 40, 60, 80]

    for dataset_name, algorithms in data.items():
        lines.append(f"### {dataset_name} Dataset")
        header = "| Algorithm | Angle | Rotation RMSE (deg) | Rotation Median (deg) | Translation RMSE (m) | Translation Median (m) | C2C RMSE | Chamfer | Hausdorff |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for algorithm_name, results in algorithms.items():
            for angle in angles_of_interest:
                if str(angle) in results:
                    metrics = results[str(angle)]
                    reg_metrics = metrics.get('registration', {})
                    row = (f"| {algorithm_name} | {angle} | "
                           f"{metrics['rotation']['rmse']:.4f} | {metrics['rotation']['median']:.4f} | "
                           f"{metrics['translation']['rmse']:.4f} | {metrics['translation']['median']:.4f} | "
                           f"{reg_metrics.get('c2c_rmse', -1):.4f} | {reg_metrics.get('chamfer', -1):.4f} | {reg_metrics.get('hausdorff', -1):.4f} |")
                    lines.append(row)
        lines.append("") # Add a newline for spacing between tables
    
    return "\n".join(lines)

def main():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: results.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode results.json.")
        return

    markdown_content = create_markdown_table(results)
    
    try:
        with open('analysis_summary.md', 'w') as f:
            f.write(markdown_content)
        print("Successfully updated analysis_summary.md")
    except IOError as e:
        print(f"Error writing to analysis_summary.md: {e}")

if __name__ == "__main__":
    main()