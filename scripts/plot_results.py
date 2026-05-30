
import json
import matplotlib
# Set font configuration before importing pyplot
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
# Try multiple sans-serif fonts in order of preference
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'  # DejaVu Sans is a good choice for math text with Arial

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# Rebuild font cache
fm.fontManager.__init__()

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Shadow

def plot_results(data):
    """Plots the rotation and translation errors for ModelNet40 and C3VD datasets using seaborn in a 2x2 grid layout."""

    # --- Data Preparation ---
    records = []
    for dataset_name, algorithms in data.items():
        for algorithm_name, results in algorithms.items():
            for angle, values in results.items():
                records.append({
                    "dataset_key": dataset_name,
                    "algorithm_key": algorithm_name,
                    "angle": int(angle),
                    "rotation_error": values['rotation']['mean'],
                    "translation_error": values['translation']['mean']
                })
    df = pd.DataFrame(records)

    # --- Style and Name Mapping ---
    # Ensure sans-serif font settings with normal weight for axis labels
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    algo_map = {
        'ICP': 'ICP',
        'DCP': 'DCP',
        'PointNetLK': 'PointNetLK',
        'PointNetLK_revisited': 'PointNetLK Rev.',
        'Mamba3D': 'MambaNetLK(ours)'
    }
    df['algorithm'] = df['algorithm_key'].map(algo_map)
    
    dataset_map = {
        "ModelNet": "(a) ModelNet40",
        "C3VD": "(b) C3VD-Raycasting-10k"
    }
    df['dataset'] = df['dataset_key'].map(dataset_map)
    
    df = df.sort_values(by=['dataset_key', 'algorithm_key'])

    # Fine-tuned line and marker styles (thinner lines, smaller markers)
    style_map = {
        'ICP':                  {'color': '#1f77b4',    'linestyle': '--', 'marker': 'o', 'linewidth': 0.8, 'markersize': 4},
        'DCP':                  {'color': '#ff7f0e',    'linestyle': '-.', 'marker': 's', 'linewidth': 0.8, 'markersize': 4},
        'PointNetLK':           {'color': '#2ca02c',    'linestyle': ':',  'marker': '^', 'linewidth': 0.8, 'markersize': 4},
        'PointNetLK Rev.':      {'color': '#A52A2A',    'linestyle': ':',  'marker': 'D', 'linewidth': 0.8, 'markersize': 4},
        'MambaNetLK(ours)':     {'color': '#8B4BCF',    'linestyle': '-',  'marker': 'p', 'linewidth': 1.5, 'markersize': 5}
    }
    
    hue_order = [algo_map[k] for k in ['ICP', 'DCP', 'PointNetLK', 'PointNetLK_revisited', 'Mamba3D']]
    
    # --- Setup Seaborn Style ---
    sns.set_style("whitegrid", {
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'grid.color': '#E8E8E8',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.family': 'sans-serif',
        'text.color': 'black',
    })
    
    # Explicitly set sans-serif font with enhanced rendering
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['text.antialiased'] = True
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better rendering
    
    # --- Create Figure with 2x2 Grid Layout ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    
    datasets = ["ModelNet", "C3VD"]
    error_types = [
        ("rotation_error", "Mean Rot. Error", "Avg. Rotation Error (degrees)"),
        ("translation_error", "Mean Trans. Error", "Avg. Translation Error")
    ]
    subplot_labels = [['(a)', '(b)'], ['(c)', '(d)']]
    
    for row_idx, (error_col, error_name, ylabel) in enumerate(error_types):
        for col_idx, dataset_key in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            df_subset = df[df['dataset_key'] == dataset_key]
            
            # Plot each algorithm
            for algo in hue_order:
                df_algo = df_subset[df_subset['algorithm'] == algo].sort_values('angle')
                if not df_algo.empty:
                    style = style_map[algo]
                    ax.plot(
                        df_algo['angle'], 
                        df_algo[error_col],
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        linewidth=style['linewidth'],
                        markersize=style['markersize'],
                        label=algo,
                        markerfacecolor=style['color'],
                        markeredgewidth=0.5,
                        markeredgecolor=style['color']
                    )
            
            # Set title with bold font
            dataset_display = "ModelNet40" if dataset_key == "ModelNet" else "C3VD"
            title = f"{subplot_labels[row_idx][col_idx]} {error_name} on {dataset_display}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            
            # Set labels
            ax.set_xlabel('Perturbation Angle (degrees)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            # Set ticks
            ax.set_xticks(np.arange(0, 91, 10))
            ax.set_xlim(-2, 92)
            
            # Set Y-axis to start from -5 but only show positive ticks for rotation, natural for translation
            if error_col == "rotation_error":
                y_min = -5
                ax.set_ylim(bottom=y_min)
                # Get current y-axis limits and set ticks only for positive values
                y_max = ax.get_ylim()[1]
                y_ticks = ax.get_yticks()
                # Filter to only show ticks >= 0
                positive_ticks = [tick for tick in y_ticks if tick >= 0]
                ax.set_yticks(positive_ticks)
            else:
                # Set Y-axis for translation error plots independently
                max_trans_error_subset = df_subset['translation_error'].max()
                y_top = max_trans_error_subset * 1.1
                y_bottom = -0.05 * y_top
                ax.set_ylim(bottom=y_bottom, top=y_top)

                # Hide negative ticks
                y_ticks = ax.get_yticks()
                positive_ticks = [tick for tick in y_ticks if tick >= 0]
                ax.set_yticks(positive_ticks)
            
            # Grid styling
            ax.grid(True, color='#E8E8E8', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)
            
            # Spine styling
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)
            
            # Only add legend to the first subplot (top-left)
            if row_idx == 0 and col_idx == 0:
                legend = ax.legend(
                    loc='upper left',
                    frameon=True,
                    edgecolor='black',
                    fancybox=False,
                    shadow=True,
                    framealpha=1.0,
                    fontsize=11
                )
                
                # Make "MambaNetLK(ours)" bold in legend
                for text in legend.get_texts():
                    if 'MambaNetLK(ours)' in text.get_text():
                        text.set_fontweight('bold')
                
                legend.get_frame().set_linewidth(0.8)
    
    plt.tight_layout()
    
    output_filename_pdf = 'rotation_translation_error_comparison.pdf'
    plt.savefig(output_filename_pdf, dpi=800, bbox_inches='tight', format='pdf', 
                edgecolor='none', facecolor='white')
    print(f"Plot saved as {output_filename_pdf}")

    output_filename_png = 'rotation_translation_error_comparison.png'
    plt.savefig(output_filename_png, dpi=800, bbox_inches='tight', format='png',
                edgecolor='none', facecolor='white')
    print(f"Plot saved as {output_filename_png}")

def main():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        plot_results(results)
    except FileNotFoundError:
        print("Error: results.json not found. Please run analyze_results.py first.")
    except Exception as e:
        import traceback
        print(f"An error occurred in main:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
