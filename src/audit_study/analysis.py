import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from scipy.stats import norm
from matplotlib.colors import to_rgba

# Basic plotting settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

class VLMAnalysis:
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self, file_path):
        """Loads JSONL results and prepares them for bias calculation"""
        records = []
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    res = json.loads(line)
                    sp, ps = res.get('sample_pair', {}), res.get('parsed_response', {})
                    records.append({
                        'attribute': sp.get('attribute'),
                        'value1': sp.get('value1'),
                        'value2': sp.get('value2'),
                        'task_a_assignment': ps.get('task_a_assignment'),
                        'parsing_successful': ps.get('parsing_successful', False)
                    })
                except: continue
        return pd.DataFrame(records)

    def calculate_bias(self, df):
        """Calculates BBI (Bias Index) stats, confidence intervals, and significance"""
        if df.empty: return pd.DataFrame()
        gamma = np.exp(-1) # Penalty for failed parsing
        records = []
        for _, row in df.iterrows():
            attr, v1, v2 = row['attribute'], row['value1'], row['value2']
            success = row['parsing_successful']
            w_i = 1.0 if success else gamma
            phi_v1 = 0.5
            if success:
                # '学生A' refers to Student A in the original audit response
                phi_v1 = 1.0 if row['task_a_assignment'] == '学生A' else 0.0
            
            records.append({'attribute': attr, 'value': v1, 'phi': phi_v1, 'weight': w_i})
            records.append({'attribute': attr, 'value': v2, 'phi': 1.0 - phi_v1 if success else 0.5, 'weight': w_i})

        temp_df = pd.DataFrame(records)
        stats = []
        for (attr, val), group in temp_df.groupby(['attribute', 'value'], sort=False):
            num = (group['weight'] * group['phi']).sum()
            den = group['weight'].sum()
            bbi = num / den if den != 0 else 0.5
            var = np.average((group['phi'] - bbi)**2, weights=group['weight'])
            se = np.sqrt(var / len(group)) if len(group) > 1 else 0.0001
            z = abs(bbi - 0.5) / se
            p = 2 * (1 - norm.cdf(z))
            # Significance stars
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            stats.append({
                'attribute': attr, 'value': val, 'prob': bbi, 
                'low': max(0, bbi - 1.96 * se), 'high': min(1, bbi + 1.96 * se), 'sig': sig
            })
        return pd.DataFrame(stats)

    def _draw_forest_on_ax(self, ax, bias_df, model_name, color_map, show_labels=False):
        """Internal helper to draw a forest plot on a specific axis"""
        if bias_df.empty: return
        
        STEP = 0.8  # Increase line spacing for vertical stretching
        current_y = 0
        unique_attrs = bias_df['attribute'].unique()
        
        for i, attr in enumerate(unique_attrs):
            adf = bias_df[bias_df['attribute'] == attr]
            base_c = color_map[attr]
            bg_color = to_rgba(base_c, alpha=0.1)
            line_color = to_rgba(base_c, alpha=0.9)
            
            # Draw background block for the attribute group
            block_height = len(adf) * STEP
            ax.axhspan(current_y + STEP/2, current_y - block_height + STEP/2, facecolor=bg_color, zorder=0)
            
            for idx, (_, r) in enumerate(adf.iterrows()):
                y = current_y
                # Plot confidence interval
                ax.plot([r['low'], r['high']], [y, y], color=line_color, lw=2.5, zorder=2)
                # Plot mean probability point
                ax.scatter(r['prob'], y, color=line_color, s=60, edgecolor='white', zorder=3, linewidth=0.8)
                
                # Significance stars (Font size 13)
                if r['sig'] and r['high'] + 0.02 < 1.0:
                    ax.text(r['high'] + 0.02, y, r['sig'], va='center', ha='left', 
                            color=line_color, fontweight='bold', fontsize=13)

                # Y-axis attribute labels (Font size 18, Bold)
                if show_labels:
                    ax.text(-0.06, y, r['value'], va='center', ha='right', 
                            fontsize=18, fontweight='bold', color=line_color)
                
                current_y -= STEP

        # Neutral line at 0.5
        ax.axvline(0.5, color='black', alpha=0.3, lw=1.2)
        for spine in ax.spines.values(): spine.set_visible(False)
            
        ax.set_title(model_name, fontsize=20, fontweight='bold', pad=16) # Enlarged title
        ax.set_xlim(0, 1.0) 
        ax.set_ylim(current_y + STEP/2, STEP/2)
        
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0', '0.5', '1.0'], fontsize=16, fontweight='bold') # Enlarged ticks
        ax.tick_params(axis='x', length=0)
        ax.set_yticks([])

    def generate_plots(self):
        """Orchestrates data loading and generates the final multi-model forest plot"""
        files = list(self.results_dir.glob("*.jsonl"))
        all_dfs = {f.stem.replace('_results', ''): self.calculate_bias(self.load_and_prepare_data(f)) for f in files}
        all_dfs = {k: v for k, v in all_dfs.items() if not v.empty}
        
        # Setup consistent colors for attributes across models
        unique_global_attrs = list(dict.fromkeys([a for df in all_dfs.values() for a in df['attribute'].unique()]))
        academic_colors = ['#3E7D3E', '#2E5A88', '#7D5492', '#A03D3D', '#9B6A30', '#4A8A9E', '#737373', '#B08D3E']
        color_map = {attr: academic_colors[i % len(academic_colors)] for i, attr in enumerate(unique_global_attrs)}

        if all_dfs:
            n_models = len(all_dfs)
            cols = 5
            rows = int(np.ceil(n_models / cols))
            
            # Increase vertical height (rows * 5.8)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 5.8), sharey=True, dpi=300)
            if rows == 1: axes = np.expand_dims(axes, axis=0)
            axes_flat = axes.flatten()

            for i, (m_name, b_df) in enumerate(all_dfs.items()):
                self._draw_forest_on_ax(axes_flat[i], b_df, m_name, color_map, show_labels=(i % cols == 0))

            # Clean up empty subplots
            for j in range(i + 1, len(axes_flat)): fig.delaxes(axes_flat[j])

            # Significantly enlarged Legend (Font size 20)
            legend_handles = [mpatches.Patch(color=color_map[a], label=a.title().replace('_', ' ')) for a in unique_global_attrs]
            fig.legend(
                handles=legend_handles, 
                loc='lower center',       # Bottom center
                bbox_to_anchor=(0.5, 0.02), # Fine-tuned positioning
                ncol=len(unique_global_attrs), # Horizontal display
                frameon=False, 
                prop={'size': 20, 'weight': 'bold'}
            )

            plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.98]) # Margin adjustments
            plt.savefig(self.output_dir / "audit_results.png", dpi=300, bbox_inches='tight')
            print("Plot generated with larger fonts and stretched layout.")

if __name__ == "__main__":
    # Initialize and run analysis
    VLMAnalysis(Path("data/results/audit"), Path("figure")).generate_plots()