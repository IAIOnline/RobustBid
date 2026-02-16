"""Module vizualization."""
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata


def plot_results_with_std(nonrobust,robust,name):
    """plot_results_with_std."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    df_non_robust = pd.read_csv(nonrobust)
    df_robust = pd.read_csv(robust)

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 1, figure=fig)
    gs.update(hspace=0.3)

    metrics = [
        {
            'name': 'total_conversions',
            'title': 'Total Conversions',
            'ylabel': 'Number of Conversions'
        },
        {
            'name': 'avg_cheap_clicks',
            'title': 'Average Cheap Clicks',
            'ylabel': 'Clicks'
        },
        {
            'name': 'avg_cpc',
            'title': 'Average Cost per Click (CPC)',
            'ylabel': 'Cost ($)'
        }
    ]

    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[idx])

        ax.errorbar(df_non_robust['epsilon'],
                   df_non_robust[f"{metric['name']}_mean"],
                   yerr=df_non_robust[f"{metric['name']}_std"],
                   color='#E24A33',  # red
                   label='Non-robust',
                   fmt='o-',
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   markersize=6)

        ax.errorbar(df_robust['epsilon'],
                   df_robust[f"{metric['name']}_mean"],
                   yerr=df_robust[f"{metric['name']}_std"],
                   color='#348ABD',  # blue
                   label='Robust',
                   fmt='s-',
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   markersize=6)

        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        ax.set_xlabel('Uncertainty Parameter (ε)', fontsize=10)
        ax.set_ylabel(metric['ylabel'], fontsize=10)
        ax.set_title(metric['title'], fontsize=12, pad=10)


        ax.legend(frameon=True,
                 fancybox=True,
                 shadow=True,
                 loc='best',
                 fontsize=9)


        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


        ax.tick_params(axis='both',
                      which='major',
                      labelsize=9,
                      width=1.5,
                      length=6)
        ax.tick_params(axis='both',
                      which='minor',
                      width=1,
                      length=3)


    fig.suptitle('Comparison of Robust vs Non-robust Strategies',
                fontsize=14,
                y=0.95)


    plt.tight_layout()


    plt.savefig(name,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')

    plt.show()



def plot_results_with_std_ctr_cvr(nonrobust, robust, _name_prefix):
    """plot_results_with_std_ctr_cvr."""
    plt.style.use('seaborn-paper')
    sns.set_palette("husl")

    df_non_robust = pd.read_csv(nonrobust)
    df_robust = pd.read_csv(robust)

    # Get all unique epsilon_CVR values from the union of both dataframes
    unique_epsilon_cvr = pd.concat([
        df_non_robust['epsilon_CVR'],
        df_robust['epsilon_CVR']
    ]).unique()
    unique_epsilon_cvr.sort()

    # Metrics to plot
    metrics = [
        {
            'name': 'total_conversions',
            'title': 'Total Conversions',
            'ylabel': 'Number of Conversions'
        },
        {
            'name': 'avg_cpc',
            'title': 'Average Cost per Click (CPC)',
            'ylabel': 'Cost ($)'
        }
    ]

    for epsilon_cvr_val in unique_epsilon_cvr:
        # Filter data by current epsilon_CVR
        df_non_robust_filtered = df_non_robust[df_non_robust['epsilon_CVR'] == epsilon_cvr_val]
        df_robust_filtered = df_robust[df_robust['epsilon_CVR'] == epsilon_cvr_val]

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(len(metrics), 1, figure=fig)
        gs.update(hspace=0.3)

        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx])

            # Check that there is data to plot
            if not df_non_robust_filtered.empty:
                ax.errorbar(
                    df_non_robust_filtered['epsilon_CTR'],
                    df_non_robust_filtered[f"{metric['name']}_mean"],
                    yerr=df_non_robust_filtered[f"{metric['name']}_std"],
                    color='#E24A33',  # red
                    label='Non-robust',
                    fmt='o-',
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=6
                )

            if not df_robust_filtered.empty:
                ax.errorbar(
                    df_robust_filtered['epsilon_CTR'],
                    df_robust_filtered[f"{metric['name']}_mean"],
                    yerr=df_robust_filtered[f"{metric['name']}_std"],
                    color='#348ABD',  # blue
                    label='Robust',
                    fmt='s-',
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=6
                )

            ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)

            ax.set_xlabel('Uncertainty Parameter CTR (ε_CTR)', fontsize=10)
            ax.set_ylabel(metric['ylabel'], fontsize=10)
            ax.set_title(f"{metric['title']} (ε_CVR = {epsilon_cvr_val})", fontsize=12, pad=10)

            ax.legend(frameon=True,
                      fancybox=True,
                      shadow=True,
                      loc='best',
                      fontsize=9)

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            ax.tick_params(axis='both',
                           which='major',
                           labelsize=9,
                           width=1.5,
                           length=6)
            ax.tick_params(axis='both',
                           which='minor',
                           width=1,
                           length=3)

        fig.suptitle(f'Comparison of Robust vs Non-robust Strategies\nfor ε_CVR = {epsilon_cvr_val}',
                     fontsize=14,
                     y=0.95)



def plot_3d_ctr_cvr(nonrobust_path, robust_path, name_prefix, el, az):
    """plot_3d_ctr_cvr."""
    df_non_robust = pd.read_csv(nonrobust_path)
    df_robust = pd.read_csv(robust_path)

    metrics = [
        {
            'name': 'total_conversions',
            'title': 'Total Conversions',
            'zlabel': 'Number of Conversions'
        },
        {
            'name': 'avg_cpc',
            'title': 'Average Cost per Click (CPC)',
            'zlabel': 'Cost ($)'
        }
    ]

    for metric in metrics:
        x_nr = df_non_robust['epsilon_CTR']
        y_nr = df_non_robust['epsilon_CVR']
        z_nr = df_non_robust[f'{metric["name"]}_mean']

        x_r = df_robust['epsilon_CTR']
        y_r = df_robust['epsilon_CVR']
        z_r = df_robust[f'{metric["name"]}_mean']

        # Define axis bounds
        x_range = [5e-7, 2e-2]
        y_range = [5e-7, 2e-2]
        z_min = min(z_nr.min(), z_r.min())
        z_max = max(z_nr.max(), z_r.max())
        z_range = [z_min * 0.95, z_max * 1.05]

        fig = go.Figure()

        # Add non-robust points
        fig.add_trace(go.Scatter3d(
            x=x_nr, y=y_nr, z=z_nr,
            mode='markers',
            marker={'size': 6, 'color': 'red', 'opacity': 0.8},
            name='Non-robust'
        ))

        # Add robust points
        fig.add_trace(go.Scatter3d(
            x=x_r, y=y_r, z=z_r,
            mode='markers',
            marker={'size': 6, 'color': 'blue', 'opacity': 0.8},
            name='Robust'
        ))

        # Set axis parameters with logarithmic scale for X and Y
        fig.update_layout(
            scene={
                'xaxis': {
                    'title': 'Uncertainty Parameter CTR (ε_CTR)',
                    'type': 'log',
                    'range': [np.log10(x_range[0]), np.log10(x_range[1])],
                    'autorange': False,
                    'zeroline': False,
                    'showspikes': False,
                    'tickformat': ".0e"
                },
                'yaxis': {
                    'title': 'Uncertainty Parameter CVR (ε_CVR)',
                    'type': 'log',
                    'range': [np.log10(y_range[0]), np.log10(y_range[1])],
                    'autorange': False,
                    'zeroline': False,
                    'showspikes': False,
                    'tickformat': ".0e"
                },
                'zaxis': {
                    'title': metric['zlabel'],
                    'range': z_range,
                    'autorange': False,
                    'zeroline': False,
                    'showspikes': False
                },
                'camera': {
                    'eye': {
                        'x': np.cos(np.radians(az)) * np.cos(np.radians(el)),
                        'y': np.sin(np.radians(az)) * np.cos(np.radians(el)),
                        'z': np.sin(np.radians(el))
                    }
                }
            },
            title=f"{metric['title']} vs ε_CTR and ε_CVR",
            legend={'font': {'size': 12}},
            width=900,
            height=700
        )

        # Save to html (interactive) and png (static) files
        fig.write_html(f"{name_prefix}_{metric['name']}_3d_scatter.html")
        fig.write_image(f"{name_prefix}_{metric['name']}_3d_scatter.png", scale=2)

        fig.show()


def plot_2d_heatmaps(nonrobust_path, robust_path, name_prefix):
    """plot_2d_heatmaps."""
    df_non_robust = pd.read_csv(nonrobust_path)
    df_robust = pd.read_csv(robust_path)

    metrics = [
        {
            'name': 'total_conversions',
            'title': 'Total Conversions',
            'zlabel': 'Conversions'
        },
        {
            'name': 'avg_cpc',
            'title': 'Average Cost per Click (CPC)',
            'zlabel': 'Cost'
        }
    ]

    for metric in metrics:
        # Non-robust data
        x_nr = df_non_robust['epsilon_CTR']
        y_nr = df_non_robust['epsilon_CVR']
        z_nr = df_non_robust[f'{metric["name"]}_mean']

        # Robust data
        x_r = df_robust['epsilon_CTR']
        y_r = df_robust['epsilon_CVR']
        z_r = df_robust[f'{metric["name"]}_mean']

        # Common value range for color normalization
        z_all = np.concatenate([z_nr.values, z_r.values])
        z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)

        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=z_min, vmax=z_max)

        fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        # Non-robust scatter
        axs[0].scatter(x_nr, y_nr, c=z_nr, cmap=cmap, norm=norm,
                             s=400, edgecolor='black', linewidth=0.7)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Uncertainty Parameter CTR (ε_CTR)')
        axs[0].set_ylabel('Uncertainty Parameter CVR (ε_CVR)')
        axs[0].set_title(f'Non-robust: {metric["title"]}')

        # Robust scatter
        sc1 = axs[1].scatter(x_r, y_r, c=z_r, cmap=cmap, norm=norm,
                             s=400, edgecolor='black', linewidth=0.7)
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Uncertainty Parameter CTR (ε_CTR)')
        axs[1].set_ylabel('Uncertainty Parameter CVR (ε_CVR)')
        axs[1].set_title(f'Robust: {metric["title"]}')

        # Shared colorbar on the right
        cbar = fig.colorbar(sc1, ax=axs.ravel().tolist(), shrink=0.9)
        cbar.set_label(metric['zlabel'])

        plt.suptitle(f"{metric['title']} vs ε_CTR and ε_CVR (Colored Points Only)", fontsize=16)

        plt.savefig(f"{name_prefix}_{metric['name']}_2d_scatter_colored_points.png", dpi=300)
        plt.show()


def plot_2d_heatmaps_interpolated(nonrobust_path, robust_path, name_prefix):
    """plot_2d_heatmaps_interpolated."""
    df_non_robust = pd.read_csv(nonrobust_path)
    df_robust = pd.read_csv(robust_path)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    metrics = [
        {
            'name': 'total_conversions',
            'title': 'TCV',
            'zlabel': 'Total Conversion Value'
        },
        {
            'name': 'avg_cpc',
            'title': 'CPC',
            'zlabel': 'Average Cost-Per-Click'
        }
    ]

    for metric in metrics:
        # Log-transform coordinates
        x_nr_log = np.log10(df_non_robust['epsilon_CTR'].values)
        y_nr_log = np.log10(df_non_robust['epsilon_CVR'].values)
        z_nr = df_non_robust[f'{metric["name"]}_mean'].values

        x_r_log = np.log10(df_robust['epsilon_CTR'].values)
        y_r_log = np.log10(df_robust['epsilon_CVR'].values)
        z_r = df_robust[f'{metric["name"]}_mean'].values

        # Grid ranges in log-space
        x_min = min(np.min(x_nr_log), np.min(x_r_log))
        x_max = max(np.max(x_nr_log), np.max(x_r_log))
        y_min = min(np.min(y_nr_log), np.min(y_r_log))
        y_max = max(np.max(y_nr_log), np.max(y_r_log))

        grid_x_log = np.linspace(x_min, x_max, 200)
        grid_y_log = np.linspace(y_min, y_max, 200)
        grid_x, grid_y = np.meshgrid(grid_x_log, grid_y_log)

        # Interpolation in log-space
        points_nr = np.column_stack((x_nr_log, y_nr_log))
        grid_z_nr = griddata(points_nr, z_nr, (grid_x, grid_y), method='cubic')

        points_r = np.column_stack((x_r_log, y_r_log))
        grid_z_r = griddata(points_r, z_r, (grid_x, grid_y), method='cubic')

        # Color normalization over the common range
        z_all = np.concatenate([z_nr, z_r])
        norm = mcolors.Normalize(vmin=np.nanmin(z_all), vmax=np.nanmax(z_all))
        cmap = plt.get_cmap('plasma')

        # Axis labels — display values in original scale (10^x)
        def format_ticks(log_vals):
            """format_ticks."""
            return [f"{10**val:.1e}" for val in log_vals]

        # Create two separate plots with identical dimensions
        for _idx, (grid_z, title, suffix) in enumerate([
            (grid_z_nr, 'NonRobustBid', 'nonrobust'),
            (grid_z_r, 'RobustBid (our)', 'robust')
        ]):
            # Same size for both plots
            fig, ax = plt.subplots(1, 1, figsize=(9, 9), constrained_layout=True)

            ax.imshow(grid_z,
                          origin='lower',
                          extent=(x_min, x_max, y_min, y_max),
                          aspect='equal',
                          cmap=cmap,
                          norm=norm)

            ax.set_xlabel(r'$\varepsilon_a$', fontsize=42)
            ax.set_ylabel(r'$\varepsilon_b$', fontsize=42)

            # Configure ticks on both axes
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=26,
                width=2.5,
                length=8,
                pad=12,
                direction='inout'
            )

            # X-axis ticks
            xticks_log = np.linspace(x_min, x_max, 4)
            ax.set_xticks(xticks_log)
            ax.set_xticklabels(
                format_ticks(xticks_log),
                fontsize=26
            )

            # Y-axis ticks
            yticks_log = np.linspace(y_min, y_max, 4)
            ax.set_yticks(yticks_log)
            ax.set_yticklabels(
                format_ticks(yticks_log),
                fontsize=26
            )

            ax.set_title(title, fontsize=36, pad=15)
            plt.tight_layout()
            plt.savefig(f"results/{metric['title']}_{suffix}_"+name_prefix, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # Create a separate image with only the colorbar
        fig, ax = plt.subplots(1, 1, figsize=(2, 9))
        ax.axis('off')  # Hide axes

        # Create colorbar only, without an image
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=1.0)
        cbar.set_label(metric['zlabel'], fontsize=48, labelpad=15)
        cbar.ax.tick_params(labelsize=24, width=2, length=6)

        # Remove left margin, make colorbar full width
        cbar.ax.set_position([0, 0, 1, 1])

        plt.savefig(f"results/{metric['title']}_colorbar_"+name_prefix, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()


def plot_2d_heatmaps_interpolated_katya(data_path, name_prefix):
    """plot_2d_heatmaps_interpolated_katya."""
    sns.reset_orig()
    df = pd.read_csv(data_path)

    os.makedirs('results', exist_ok=True)

    df_non_robust = df[df['bidder_type'] == 'simple'].copy()
    df_robust = df[df['bidder_type'] == 'robust'].copy()

    metrics = [
        {
            'name': 'tvc',
            'title': 'TCV',
            'zlabel': 'Total Conversion Value'
        },
        {
            'name': 'cpc_avg',
            'title': 'CPC',
            'zlabel': 'Average Cost-Per-Click'
        }
    ]

    for metric in metrics:
        x_nr_log = np.log10(df_non_robust['eps_ctr'].values)
        y_nr_log = np.log10(df_non_robust['eps_cvr'].values)
        z_nr = df_non_robust[f'mean_{metric["name"]}'].values

        x_r_log = np.log10(df_robust['eps_ctr'].values)
        y_r_log = np.log10(df_robust['eps_cvr'].values)
        z_r = df_robust[f'mean_{metric["name"]}'].values

        # Grid ranges in log-space
        x_min = min(np.min(x_nr_log), np.min(x_r_log))
        x_max = max(np.max(x_nr_log), np.max(x_r_log))
        y_min = min(np.min(y_nr_log), np.min(y_r_log))
        y_max = max(np.max(y_nr_log), np.max(y_r_log))

        grid_x_log = np.linspace(x_min, x_max, 200)
        grid_y_log = np.linspace(y_min, y_max, 200)
        grid_x, grid_y = np.meshgrid(grid_x_log, grid_y_log)

        # Interpolation in log-space
        points_nr = np.column_stack((x_nr_log, y_nr_log))
        grid_z_nr = griddata(points_nr, z_nr, (grid_x, grid_y), method='cubic')

        points_r = np.column_stack((x_r_log, y_r_log))
        grid_z_r = griddata(points_r, z_r, (grid_x, grid_y), method='cubic')

        # Color normalization over the common range
        z_all = np.concatenate([z_nr, z_r])
        norm = mcolors.Normalize(vmin=np.nanmin(z_all), vmax=np.nanmax(z_all))
        cmap = plt.get_cmap('plasma')

        # Axis labels — display values in original scale (10^x)
        def format_ticks(log_vals):
            """format_ticks."""
            return [f"{10**val:.1e}" for val in log_vals]

        # Create two separate plots with identical dimensions
        for _idx, (grid_z, title, suffix) in enumerate([
            (grid_z_nr, 'NonRobustBid', 'nonrobust'),
            (grid_z_r, 'RobustBid (our)', 'robust')
        ]):
            # Same size for both plots
            fig, ax = plt.subplots(1, 1, figsize=(9, 9), constrained_layout=True)

            ax.imshow(grid_z,
                          origin='lower',
                          extent=(x_min, x_max, y_min, y_max),
                          aspect='equal',
                          cmap=cmap,
                          norm=norm)

            ax.set_xlabel(r'$\varepsilon_a$', fontsize=42)
            ax.set_ylabel(r'$\varepsilon_b$', fontsize=42)

            # Configure ticks on both axes
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=26,
                width=2.5,
                length=8,
                pad=12,
                direction='inout'
            )

            # X-axis ticks
            xticks_log = np.linspace(x_min, x_max, 4)
            ax.set_xticks(xticks_log)
            ax.set_xticklabels(
                format_ticks(xticks_log),
                fontsize=26
            )

            # Y-axis ticks
            yticks_log = np.linspace(y_min, y_max, 4)
            ax.set_yticks(yticks_log)
            ax.set_yticklabels(
                format_ticks(yticks_log),
                fontsize=26
            )

            ax.set_title(title, fontsize=36, pad=15)
            plt.tight_layout()
            plt.savefig(f"results/{metric['title']}_{suffix}_"+name_prefix, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # Create a separate image with only the colorbar
        fig, ax = plt.subplots(1, 1, figsize=(2, 9))
        ax.axis('off')  # Hide axes

        # Create colorbar only, without an image
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=1.0)
        cbar.set_label(metric['zlabel'], fontsize=48, labelpad=15)
        cbar.ax.tick_params(labelsize=24, width=2, length=6)

        # Remove left margin, make colorbar full width
        cbar.ax.set_position([0, 0, 1, 1])

        plt.savefig(f"results/{metric['title']}_colorbar_"+name_prefix, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
