from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.interpolate import griddata


LINEWIDTH = 5
FONTSIZE = 16


def plot_metric_with_error(agg_metrics, metric_mean_col, metric_std_col, metric_name, y_label, loss_type):
    sns.set_theme(style="whitegrid", context="talk")

    eps_ctr_values = agg_metrics['eps_ctr'].unique()

    for eps_ctr in eps_ctr_values:
        plt.figure(figsize=(12, 12))

        subset = agg_metrics[agg_metrics['eps_ctr'] == eps_ctr]

        # Simple bidder
        simple_data = subset[subset['bidder_type'] == 'simple']
        plt.errorbar(
            simple_data['eps_cvr'], simple_data[metric_mean_col],
            yerr=simple_data[metric_std_col],
            fmt='s--', label='Simple', color='crimson',
            capsize=16, capthick=8, linewidth=8, markersize=16
        )

        # Robust bidder
        robust_data = subset[subset['bidder_type'] == 'robust']
        plt.errorbar(
            robust_data['eps_cvr'], robust_data[metric_mean_col],
            yerr=robust_data[metric_std_col],
            fmt='s--', label='Robust', color='royalblue',
            capsize=16, capthick=8, linewidth=8, markersize=16
        )

        plt.xlabel('Eps CVR', fontsize=32, labelpad=10)
        plt.ylabel(y_label, fontsize=32, labelpad=10)

        plt.title(f'Eps CTR = {eps_ctr}', fontsize=32, pad=20)

        plt.legend(fontsize=32, frameon=True, loc='best', fancybox=True, shadow=True)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xscale('log')
        plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'../results/{metric_mean_col}_bat_{loss_type.lower()}_ctr{eps_ctr}.png')
        plt.show()


def plot_2d_heatmaps_interpolated_old(data_path, name_prefix):
    df = pd.read_csv(data_path)

    df_non_robust = df[df['bidder_type'] == 'simple'].copy()
    df_robust = df[df['bidder_type'] == 'robust'].copy()

    metrics = [
        {
            'name': 'tvc',
            'title': 'Total Conversions',
            'zlabel': 'Number of Conversions'
        },
        {
            'name': 'cpc_avg',
            'title': 'Average Cost per Click (CPC)',
            'zlabel': 'Cost ($)'
        }
    ]

    for metric in metrics:
        x_nr_log = np.log10(df_non_robust['eps_ctr'].values)
        y_nr_log = np.log10(df_non_robust['eps_cvr'].values)
        z_nr = df_non_robust[f'mean_{metric["name"]}'].values

        x_r_log = np.log10(df_robust['eps_ctr'].values)
        y_r_log = np.log10(df_robust['eps_cvr'].values)
        z_r = df_robust[f'mean_{metric["name"]}'].values

        x_min = min(np.min(x_nr_log), np.min(x_r_log))
        x_max = max(np.max(x_nr_log), np.max(x_r_log))
        y_min = min(np.min(y_nr_log), np.min(y_r_log))
        y_max = max(np.max(y_nr_log), np.max(y_r_log))

        grid_x_log = np.linspace(x_min, x_max, 200)
        grid_y_log = np.linspace(y_min, y_max, 200)
        grid_x, grid_y = np.meshgrid(grid_x_log, grid_y_log)

        points_nr = np.column_stack((x_nr_log, y_nr_log))
        grid_z_nr = griddata(points_nr, z_nr, (grid_x, grid_y), method='cubic')

        points_r = np.column_stack((x_r_log, y_r_log))
        grid_z_r = griddata(points_r, z_r, (grid_x, grid_y), method='cubic')

        z_all = np.concatenate([z_nr, z_r])
        norm = mcolors.Normalize(vmin=np.nanmin(z_all), vmax=np.nanmax(z_all))
        cmap = plt.get_cmap('viridis')

        fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        axs[0].imshow(grid_z_nr,
                      origin='lower',
                      extent=(x_min, x_max, y_min, y_max),
                      aspect='auto',
                      cmap=cmap,
                      norm=norm)

        im1 = axs[1].imshow(grid_z_r,
                            origin='lower',
                            extent=(x_min, x_max, y_min, y_max),
                            aspect='auto',
                            cmap=cmap,
                            norm=norm)

        def format_ticks(log_vals):
            return [f"{10**val:.1e}" for val in log_vals]

        for ax in axs:
            ax.set_xlabel('Uncertainty Parameter CTR (ε_CTR)')
            ax.set_ylabel('Uncertainty Parameter CVR (ε_CVR)')

            xticks_log = np.linspace(x_min, x_max, 6)
            ax.set_xticks(xticks_log)
            ax.set_xticklabels(format_ticks(xticks_log))

            yticks_log = np.linspace(y_min, y_max, 6)
            ax.set_yticks(yticks_log)
            ax.set_yticklabels(format_ticks(yticks_log))

        axs[0].set_title(f'Simple: {metric["title"]}')
        axs[1].set_title(f'Robust: {metric["title"]}')

        cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.9)
        cbar.set_label(metric['zlabel'])

        plt.suptitle(f"{metric['title']} vs ε_CTR and ε_CVR (Interpolated Heatmap)", fontsize=16)

        plt.savefig(f"../results/{name_prefix}_{metric['name']}_2d_heatmap.png", dpi=300)
        plt.show()


def plot_2d_heatmaps_interpolated(data_path, name_prefix):
    sns.reset_orig()
    df = pd.read_csv(data_path)

    df_non_robust = df[df['bidder_type'] == 'simple'].copy()
    df_robust = df[df['bidder_type'] == 'robust'].copy()

    metrics = [
        {
            'name': 'tvc',
            'title': 'TCV',
            'zlabel': 'Number of Conversions'
        },
        {
            'name': 'cpc_avg',
            'title': 'CPC',
            'zlabel': 'Cost'
        }
    ]

    for metric in metrics:
        x_nr_log = np.log10(df_non_robust['eps_ctr'].values)
        y_nr_log = np.log10(df_non_robust['eps_cvr'].values)
        z_nr = df_non_robust[f'mean_{metric["name"]}'].values

        x_r_log = np.log10(df_robust['eps_ctr'].values)
        y_r_log = np.log10(df_robust['eps_cvr'].values)
        z_r = df_robust[f'mean_{metric["name"]}'].values

        x_min = min(np.min(x_nr_log), np.min(x_r_log))
        x_max = max(np.max(x_nr_log), np.max(x_r_log))
        y_min = min(np.min(y_nr_log), np.min(y_r_log))
        y_max = max(np.max(y_nr_log), np.max(y_r_log))

        grid_x_log = np.linspace(x_min, x_max, 200)
        grid_y_log = np.linspace(y_min, y_max, 200)
        grid_x, grid_y = np.meshgrid(grid_x_log, grid_y_log)

        points_nr = np.column_stack((x_nr_log, y_nr_log))
        grid_z_nr = griddata(points_nr, z_nr, (grid_x, grid_y), method='cubic')

        points_r = np.column_stack((x_r_log, y_r_log))
        grid_z_r = griddata(points_r, z_r, (grid_x, grid_y), method='cubic')

        z_all = np.concatenate([z_nr, z_r])
        norm = mcolors.Normalize(vmin=np.nanmin(z_all), vmax=np.nanmax(z_all))
        cmap = plt.get_cmap('plasma')

        fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        axs[0].imshow(grid_z_nr,
                      origin='lower',
                      extent=(x_min, x_max, y_min, y_max),
                      aspect='auto',
                      cmap=cmap,
                      norm=norm)

        im1 = axs[1].imshow(grid_z_r,
                            origin='lower',
                            extent=(x_min, x_max, y_min, y_max),
                            aspect='auto',
                            cmap=cmap,
                            norm=norm)

        def format_ticks(log_vals):
            return [f"{10**val:.1e}" for val in log_vals]

        for ax in axs:
            ax.set_xlabel(r'$\varepsilon_a$', fontsize=26)
            ax.set_ylabel(r'$\varepsilon_b$', fontsize=26)

            xticks_log = np.linspace(x_min, x_max, 6)
            ax.tick_params(axis='x', pad=10)
            ax.set_xticks(xticks_log)
            ax.set_xticklabels(format_ticks(xticks_log), fontsize=15)

            yticks_log = np.linspace(y_min, y_max, 6)
            ax.set_yticks(yticks_log)
            ax.set_yticklabels(format_ticks(yticks_log), fontsize=15)

        axs[0].set_title(f'Non-robust: {metric["title"]}', fontsize=20)
        axs[1].set_title(f'Robust: {metric["title"]}', fontsize=20)

        cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.9)
        cbar.set_label(metric['zlabel'], fontsize=20)
        cbar.ax.tick_params(labelsize=15)

        plt.savefig(f"../results/{name_prefix}_{metric['name']}_2d_heatmap.png", dpi=300)
        plt.show()


def plot_metric_with_error_CTR(eps_set, agg_metrics, metric_mean_col, metric_std_col, metric_name, y_label, loss_type):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 12))

    # Simple
    simple_data = agg_metrics[agg_metrics['bidder_type'] == 'simple']
    plt.errorbar(
        simple_data['eps'], simple_data[metric_mean_col], yerr=simple_data[metric_std_col],
        fmt='s--', label='Simple', color='crimson', capsize=16, capthick=8, linewidth=8, markersize=16
    )

    # Robust
    robust_data = agg_metrics[agg_metrics['bidder_type'] == 'robust']
    plt.errorbar(
        robust_data['eps'], robust_data[metric_mean_col], yerr=robust_data[metric_std_col],
        fmt='s--', label='Robust', color='royalblue', capsize=16, capthick=8, linewidth=8, markersize=16
    )

    plt.xlabel('Eps', fontsize=32, labelpad=10)
    plt.ylabel(y_label, fontsize=32, labelpad=10)

    plt.legend(fontsize=32, frameon=True, loc='best', fancybox=True, shadow=True)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xscale('log')
    plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'../results/{metric_mean_col}_BAT_{loss_type.lower()}.png')
    plt.show()
