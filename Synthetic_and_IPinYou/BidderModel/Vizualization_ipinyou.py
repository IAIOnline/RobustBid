import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def visualize_bid_distributions(bid_data, bid_generator, num_generated=1000):
    """
    Visualize original and generated bids for each advertiser
    """
    sns.set_theme(style="whitegrid")

    num_advertisers = len(bid_data)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_advertisers))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    generated_bids_all = {}

    for idx, (advertiser_id, bids) in enumerate(bid_data.items()):
        original_bids = np.array(bids)
        generated_bids = []
        
        # Generate bids using the bid_generator instance
        for _ in range(num_generated):
            generated_bids.append(bid_generator.generate_bid(advertiser_id, bid_data))
        generated_bids = np.array(generated_bids)
        generated_bids_all[advertiser_id] = generated_bids

        # Plot histograms
        sns.histplot(data=original_bids, bins=50, alpha=0.3, color=colors[idx],
                     label=f'Original Bids (Adv {advertiser_id})', ax=ax1,
                     stat='density')
        sns.histplot(data=generated_bids, bins=50, alpha=0.3, color=colors[idx],
                     linestyle='--', label=f'Generated Bids (Adv {advertiser_id})',
                     ax=ax1, stat='density')

        # Plot KDE
        # Use cached KDE if available
        if advertiser_id in bid_generator.cached_kdes:
            kde_original = bid_generator.cached_kdes[advertiser_id]
            x_range = bid_generator.cached_x_ranges[advertiser_id]
        else:
            kde_original = stats.gaussian_kde(original_bids)
            x_range = np.linspace(min(original_bids), max(original_bids), 1000)

        kde_generated = stats.gaussian_kde(generated_bids)

        ax2.plot(x_range, kde_original(x_range), color=colors[idx],
                 label=f'Original KDE (Adv {advertiser_id})')
        ax2.plot(x_range, kde_generated(x_range), color=colors[idx],
                 linestyle='--', label=f'Generated KDE (Adv {advertiser_id})')

    # Customize plots
    ax1.set_title('Histogram of Original vs Generated Bids')
    ax1.set_xlabel('Bid Value')
    ax1.set_ylabel('Density')
    ax1.legend()

    ax2.set_title('KDE of Original vs Generated Bids')
    ax2.set_xlabel('Bid Value')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Statistical comparison
    print("\nStatistical Comparison:")
    print("-----------------------")
    for advertiser_id in bid_data.keys():
        original = np.array(bid_data[advertiser_id])
        generated = generated_bids_all[advertiser_id]

        print(f"\nAdvertiser {advertiser_id}:")
        print(f"Original  - Mean: {np.mean(original):.2f}, Std: {np.std(original):.2f}")
        print(f"Generated - Mean: {np.mean(generated):.2f}, Std: {np.std(generated):.2f}")

        ks_stat, p_value = stats.ks_2samp(original, generated)
        print(f"KS-test p-value: {p_value:.4f}")


def plot_bid_time_series(bid_data, bid_generator, num_points=100):
    """
    Plot time series of original and generated bids
    """
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(15, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bid_data)))

    for idx, (advertiser_id, bids) in enumerate(bid_data.items()):
        # Original bids
        plt.plot(range(min(len(bids), num_points)),
                 bids[:num_points], 
                 color=colors[idx], alpha=0.5,
                 label=f'Original Bids (Adv {advertiser_id})')

        # Generated bids
        generated_bids = [bid_generator.generate_bid(advertiser_id, bid_data)
                          for _ in range(num_points)]
        plt.plot(range(len(generated_bids)), generated_bids,
                 color=colors[idx], linestyle='--', alpha=0.5, label=f'Generated Bids (Adv {advertiser_id})')

    plt.title('Time Series of Original vs Generated Bids')
    plt.xlabel('Bid Sequence')
    plt.ylabel('Bid Value')
    plt.legend()
    plt.show()


# Usage example:
# def ipinyou_bids_visualization(bid_data):
#     # Create bid generator instance
#     bid_generator = BidGenerator()

#     # Run visualizations
#     visualize_bid_distributions(bid_data, bid_generator)
#     plot_bid_time_series(bid_data, bid_generator)


# # Run the analysis
# if __name__ == "__main__":
#     file_list = ['bid_ipinyou_19.txt', 'bid_ipinyou_11.txt', 'bid_ipinyou.txt', 'bid_ipinyou_25.txt', 'bid_ipinyou_28.txt']
#     bid_data = process_ipinyou_files(file_list)
#     run_visualization(bid_data)
