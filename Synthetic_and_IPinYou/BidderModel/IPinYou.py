import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os


def process_ipinyou_files(file_list):
    """Process multiple iPinYou bid log files and combine advertiser bid data."""

    column_names = ['Bid ID', 'Timestamp', 'iPinYou ID', 'User-Agent', 'IP',
                    'Region ID', 'City ID', 'Ad Exchange', 'Domain', 'URL',
                    'Anonymous URL', 'Ad Slot ID', 'Ad Slot Width', 'Ad Slot Height',
                    'Ad Slot Visibility', 'Ad Slot Format', 'Ad Slot Floor Price',
                    'Creative ID', 'Bidding Price', 'Advertiser ID', 'User Profile IDs']

    advertiser_bids = {}

    # Process each file
    for file_name in file_list:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, sep='\t', names=column_names)

            # Group by advertiser and collect bid information
            for advertiser_id in df['Advertiser ID'].unique():
                advertiser_df = df[df['Advertiser ID'] == advertiser_id]
                bids = advertiser_df['Bidding Price'].values

                if advertiser_id not in advertiser_bids:
                    advertiser_bids[advertiser_id] = []
                advertiser_bids[advertiser_id].extend(bids)

    # Create list to store rows
    rows_list = []

    for adv_id, bids in advertiser_bids.items():
        bids = np.array(bids)
        row = {
            'Advertiser ID': adv_id,
            'Mean Bid': np.mean(bids),
            'Std Bid': np.std(bids),
            'Min Bid': np.min(bids),
            'Max Bid': np.max(bids),
            'Bid Sample': ','.join(map(str, np.random.choice(bids, min(100, len(bids)))))
        }
        rows_list.append(row)

    # Create DataFrame from rows list
    result_df = pd.DataFrame(rows_list)

    result_df.to_csv('ipinyou_advertiser_bid.csv', index=False)
    return advertiser_bids


class BidGenerator:
    def __init__(self):
        self.cached_bid_data = None
        self.cached_kdes = {}
        self.cached_x_ranges = {}
        self.cached_kde_values = {}

    def load_csv_data(self):
        """Load CSV data once and cache it"""
        if self.cached_bid_data is None:
            df = pd.read_csv('ipinyou_advertiser_bid.csv')
            self.cached_bid_data = {
                row['Advertiser ID']: np.array([float(x) for x in row['Bid Sample'].split(',')])
                for _, row in df.iterrows()
            }

    def prepare_kde(self, advertiser_id, bid_sample):
        """Prepare KDE and related data for an advertiser"""
        if advertiser_id not in self.cached_kdes:
            kde = gaussian_kde(bid_sample)
            x_range = np.linspace(min(bid_sample), max(bid_sample), 1000)
            kde_values = kde(x_range)
            normalized_kde_values = kde_values/sum(kde_values)

            self.cached_kdes[advertiser_id] = kde
            self.cached_x_ranges[advertiser_id] = x_range
            self.cached_kde_values[advertiser_id] = normalized_kde_values

    def generate_bid(self, advertiser_id, N_bids=1, bid_data=None):
        """
        Generate semi-stochastic bids for given advertiser.

        Parameters:
        - advertiser_id: ID of the advertiser
        - N_bids: Number of bids to generate (default=1)
        - bid_data: Optional external bid data

        Returns:
        - numpy array of N_bids bids
        """
        if bid_data is None:
            # Use cached CSV data
            if self.cached_bid_data is None:
                self.load_csv_data()
            bid_sample = self.cached_bid_data[advertiser_id]
        else:
            bid_sample = np.array(bid_data[advertiser_id])

        # Prepare or get cached KDE data
        self.prepare_kde(advertiser_id, bid_sample)

        # Sample N_bids from the cached KDE distribution
        bids = np.random.choice(
            self.cached_x_ranges[advertiser_id],
            size=N_bids,
            p=self.cached_kde_values[advertiser_id]
        )

        # Add small random noise (±5%) to each bid
        noise = np.random.uniform(-0.05, 0.05, size=N_bids)
        bids = bids * (1 + noise)

        # Ensure non-negative bids
        return np.maximum(bids, 0)
