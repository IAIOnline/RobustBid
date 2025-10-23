from .utils import price2bin, bin2price
from .modules import SimulationResult, Campaign, History
from ..model.bidder import _Bidder
import pandas as pd
import numpy as np
from typing import Tuple


CVR_column = 'CRPredicts'  # 'CRPredicts_noised'
CTR_column = 'CTRPredicts_noised'


def simulate_step(
    stats_pdf: pd.DataFrame,
    campaign: Campaign,
    bid: float,
) -> SimulationResult:
    """
    Simulate one step of the auction for a 1-hour period.

    Args:
        stats_pdf: Statistics dataframe
        campaign: Current campaign object
        bid: Bid value
    Returns:
        SimulationResult: Results of the auction step
    """

    # Convert bid to bin or set to minimum if bid is zero or negative
    bid_price_bin = price2bin(bid) if bid > 0 else -1000

    # Filter stats for the current time window and campaign
    stats_window = stats_pdf[
        (stats_pdf['period'] >= campaign.curr_time) &
        (stats_pdf['period'] < campaign.curr_time + 3600) &
        (stats_pdf['campaign_id'] == campaign.campaign_id)
    ].copy()

    # Aggregate data for bids less than or equal to the current bid
    agg_data = stats_window[
        stats_window['contact_price_bin'] <= bid_price_bin
    ][
        [
            'AuctionWinBidSurplus',
            'AuctionVisibilitySurplus',
            'AuctionClicksSurplus',
            'AuctionContactsSurplus'
        ]
    ].sum()
    return SimulationResult(
        spent=bid,
        visibility=agg_data['AuctionVisibilitySurplus'],
        clicks=agg_data['AuctionClicksSurplus'],
        contacts=agg_data['AuctionContactsSurplus'],
        bid=bid,
    )


def simulate_campaign(
    campaign: Campaign,
    bidder: _Bidder,
    stats_file: pd.DataFrame,
    start_time: int = None,
    loss_type: str = 'MSE',
    THRESHOLD: float = 0.1
) -> History:
    """
    Simulate a campaign using historical data.

    Args:
        campaign: Campaign object
        bidder: Bidder object
        stats_file: Historical statistics
        start_time: Start time for simulation. Defaults to None.
    Returns:
        History: Simulation history of spending and clicks
    """
    if start_time:
        campaign.curr_time = start_time // 3600 * 3600
    else:
        campaign.curr_time = campaign.campaign_start // 3600 * 3600

    simulation_history = History()

    bidder_spend = 0
    bidder_clicks = 0
    # For M-PID only, not used for cold start, so could be set any
    campaign_ctr, campaign_cr = 0.0, 0.0
    wp_for_lp: float = None
    ctr_for_lp: float = None
    cr_for_lp: float = None
    cur_cpc: float = None
    # for new metrics (TVC, cpc_avg, cpc+)
    win_mask: list = []  # x_t
    # tvc_list: list = []
    CPC: float = bidder.C
    cpc_list: list = []
    bid_list: list = []
    ctr_list: list = []
    cr_list: list = []  # list of all the CVRs (CTR*CVR)
    cvr_list: list = []  # list of the CVRs if win for mse (CVR)
    wp_list: list = []

    while campaign.curr_time < campaign.campaign_end:
        # Request bid from bidder
        bid = bidder.place_bid(
            history=simulation_history,
            # TODO: delete excessive parameters
            bidding_input_params={
                    'item_id': campaign.item_id,
                    'loc_id': campaign.loc_id,
                    'region_id': campaign.region_id,
                    'logical_category': campaign.logical_category,
                    'microcat_ext': campaign.microcat_ext,
                    'balance': campaign.balance,
                    'initial_balance': campaign.initial_balance,
                    'clicks': campaign.clicks,
                    'campaign_id': campaign.campaign_id,
                    'campaign_start_time': campaign.campaign_start,
                    'campaign_end_time': campaign.campaign_end,
                    'curr_time': campaign.curr_time,
                    'prev_balance': campaign.prev_balance,
                    'prev_bid': campaign.prev_bid,
                    'prev_clicks': campaign.prev_clicks,
                    'prev_contacts': campaign.prev_contacts,
                    'prev_time': campaign.prev_time,
                    'desired_clicks': campaign.desired_clicks,
                    'desired_time': campaign.desired_time,
                    'prev_ctr': campaign_ctr,
                    'prev_cr': campaign_cr,  # campaign_cr / campaign_ctr if campaign_ctr else 0,
                    'ctr_for_lp': ctr_for_lp,
                    'cr_for_lp': cr_for_lp,
                    'wp_for_lp': wp_for_lp,
                    'winning': campaign.winning,
                    'T': campaign.T,
                    'cvr_list': cvr_list,  # list of the CVRs of the winning auctions
                    'ctr_list': ctr_list,
                    'cr_list': cr_list,  # list of all the CVRs
                    'wp_list': wp_list,
                    'win_mask': win_mask,
                }
        )

        # Simulate auction results for a 1-hour window
        simulation_result = simulate_step(
            stats_pdf=stats_file,  # simulate with the real data
            campaign=campaign,
            bid=bid,
        )
        bidder_spend = simulation_result.spent
        bidder_clicks = simulation_result.clicks
        if campaign_ctr and bidder_clicks:  # скипаем стартовый шаг
            cur_cpc = (bidder_spend / bidder_clicks) / campaign_ctr
            cpc_list.append(cur_cpc)

        # Adjust results if spend exceeds budget
        coef = 1.0
        if simulation_result.spent > campaign.balance:
            coef = campaign.balance / simulation_result.spent

        # Update campaign status
        campaign.prev_balance = campaign.balance
        campaign.prev_clicks = campaign.clicks
        campaign.prev_time = campaign.curr_time
        campaign.prev_bid = bid
        campaign.balance -= simulation_result.spent * coef
        campaign.clicks += simulation_result.clicks * coef
        campaign.contacts += simulation_result.contacts * coef
        campaign.curr_time += 3600

        # Find the nearest stats window with logs
        stats_window = pd.DataFrame()
        i = 0
        while stats_window.empty:
            stats_window = (
                stats_file
                [
                    (stats_file['period'] >= campaign.curr_time - 3600 * (i - 1)) &
                    (stats_file['period'] < campaign.curr_time - 3600 * (i - 2)) &
                    (stats_file['campaign_id'] == campaign.campaign_id)
                ]
                .copy()
            )
            i += 1

        # Collect CTR, CVR with campaign's history
        if ctr_for_lp is None:
            ctr_for_lp, cr_for_lp, wp_for_lp = ctr_cvr_count_for_lp(stats_window)

        campaign_ctr, campaign_cr, wp = ctr_cvr_count(stats_window, bid)
        ctr_list.append(campaign_ctr)
        cr_list.append(campaign_cr)
        wp_list.append(wp)
        bid_list.append(bid)
        if loss_type == 'mse':
            if bidder_clicks > THRESHOLD:
                # we appended just cr value because in our data cr contains ctr prob
                # cvr_list.append(campaign_cr / campaign_ctr)
                cvr_list.append(campaign_cr)
                win_mask.append(1)
                campaign.winning = True
                campaign.T += 1
            else:
                win_mask.append(0)
                campaign.winning = False
        else:
            if campaign_ctr:
                # cvr_list.append(campaign_cr / campaign_ctr)
                cvr_list.append(campaign_cr)
            else:
                # if click probabilty is 0 than conversion probability is also 0
                cvr_list.append(0)
            win_mask.append(1)
            campaign.winning = True
            campaign.T += 1
        tvc, cpc_percent, cpc_avg_up, cpc_avg_down = metrics_count(np.array(win_mask),
                                                                   np.array(bid_list),
                                                                   np.array(ctr_list),
                                                                   np.array(cr_list),
                                                                   CPC)

        # Add record to simulation history
        simulation_history.add(
            campaign=campaign,
            bid=bid,
            spend=bidder_spend,
            clicks=bidder_clicks,
            tvc=tvc,
            cpc_percent=cpc_percent,
            cpc_avg_up=cpc_avg_up,
            cpc_avg_down=cpc_avg_down,
        )
        if campaign.balance < 0.00001:
            break
    return simulation_history


def metrics_count(win_mask: list,
                  bid_list: list,
                  ctr_list: list,
                  cr_list: list,
                  CPC: float) -> Tuple[float, float, float]:
    n = len(win_mask)
    tvc = sum(cr_list * win_mask) if n else 0

    # cpc_mask = bid_list / ctr_list < CPC
    # cpc_percent = sum(win_mask * ctr_list * cpc_mask) / sum(win_mask * ctr_list) if sum(win_mask) else 0
    cpc_percent = 0

    cpc_avg_up = sum(win_mask * bid_list)
    cpc_avg_down = sum(win_mask * ctr_list)

    return tvc, cpc_percent, cpc_avg_up, cpc_avg_down


def ctr_cvr_count(stats_window: pd.DataFrame, bid: float) -> Tuple[float, float]:
    """
    Calculate CTR and CVR based on the closest bin to the given bid.
    """
    filtered_bins = stats_window[stats_window.contact_price_bin > price2bin(bid)]['contact_price_bin']
    if not filtered_bins.empty:
        closest_bin = filtered_bins.min()
    else:
        closest_bin = stats_window['contact_price_bin'].max()
    campaign_ctr = stats_window[stats_window.contact_price_bin == closest_bin][CTR_column].max()
    campaign_cr = stats_window[stats_window.contact_price_bin == closest_bin][CVR_column].max()
    return campaign_ctr, campaign_cr, closest_bin


def ctr_cvr_count_for_lp(stats_window: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extract CTR, CVR, and winning prices for LP problem in M-PID.
    """
    campaign_ctr = stats_window[CTR_column]  # CTRPredicts_noised
    campaign_cr = stats_window[CVR_column]
    wp_for_lp = bin2price(stats_window['contact_price_bin'])
    return campaign_ctr, campaign_cr, wp_for_lp
