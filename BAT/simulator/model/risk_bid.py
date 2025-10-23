"""https://arxiv.org/pdf/1701.02433"""

from math import sqrt

import numpy as np
from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RiskBid(_Bidder):
    """Risk-aware bidding strategy based on https://arxiv.org/pdf/1701.02433."""

    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'CPC': 300.,
        # corresponds to \alpha in the original paper, 0 in case of non-risky method
        'uncertainty': 100.,
        # corresponds to v in the original paper, value of impressions
        'value': 0.42,  # optimal value through optuna trials with non-risky method
    }

    def __init__(self, params: dict = None):
        """Initialize risk-aware bidder with uncertainty parameters."""
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        # self.LP = params.get('LP', self.default_params['LP'])
        # self.p = params.get('p', self.default_params['p'])  # gamma
        # self.q = params.get('q', self.default_params['q'])  # u_0
        eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * eps)
        self.uncertainty = params.get('uncertainty', self.default_params['uncertainty'])
        self.value = params.get('value', self.default_params['value'])
        self.C = params.get('CPC', self.default_params['CPC'])

    def place_bid(self, bidding_input_params: dict, _history: History) -> float:
        """Calculate bid adjusting CTR by uncertainty-weighted standard deviation."""
        ctrs = np.array(bidding_input_params['ctr_list'])
        # cvrs = np.array(bidding_input_params['cvr_list'])
        if len(ctrs) > 1:
            CTR_std = np.std(ctrs)
        else:
            CTR_std = 0
        if len(ctrs):
            CTR = ctrs[-1]
        else:
            return 0
        # CVR = bidding_input_params['prev_cr']
        bid = self.value * (CTR - self.uncertainty * CTR_std)
        return max(bid, 0)
