"""Risk Bid Simple module."""

from math import sqrt
from typing import Any

import cvxpy as cp
import numpy as np
from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class SimpleBidRisk(_Bidder):
    """Class docstring."""
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'p': 0.1,
        'q': 0.1,
        'CPC': 300.,
        'LP': False,
        'uncertainty': 0.1,
    }

    def __init__(self, params: dict = None):
        """Initialize instance."""
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.p = params.get('p', self.default_params['p'])  # gamma
        self.q = params.get('q', self.default_params['q'])  # u_0
        eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * eps)
        self.C = params.get('CPC', self.default_params['CPC'])
        self.uncertainty = params.get('uncertainty', self.default_params['uncertainty'])

    def place_bid(self, bidding_input_params, history: History) -> float:
        """place_bid."""
        if (len(history.rows) == 1) and self.LP:
            self.p, self.q = self.calculate_pq(bidding_input_params)
        p_q = max(self.p + self.q, 1e-4)
        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        CTR -= np.std(CTR) * self.uncertainty
        bid = (CTR * CVR + self.q * self.C * CTR) / p_q  # CTR * CVR -> CVR
        return max(bid, 0)

    def calculate_pq(self, bidding_input_params: dict[str, Any]):
        """Method implementation."""
        wp = np.array(bidding_input_params['wp_for_lp'])

        CTR = np.array(bidding_input_params['ctr_for_lp'])
        std_history = np.zeros_like(CTR, dtype=float)
        for i in range(1, len(CTR)):
            std_history[i] = np.std(CTR[:i])
        CTR -= self.uncertainty * std_history

        CVR = np.array(bidding_input_params['cr_for_lp'])
        budget = bidding_input_params['balance']
        n = len(wp)

        p = cp.Variable(nonneg=True)
        q = cp.Variable(nonneg=True)
        r = cp.Variable(n, nonneg=True)

        v = CTR * CVR
        constraints = [
            wp[i] * p + (wp[i] - CTR[i] * self.C) * q + r[i] >= v[i] for i in range(n)
        ]

        objective = cp.Minimize(budget * p + cp.sum(r))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, abstol=1e-10, reltol=1e-10, feastol=1e-10)

        return p.value, q.value
