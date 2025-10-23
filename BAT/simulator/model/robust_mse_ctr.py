"""Robust Mse Ctr module."""

from math import sqrt
from typing import Any

import cvxpy as cp
import numpy as np
from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RobustBidMSE(_Bidder):
    """Class docstring."""
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'gamma': 1.,
        'u_0': 1.,
        'CPC': 100.,
        'LP': False
    }

    def __init__(self, params: dict = None):
        """Initialize instance."""
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.gamma = params.get('gamma', self.default_params['gamma'])
        self.u_0 = params.get('u_0', self.default_params['u_0'])
        self.C = params.get('CPC', self.default_params['CPC'])
        eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * eps)

    def place_bid(self, bidding_input_params, history: History) -> float:
        """place_bid."""
        if (len(history.rows) == 1) and self.LP:
            self.gamma, self.u_0 = self.calculate_dual_params(bidding_input_params)
        gamma_u = max(self.gamma + self.u_0, 1e-4)
        n_transactions = bidding_input_params['T']
        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        bid = (CTR * CVR + self.u_0 * self.C * CTR) / gamma_u
        cvr_list = bidding_input_params['cvr_list']
        if bidding_input_params['winning'] and (len(cvr_list) == 0):
            bid += -self.alpha / gamma_u * (self.u_0 / sqrt(n_transactions))
        elif bidding_input_params['winning']:
            cvr_norm = np.linalg.norm(cvr_list)
            bid += -self.alpha / gamma_u * (CVR ** 2 / cvr_norm)
        return max(bid, 0)

    def solve_robust_primal(self, CTR, CVR, wp, budget, alpha, C, is_win, cvr_list, _n_transactions):
        """Method implementation."""
        n = len(CTR)

        delta = - CTR
        gamma = cp.Variable(nonneg=True)
        u_0 = cp.Variable(nonneg=True)
        if is_win:
            cvr_norm = np.linalg.norm(cvr_list)
            if cvr_norm > 0:
                delta += alpha * CVR / cvr_norm
            else:
                delta += self.alpha * CVR
            u = 1 / np.sqrt(n) * np.ones(n)
        else:
            u = np.zeros(n)
        v = - cp.multiply(delta, CVR)
        expression = v - wp * gamma - alpha * u - u_0 * wp + C * u_0 * CTR
        objective = (gamma * budget + cp.sum(cp.maximum(0, expression)))

        problem = cp.Problem(cp.Minimize(objective))
        try:
            problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=1000, verbose=False)
        except cp.error.SolverError:
            try:
                problem.solve(solver=cp.SCS, eps=1e-6, max_iters=5000, verbose=False)
            except cp.error.SolverError:
                return self.gamma, self.u_0

        gamma_opt = gamma.value
        u_0_opt = u_0.value
        return gamma_opt, u_0_opt

    def calculate_dual_params(self, bidding_input_params: dict[str, Any]) -> tuple[float, float]:
        """calculate_dual_params."""
        is_win = bidding_input_params['winning']
        n_transactions = bidding_input_params['T']
        wp = np.array(bidding_input_params['wp_for_lp'])
        CTR = np.array(bidding_input_params['ctr_for_lp'])
        CVR = np.array(bidding_input_params['cr_for_lp'])
        budget = bidding_input_params['balance']
        cvr_list = bidding_input_params['cvr_list']
        cvr_list = np.array(cvr_list).flatten()
        n_transactions = bidding_input_params['T']  # T is the number of wins
        gamma, u_0 = self.solve_robust_primal(
            CTR, CVR, wp, budget, self.alpha, self.C, is_win, cvr_list, n_transactions
        )
        return gamma, u_0
