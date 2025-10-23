import numpy as np
import cvxpy as cp

from typing import Tuple, Dict, Any
from math import sqrt

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RobustBidCE(_Bidder):
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'gamma': 1.,
        'beta': 1.,
        'lambda_': 1.,
        'chi': 1.,
        'theta': 1.,
        'delta': 1.,
        'kappa': 1.,
        'CPC': 100.,
        'LP': False
    }

    def __init__(self, params: dict = None):
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.gamma = params.get('gamma', self.default_params['gamma'])
        self.beta = params.get('beta', self.default_params['beta'])
        self.lambda_ = params.get('lambda_', self.default_params['lambda_'])
        self.chi = params.get('chi', self.default_params['chi'])
        self.theta = params.get('theta', self.default_params['theta'])
        self.delta = params.get('delta', self.default_params['delta'])
        self.kappa = params.get('kappa', self.default_params['kappa'])
        self.C = params.get('CPC', self.default_params['CPC'])
        self.eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * self.eps)
        self.wp = []
        self.CTR = []
        self.CVR = []

    def place_bid(self, bidding_input_params, history: History) -> float:
        if len(history.rows) and self.LP:
            gamma, beta, theta, delta = self.calculate_dual_params(bidding_input_params)
            CVR = bidding_input_params['cvr_list']
        else:
            gamma, beta = self.gamma, self.beta
            theta, delta = [self.theta], [self.delta]
            CVR = [0.01]
        gamma_beta = max(gamma + beta, 1e-4)
        bid = (delta[-1] * CVR[-1] + theta[-1] * self.C) / gamma_beta
        return bid

    def solve_dual_problem(self, CVR, CTR, wp, B, C, epsilon, T):
        lambda_ = 1
        psi = cp.Variable(T)
        psi_2T_1 = cp.Variable()
        delta = cp.Variable(T)
        beta = cp.Variable()
        phi = cp.Variable(T)
        phi_2T_1 = cp.Variable()
        theta = cp.Variable(T)
        gamma = cp.Variable()
        xi_1 = cp.Variable(T)
        xi_2 = cp.Variable(T)
        c = np.log(CTR) - np.log(1-CTR)
        d = -np.log(1-CTR)

        # Check for potential numerical issues
        if np.any(CTR <= 0) or np.any(CTR >= 1):
            raise ValueError("CTR values must be strictly between 0 and 1")

        constraints = []

        for t in range(T):
            constraints.append(
                delta[t] * CVR[t] + theta[t] * C - gamma * wp[t] - beta * wp[t] +
                xi_1[t] - xi_2[t] <= 0
            )

            constraints.append(
                -lambda_ + psi[t] + delta[t] == 0
            )

            constraints.append(
                -beta + phi[t] + theta[t] == 0
            )

        constraints.append(
            psi_2T_1 + epsilon - cp.sum(d) - cp.sum(cp.multiply(c, delta)) == 0
        )

        constraints.append(
            phi_2T_1 + epsilon + cp.sum(beta * c) - cp.sum(cp.multiply(c, d)) == 0
        )

        reg_term = 2e-5

        constraints.extend([
            psi >= 0,
            psi_2T_1 >= - 1 / (reg_term * epsilon),
            delta >= 0,
            beta >= reg_term,
            phi >= reg_term / T,
            phi_2T_1 >= - 1 / (reg_term * epsilon),
            theta >= 0,
            gamma >= reg_term,
            xi_1 >= 0,
            xi_2 >= 0
        ])
        objective = cp.Minimize(gamma * B + cp.sum(xi_2))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("Problem not solved successfully")

        return gamma.value, beta.value, theta.value, delta.value

    def calculate_dual_params(self, bidding_input_params: Dict[str, Any]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        wp = np.array(bidding_input_params['wp_list'])
        CVR = np.array(bidding_input_params['cvr_list'])
        CTR = np.array(bidding_input_params['ctr_list'])
        B = bidding_input_params['balance']
        T = bidding_input_params['T']
        C = self.C
        assert len(CTR) == T

        gamma, beta, theta, delta = self.solve_dual_problem(
            CVR, CTR, wp, B, C, self.eps, T
        )

        cvr_list = bidding_input_params['cvr_list']
        cvr_list = np.array(cvr_list).flatten()

        return gamma, beta, theta, delta
