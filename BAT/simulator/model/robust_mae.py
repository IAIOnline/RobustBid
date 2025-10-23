import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Any

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RobustBidMAE(_Bidder):
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
        self.wp = []
        self.CTR = []
        self.CVR = []

    def place_bid(self, bidding_input_params, history: History) -> float:
        if len(history.rows) and self.LP:
            gamma, beta, lambda_, chi, theta, delta, kappa = self.calculate_dual_params(bidding_input_params)
            CTR, CVR = bidding_input_params['ctr_list'], bidding_input_params['cvr_list']
        else:
            gamma, beta, lambda_ = self.gamma, self.beta, self.lambda_
            chi, theta, delta, kappa = [self.chi], [self.theta], [self.delta], [self.kappa]
            CTR, CVR = [0.01], [0.01]
        gamma_beta = max(gamma + beta, 1e-4)
        bid = (lambda_ * CTR[-1] * CVR[-1]
               + (chi[-1] - delta[-1]) * CVR[-1]
               + self.beta * CTR[-1] * self.C
               + self.C * (theta[-1] - kappa[-1])
               ) / gamma_beta
        return bid

    def solve_dual_problem(self, CTR, CVR, wp, B, C, T, eps):
        lambda_ = cp.Variable()
        psi = cp.Variable(T)
        psi_4T_1 = cp.Variable()
        chi = cp.Variable(T)
        delta = cp.Variable(T)
        beta = cp.Variable()
        phi = cp.Variable(T)
        phi_4T_1 = cp.Variable()
        theta = cp.Variable(T)
        kappa = cp.Variable(T)
        gamma = cp.Variable()
        xi_1 = cp.Variable(T)
        xi_2 = cp.Variable(T)

        constraints = []

        constraints.append(lambda_ == 1)

        for t in range(T):
            constraints.append(
                lambda_ * CTR[t] * CVR[t] + chi[t] * CVR[t] - delta[t] * CVR[t] -
                beta * wp[t] + beta * CTR[t] * C + C * (theta[t] - kappa[t]) -
                gamma * wp[t] + xi_1[t] - xi_2[t] == 0
            )

            constraints.append(
                -lambda_ + lambda_ * CTR[t] + psi[t] + chi[t] - delta[t] == 0
            )

            constraints.append(
                -lambda_ * CTR[t] + psi[t] - chi[t] + delta[t] == 0
            )

            constraints.append(
                -beta + beta * CTR[t] + phi[t] + theta[t] - kappa[t] == 0
            )

            constraints.append(
                -beta * CTR[t] + phi[t] + kappa[t] - theta[t] == 0
            )

        constraints.append(
            -lambda_ * eps + psi_4T_1 + cp.sum(chi + delta) == 0
        )

        constraints.append(
            -beta * eps + phi_4T_1 + cp.sum(theta + kappa) == 0
        )

        reg_term = 1e-1

        constraints.extend([
            psi >= reg_term / T,
            psi_4T_1 >= -1 / (reg_term * eps),
            chi >= 0,
            delta >= 0,
            beta >= reg_term,
            phi >= reg_term / T,
            phi_4T_1 >= -1 / (reg_term * eps),
            theta >= 0,
            kappa >= 0,
            gamma >= reg_term,
            xi_1 >= 0,
            xi_2 >= 0
        ])

        objective = cp.Minimize(gamma * B + cp.sum(xi_2))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)

        return (gamma.value, beta.value, lambda_.value, chi.value,
                theta.value, delta.value, kappa.value)

    def calculate_dual_params(self, bidding_input_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray,
                                                                                   float, np.ndarray, np.ndarray,
                                                                                   float, np.ndarray]:
        wp = np.array(bidding_input_params['wp_list'])
        CTR = np.array(bidding_input_params['ctr_list'])
        CVR = np.array(bidding_input_params['cvr_list'])
        B = bidding_input_params['balance']
        T = bidding_input_params['T']  # T is the number of auctions
        C = self.C
        assert len(CTR) == T

        gamma, beta, lambda_val, chi, theta, delta, kappa = self.solve_dual_problem(
            CTR, CVR, wp, B, C, T, self.eps
        )

        cvr_list = bidding_input_params['cvr_list']
        cvr_list = np.array(cvr_list).flatten()

        return gamma, beta, lambda_val, chi, theta, delta, kappa
