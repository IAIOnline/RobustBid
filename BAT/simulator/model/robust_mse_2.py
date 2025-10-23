import numpy as np
from math import sqrt
from scipy.optimize import minimize

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RobustBidMSE_CTRCVR(_Bidder):
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps_ctr': 0.01,
        'eps_cvr': 0.01,
        'gamma': 0.5,
        'u_0': 0.5,
        'CPC': 100.,
        'LP': False
    }

    def __init__(self, params: dict = None):
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.gamma = params.get('gamma', self.default_params['gamma'])
        self.u_0 = params.get('u_0', self.default_params['u_0'])
        self.C = params.get('CPC', self.default_params['CPC'])
        eps_ctr = params.get('eps_ctr', self.default_params['eps_ctr'])
        eps_cvr = params.get('eps_cvr', self.default_params['eps_cvr'])
        self.u_list = [self.u_0]
        self.lambda_ctr, self.lamda_cvr = 0.26, 0.26
        self.r_ctr = sqrt(2 * eps_ctr)
        self.r_cvr = sqrt(2 * eps_cvr)

        self.alpha = sqrt(2 * eps_ctr)

    def place_bid(self, bidding_input_params, history: History) -> float:
        T = min(2, bidding_input_params['T'])
        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        ctr_list = bidding_input_params['ctr_list']
        cvr_list = bidding_input_params['cr_list']
        wp_list = bidding_input_params['wp_list']
        x_list = bidding_input_params['win_mask']
        if len(history.rows):
            lambda_ctr, lambda_cvr = self.solve_for_lambda(ctr_list, cvr_list,
                                                           wp_list, x_list, bidding_input_params['balance'],
                                                           self.C, T, self.r_ctr, self.r_cvr,
                                                           self.lambda_ctr, self.lamda_cvr)
            self.lambda_ctr, self.lamda_cvr = lambda_ctr, lambda_cvr
            B = bidding_input_params['balance']
            self.gamma, self.u_0 = self.calculate_dual_params(ctr_list, cvr_list, wp_list,
                                                              CTR, CVR, B, T,
                                                              lambda_ctr, lambda_cvr)
            self.u_list.append(self.u_0)
        gamma_u = max(self.gamma + self.u_0, 1e-4)
        bid = (CTR * CVR + self.u_0 * self.C * CTR) / gamma_u
        if bidding_input_params['winning']:
            bid += -self.alpha / gamma_u * (self.u_0 / sqrt(T))
            if (lambda_ctr * lambda_cvr - 0.25) <= 1e-8:
                lambda_cvr *= 1.01
                lambda_ctr *= 1.01
            f_1 = 4 * lambda_ctr * lambda_cvr - 1
            f_2 = lambda_ctr * CTR ** 2 + lambda_cvr * CVR ** 2
            f_3 = CTR * CVR
            df_dx = (f_1 * (4 * f_2 - 6 * f_3) + 4 * (f_2 - f_3)) / f_1 ** 2
            bid += - self.alpha / gamma_u * df_dx
        return max(bid, 1e-5)

    def calculate_dual_params(self, ctr_list, cvr_list, wp_list, a_t0, b_t0, B, T, lambda_ctr, lambda_cvr):
        def objective(params, *args):
            gamma, u_0 = params
            B, T, ctr_list, cvr_list, wp_list, alpha, C, lambda_ctr, lambda_cvr = args
            sum_max = 0
            for t in range(T):
                f_1 = 4 * lambda_ctr * lambda_cvr - 1
                if abs(f_1) < 1e-10:
                    df_dx = 0
                else:
                    f_2 = lambda_ctr * ctr_list[t]**2 + lambda_cvr * cvr_list[t]**2
                    f_3 = ctr_list[t] * cvr_list[t]
                    df_dx = (f_1 * (4 * f_2 - 6 * f_3) + 4 * (f_2 - f_3)) / (f_1 ** 2)

                u_term = self.u_list[t] * alpha if t < len(self.u_list) else 0
                term = a_t0 * b_t0 - gamma * wp_list[t] + u_0 * (-wp_list[t] + C * ctr_list[t]) + u_term - df_dx
                sum_max += max(0, term)
            return gamma * B + sum_max

        alpha = self.alpha
        C = self.C
        initial_params = [self.gamma, self.u_0]
        u_norm = np.linalg.norm(self.u_list)

        constraints = {'type': 'ineq', 'fun': lambda x: x[1] - u_norm}
        bounds = [(1e-6, None), (1e-6, None)]
        args = (B, T, ctr_list[:T], cvr_list[:T], wp_list[:T], alpha, C, lambda_ctr, lambda_cvr)

        result = minimize(objective, initial_params, args=args,
                          bounds=bounds, constraints=constraints,
                          method='COBYLA',
                          options={'maxiter': 1000, 'rhobeg': 0.2, 'tol': 1e-4})

        if result.success:
            return result.x

        best_result = result
        best_fun = result.fun

        for g_start in [0.1, 1.0, 5.0]:
            for u_start in [u_norm*1.5, u_norm*3]:
                alt_result = minimize(objective, [g_start, u_start], args=args,
                                      bounds=bounds, constraints=constraints,
                                      method='COBYLA',
                                      options={'maxiter': 500, 'tol': 1e-5})

                if alt_result.fun < best_fun:
                    best_fun = alt_result.fun
                    best_result = alt_result

        if best_result.success:
            return best_result.x
        else:
            return self.gamma, self.u_0

    @staticmethod
    def solve_for_lambda(ctr_list, cvr_list, wp_list, x_list, B, C, T, r_ctr, r_cvr, last_lambda_a, last_lambda_b):
        ctr_list = np.array(ctr_list)[:T]
        cvr_list = np.array(cvr_list)[:T]
        wp_list = np.array(wp_list)[:T]
        x_list = np.array(x_list)[:T]

        def objective(lambdas):
            lambda_a, lambda_b = lambdas

            term1 = lambda_a * r_ctr**2 + lambda_b * r_cvr**2
            term2 = np.sum(x_list * ctr_list * cvr_list)
            term3 = 0

            for t in range(len(x_list)):
                denom = 4 * lambda_a * lambda_b - x_list[t]**2
                if denom > 1e-8:
                    num = 2 * x_list[t]**2 * (lambda_b * cvr_list[t]**2 + lambda_a * ctr_list[t]**2 -
                                              x_list[t] * ctr_list[t] * cvr_list[t])
                    term3 += num / denom
            return -(term2 - term1 - term3)

        def lambda_constraints(lambdas):
            lambda_a, lambda_b = lambdas
            max_x_squared = np.max(x_list**2)
            return lambda_a * lambda_b - 0.25 * max_x_squared

        x0 = np.array([last_lambda_a, last_lambda_b])

        constraints = [{'type': 'ineq', 'fun': lambda_constraints}]

        bounds = [(1e-6, None), (1e-6, None)]

        result = minimize(
            objective,
            x0,
            method='COBYLA',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000, 'rhobeg': 0.5, 'tol': 1e-5}
        )

        if result.success:
            lambda_a, lambda_b = result.x

            max_x_squared = np.max(x_list**2)
            lambda_prod = lambda_a * lambda_b
            if lambda_prod < 0.25 * max_x_squared - 1e-6:
                scale_factor = np.sqrt(0.25 / lambda_prod) * 1.05
                lambda_a *= scale_factor
                lambda_b *= scale_factor
            return lambda_a, lambda_b
        else:
            max_x_squared = np.max(x_list**2)
            lambda_safe = np.sqrt(0.25 * max_x_squared) * 1.05
            return lambda_safe, lambda_safe
