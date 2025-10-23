import sys
import math
sys.path.append("../")

from Utils.LP import (
    solve_dual_problem_mse, solve_dual_problem_mae,
    solve_dual_problem_ce, solve_non_robust_primal
)


def bid(CTR, CVR, wp, B, alpha, C, is_win, T, maxpq, p_old, q_old, N_resolve, N_solve):
    if len(CTR) % N_resolve == 0:
        p, q = solve_non_robust_primal(CTR, CVR, wp, B, alpha, C, is_win, T, N_solve)
    else:
        p = p_old
        q = q_old
    if p == 0 and q == 0:
        return -1, p, q
    else:
        p = max(p, maxpq)
        q = max(q, maxpq)

        bid = (1/(p+q))*CTR[-1]*CVR[-1] + (q/(p+q))*C*CTR[-1]

        return max(bid, 0), p, q


def robust_bid_mse(CTR, CVR, wp, B, alpha, C, is_win, T, maxpq, p_old, q_old, N_resolve, N_solve):
    if len(CTR) % N_resolve == 0:
        p, q = solve_dual_problem_mse(CTR, CVR, wp, B, alpha, C, is_win, T, N_solve)
    else:
        p = p_old
        q = q_old
    if p == 0 and q == 0:
        return -1, p, q
    else:
        p = max(p, maxpq)
        q = max(q, maxpq)

        bid = (1/(p+q))*CTR[-1]*CVR[-1] + (q/(p+q))*C*CTR[-1]

        correction_1 = 0
        correction_2 = 0

        N_solve = 50
        if len(CTR) > N_solve:
            CVR = CVR[-N_solve:]
            is_win = is_win[-N_solve:]
            T = sum(is_win)
        if T > 0 and bid >= wp[-1]:
            correction_1 = -q/(p+q)*alpha/math.sqrt(T)
            correction_2 = -alpha/(p+q)*CVR[-1]**2/math.sqrt(sum((CVR*is_win) ** 2))
        bid += correction_1 + correction_2

        return max(bid, 0), p, q


def robust_bid_mae(CTR, CVR, wp, B, epsilon, C, is_win, T, maxpq, gamma_old, beta_old,
                   lambda_old, chi_old, theta_old, delta_old, kappa_old, N_resolve, N_solve):
    if len(CTR) % N_resolve == 0:
        gamma, beta, lambda_, chi, theta, delta, kappa = solve_dual_problem_mae(
            CTR[max(0, len(CTR)-N_solve):],
            CVR[max(0, len(CTR)-N_solve):],
            wp[max(0, len(CTR)-N_solve):],
            B, epsilon, C, min(len(CTR), N_solve)
        )
    else:
        gamma, beta, lambda_, chi, theta, delta, kappa = (
            gamma_old, beta_old, lambda_old, chi_old, theta_old, delta_old, kappa_old
        )
    if gamma == 0 and beta == 0:
        return -1, gamma, beta, lambda_, chi, theta, delta, kappa
    else:
        gamma = max(gamma, maxpq)
        beta = max(beta, maxpq)

        bid = 1/(gamma+beta)*((lambda_*CTR[-1]+chi+delta)*CVR[-1] +
                              C*(beta*CTR[-1]+theta-kappa))

        return max(bid, 0), gamma, beta, lambda_, chi, theta, delta, kappa


def robust_bid_ce(CTR, CVR, wp, B, epsilon, C, is_win, T,
                  c, d, gamma_old, beta_old, theta_old, delta_old, N_resolve, N_solve):
    if len(CTR) % N_resolve == 0:
        gamma, beta, theta, delta = solve_dual_problem_ce(
            CVR[max(0, len(CTR)-N_solve):],
            wp[max(0, len(CTR)-N_solve):],
            B, C, d[max(0, len(CTR)-N_solve):],
            c[max(0, len(CTR)-N_solve):], epsilon, min(len(CTR), N_solve)
        )
    else:
        beta = beta_old
        gamma = gamma_old
        theta = theta_old
        delta = delta_old
    if beta == 0 and gamma == 0:
        return -1, gamma, beta, theta, delta
    else:
        bid = (1/(beta+gamma))*(delta*CVR[-1] + theta*C)

        return max(bid, 0), gamma, beta, theta, delta