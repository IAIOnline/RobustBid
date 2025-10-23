import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


def solve_dual_problem_mse(CTR, cvr_list, wp, B, alpha, C, is_win, T, N_solve):
    n = len(CTR)
    if n >= N_solve:
        n = N_solve
        B = B - np.sum(wp[:-N_solve] * is_win[:-N_solve])
        CTR = CTR[-N_solve:]
        cvr_list = cvr_list[-N_solve:]
        wp = wp[-N_solve:]
        is_win = is_win[-N_solve:]
        T = np.sum(is_win)

    delta = -CTR

    gamma = cp.Variable(nonneg=True)
    u_0 = cp.Variable(nonneg=True)
    # r = cp.Variable(n, nonneg=True)

    cvr_norm = np.linalg.norm(cvr_list * is_win)
    if cvr_norm > 0:
        delta += (alpha * cvr_list / cvr_norm) * is_win
        u = 1 / np.sqrt(T) * is_win
    else:
        u = np.zeros(n)

    objective = gamma * B + cp.sum(cp.maximum(0, -delta * cvr_list - wp * gamma -
                                   (wp - CTR * C) * u_0 - alpha * u_0 * u))

    problem = cp.Problem(cp.Minimize(objective))

    try:
        problem.solve(solver=cp.ECOS, abstol=1e-10, reltol=1e-10, feastol=1e-10)
        return gamma.value, u_0.value
    except cp.error.SolverError:
        return 0, 0


def solve_dual_problem_mse_alpha_ab(ctr_list, cvr_list, wp_list, B, C, is_win,
                                    epsilon_CTR, epsilon_CVR, T, N_solve):
    n = len(ctr_list)
    r_ctr = np.sqrt(2 * epsilon_CTR)
    r_cvr = np.sqrt(2 * epsilon_CVR)
    x_list = is_win.copy()

    if n >= N_solve:
        n = N_solve
        B = B - np.sum(wp_list[:-N_solve] * is_win[:-N_solve])
        ctr_list = ctr_list[-N_solve:]
        cvr_list = cvr_list[-N_solve:]
        x_list = x_list[-N_solve:]
        wp_list = wp_list[-N_solve:]
        is_win = is_win[-N_solve:]
        # T = np.sum(is_win)

    def objective(lambdas):
        lambda_a, lambda_b = lambdas

        term1 = lambda_a * r_ctr**2 + lambda_b * r_cvr**2
        term2 = np.sum(x_list * ctr_list * cvr_list)
        term3 = 0

        for t in range(len(x_list)):
            denom = 4 * lambda_a * lambda_b - x_list[t]**2
            if denom > 1e-8:
                num = 2 * x_list[t]**2 * (lambda_b * cvr_list[t]**2 +
                                          lambda_a * ctr_list[t]**2 -
                                          x_list[t] * ctr_list[t] * cvr_list[t])
                term3 += num / denom
        return -(term2 - term1 - term3)

    def lambda_constraints(lambdas):
        lambda_a, lambda_b = lambdas
        max_x_squared = np.max(x_list**2)
        return lambda_a * lambda_b - 0.25 * max_x_squared

    x0 = np.array([1.0, 1.0])

    constraints = [{'type': 'ineq', 'fun': lambda_constraints}]

    bounds = [(1e-6, None), (1e-6, None)]

    result = minimize(
        objective,
        x0,
        method='COBYLA',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 300, 'rhobeg': 0.1, 'tol': 1e-6}
    )

    if result.success:
        lambda_a, lambda_b = result.x

        max_x_squared = np.max(x_list**2)
        lambda_prod = lambda_a * lambda_b
        if lambda_prod < 0.25 * max_x_squared - 1e-6:
            print(f"   {lambda_prod:.6f} < {0.25 * max_x_squared:.6f}")

            scale_factor = np.sqrt((0.25 * max_x_squared) / lambda_prod)
            lambda_a *= scale_factor
            lambda_b *= scale_factor

        return lambda_a, lambda_b
    else:
        max_x_squared = np.max(x_list**2)
        lambda_safe = np.sqrt(0.25 * max_x_squared) * 1.01  # Safety margin of 1%
        return lambda_safe, lambda_safe


def solve_dual_problem_mse_pq(ctr_list, cvr_list, wp_list, B, C, is_win, epsilon_CTR,
                              epsilon_CVR, T, N_solve, lambda_ctr, lambda_cvr, u_list):
    n = len(ctr_list)
    a_t0 = ctr_list[-1]
    b_t0 = cvr_list[-1]
    T = int(T)

    if n >= N_solve:
        n = N_solve
        B = B - np.sum(wp_list[:-N_solve] * is_win[:-N_solve])
        ctr_list = ctr_list[-N_solve:]
        cvr_list = cvr_list[-N_solve:]
        u_list = u_list[-N_solve:]
        wp_list = wp_list[-N_solve:]
        is_win = is_win[-N_solve:]
        T = int(np.sum(is_win))

    gamma = cp.Variable(nonneg=True)
    u_0 = cp.Variable(nonneg=True)

    def objective(params, *args):
        gamma, u_0 = params
        B, T, ctr_list, cvr_list, wp_list, C, lambda_ctr, lambda_cvr, epsilon_CTR = args
        alpha = np.sqrt(2 * epsilon_CTR)

        sum_max = 0
        for t in range(T):
            f_1 = 4 * lambda_ctr * lambda_cvr - 1
            f_2 = lambda_ctr * ctr_list[t]**2 + lambda_cvr * cvr_list[t]**2
            f_3 = ctr_list[t] * cvr_list[t]
            df_dx = (f_1 * (4 * f_2 - 6 * f_3) + 4 * (f_2 - f_3)) / (f_1 ** 2)
            term = a_t0 * b_t0 - gamma * wp_list[t] + u_0 * (-wp_list[t] + C * ctr_list[t])
            term += u_list[t] * alpha - df_dx if t < len(u_list) else 0
            sum_max += max(0, term)
        return gamma * B + sum_max

    p_0 = 0.001
    q_0 = 0.001

    initial_params = [p_0, q_0]

    constraints = {'type': 'ineq', 'fun': lambda x: x[1] - np.linalg.norm(u_list)}
    bounds = [(1e-6, None), (1e-6, None)]
    args = (B, T, ctr_list, cvr_list, wp_list, C, lambda_ctr, lambda_cvr, epsilon_CTR)
    result = minimize(objective, initial_params, args=args,
                      bounds=bounds, constraints=constraints,
                      method='SLSQP')

    if result.success:
        gamma_opt, u0_opt = result.x
        return gamma_opt, u0_opt
    else:
        return gamma, u_0


def solve_non_robust_primal(CTR, cvr_list, wp, B, alpha, C, is_win, T, N_solve):
    N = len(CTR)

    if N >= N_solve:
        N = N_solve
        B = B - np.sum(wp[:-N_solve] * is_win[:-N_solve])
        CTR = CTR[-N_solve:]
        cvr_list = cvr_list[-N_solve:]
        wp = wp[-N_solve:]
        is_win = is_win[-N_solve:]
        # T = np.sum(is_win)
    delta = -CTR

    gamma = cp.Variable(nonneg=True)
    u_0 = cp.Variable(nonneg=True)
    # r = cp.Variable(N, nonneg=True)

    objective = gamma * B + cp.sum(cp.maximum(0, -delta * cvr_list - wp * gamma -
                                   (wp - CTR * C) * u_0))

    problem = cp.Problem(cp.Minimize(objective))

    try:
        problem.solve(solver=cp.ECOS, abstol=1e-10, reltol=1e-10, feastol=1e-10)
        return gamma.value, u_0.value
    except cp.error.SolverError:
        return 0, 0


def solve_dual_problem_mae(CTR, CVR, wp, B, epsilon, C, T):
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
            -lambda_*epsilon + lambda_ * CTR[t] + psi[t] + chi[t] - delta[t] == 0
        )

        constraints.append(
            -lambda_ * CTR[t] + psi[t] - chi[t] + delta[t] == 0
        )

        constraints.append(
            -beta*epsilon + beta * CTR[t] + phi[t] + theta[t] - kappa[t] == 0
        )

        constraints.append(
            -beta * CTR[t] + phi[t] + kappa[t] - theta[t] == 0
        )

    constraints.append(
        -lambda_ + psi_4T_1 + cp.sum(chi + delta) == 0
    )

    constraints.append(
        -beta + phi_4T_1 + cp.sum(theta + kappa) == 0
    )

    constraints.extend([
        psi >= 0,
        psi_4T_1 >= 0,
        chi >= 0,
        delta >= 0,
        beta >= 0,
        phi >= 0,
        phi_4T_1 >= 0,
        theta >= 0,
        kappa >= 0,
        gamma >= 0,
        xi_1 >= 0,
        xi_2 >= 0
    ])

    objective = cp.Minimize(gamma * B + cp.sum(xi_2))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, abstol=1e-10, reltol=1e-10, feastol=1e-10)

    return gamma.value, beta.value, lambda_.value, chi[-1].value, theta[-1].value, delta[-1].value, kappa[-1].value


def solve_dual_problem_ce(CVR, wp, B, C, d, c, epsilon, T):
    lambda_ = cp.Variable()
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

    constraints = []
    constraints.append(lambda_ == 1)
    for t in range(T):
        constraints.append(
            delta[t] * CVR[t] + theta[t] * C - gamma * wp[t] - beta * wp[t] +
            xi_1[t] - xi_2[t] == 0
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
        phi_2T_1 + epsilon + cp.sum(cp.multiply(beta, d)) - cp.sum(cp.multiply(c, theta)) == 0
    )

    constraints.extend([
        lambda_ >= 0,
        psi >= 0,
        psi_2T_1 >= 0,
        delta >= 0,
        beta >= 0,
        phi >= 0,
        phi_2T_1 >= 0,
        theta >= 0,
        gamma >= 0,
        xi_1 >= 0,
        xi_2 >= 0
    ])

    objective = cp.Minimize(gamma * B + cp.sum(xi_2))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, abstol=1e+1, reltol=1e+1, feastol=1e+1)

    return gamma.value, beta.value, theta[-1].value, delta[-1].value
