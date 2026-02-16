import numpy as np
import math


def noise_CTR_mse(I, T, CTR_real, epsilon, SEED, CTRmin, CTRmax):
    CTR_noised = np.zeros((I, T))
    for i in range(I):
        np.random.seed(SEED)
        noise = np.random.normal(0, 1, T)

        noise_norm = np.linalg.norm(noise)
        normalized_noise = (noise / noise_norm) * math.sqrt(2*epsilon)

        CTR_noised[i] = CTR_real[i] + normalized_noise
        CTR_noised[i] = np.maximum(CTR_noised[i], CTRmin)
        CTR_noised[i] = np.minimum(CTR_noised[i], CTRmax)

    return CTR_noised


def noise_CTR_mae(I, T, CTR_real, epsilon, SEED, CTRmin, CTRmax):
    CTR_noised = np.zeros((I, T))
    for i in range(I):
        np.random.seed(SEED)
        noise = np.random.normal(0, 1, T)
        current_mae = np.sum(np.abs(noise))

        scaling_factor = epsilon / current_mae
        scaled_noise = noise * scaling_factor

        CTR_noised[i] = CTR_real[i] + scaled_noise
        CTR_noised[i] = np.clip(CTR_noised[i], CTRmin, CTRmax)

        actual_mae = np.sum(np.abs(CTR_noised[i] - CTR_real[i]))
        if actual_mae > epsilon:
            scaling_factor = epsilon / actual_mae
            CTR_noised[i] = CTR_real[i] + (CTR_noised[i] - CTR_real[i]) * scaling_factor

    return CTR_noised


def noise_CTR_ce(I, T, epsilon, SEED, CTRmin, CTRmax, N_iterations):
    def binary_crossentropy(ctr_real, ctr_noised):

        ctr_real = np.clip(ctr_real, 1e-7, 1-1e-7)
        ctr_noised = np.clip(ctr_noised, 1e-7, 1-1e-7)
        return -(ctr_real * np.log(ctr_noised) + (1-ctr_real) * np.log(1-ctr_noised))

    CTR_real = np.zeros((I, T))
    CTR_noised = np.zeros((I, T))

    for i in range(0, I):
        np.random.seed(SEED)

        iterations1 = 0

        while True:
            iterations1 += 1
            if iterations1 > N_iterations:
                raise Exception("Could not get acceptable CTR_real")

            CTR_real[i] = np.random.uniform(CTRmin, CTRmax, (1, T))
            initial_ce = np.sum(binary_crossentropy(CTR_real[i], CTR_real[i]))

            if initial_ce < epsilon:
                break

    for i in range(I):
        noise = np.random.normal(0, 0.5, T)
        scale = 0.01

        iterations2 = 0

        while True:
            iterations2 += 1
            if iterations2 > N_iterations:
                raise Exception("Could not get acceptable noise")

            scaled_noise = noise * scale
            CTR_noised[i] = np.clip(CTR_real[i] + scaled_noise, CTRmin, CTRmax)
            current_ce = np.sum(binary_crossentropy(CTR_real[i], CTR_noised[i]))

            if current_ce > epsilon or current_ce < 0.9*epsilon:
                scale = scale*epsilon/current_ce
            else:
                break
    return CTR_real, CTR_noised
