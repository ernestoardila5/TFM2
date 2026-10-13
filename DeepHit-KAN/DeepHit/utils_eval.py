"""
This provides time-dependent Concordance index and Brier Score:
    - Use weighted_c_index and weighted_brier_score, which are the unbiased estimates.
    
See equations and descriptions eq. (11) and (12) of:
    - Lee et al., "Temporal Quilting for Survival Analysis", AISTATS 2019
"""

import numpy as np
from lifelines import KaplanMeierFitter


# C-index (standard, not weighted)
def c_index(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))

    for i in range(N):
        A[i, Time_survival[i] < Time_survival] = 1
        Q[i, Prediction[i] > Prediction] = 1

        if Time_survival[i] <= Time and Death[i] == 1:
            N_t[i, :] = 1

    Num = np.sum((A * N_t) * Q)
    Den = np.sum(A * N_t)

    if Num == 0 and Den == 0:
        return -1  # Not computable
    return float(Num / Den)


# Brier Score
def brier_score(Prediction, Time_survival, Death, Time):
    y_true = ((Time_survival <= Time) * Death).astype(float)
    return np.mean((Prediction - y_true) ** 2)


# Estimate censoring probability G(T)
def CensoringProb(Y, T):
    T = T.reshape(-1)
    Y = Y.reshape(-1)

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y == 0).astype(int))  # 1 = censored

    G = np.asarray(kmf.survival_function_.reset_index()).T
    G[1, G[1] == 0] = G[1, G[1] != 0][-1]  # Fill zeros with last nonzero (zero-order hold)

    return G


# Weighted C-index (Lee et al.)
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))

    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]
        W = (1.0 / G[1, tmp_idx[0]]) ** 2 if len(tmp_idx) > 0 else (1.0 / G[1, -1]) ** 2

        A[i, T_test[i] < T_test] = W
        Q[i, Prediction[i] > Prediction] = 1.0

        if T_test[i] <= Time and Y_test[i] == 1:
            N_t[i, :] = 1.0

    Num = np.sum((A * N_t) * Q)
    Den = np.sum(A * N_t)

    if Num == 0 and Den == 0:
        return -1
    return float(Num / Den)


# Weighted Brier Score (Lee et al.)
def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(N)
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        G1 = G[1, tmp_idx1[0]] if len(tmp_idx1) > 0 else G[1, -1]
        G2 = G[1, tmp_idx2[0]] if len(tmp_idx2) > 0 else G[1, -1]

        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)
