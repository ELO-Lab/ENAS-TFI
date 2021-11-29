import numpy as np
from utils import get_front_0
from zero_cost_methods import modify_input_for_fitting


def get_population_X(tmp_P, pop_size, problem, zc_predictor):
    tmp_P_F = []

    if problem.type_of_problem == 'single-objective':
        for i in range(len(tmp_P)):
            X_modified = modify_input_for_fitting(tmp_P[i].X, problem.name)
            proxy_score = zc_predictor.query_one_arch(X_modified)

            tmp_P_F.append(proxy_score)
    elif problem.type_of_problem == 'multi-objective':
        for i in range(len(tmp_P)):
            X_modified = modify_input_for_fitting(tmp_P[i].X, problem.name)

            score = zc_predictor.query_one_arch(X_modified)
            complexity_metric = problem.get_complexity_metric(tmp_P[i].X)

            tmp_P_F.append([complexity_metric, -score])
    else:
        raise ValueError()

    tmp_P_F = np.array(tmp_P_F)
    P_X = get_k_best_solutions(tmp_P=tmp_P, tmp_P_F=tmp_P_F, type_of_problem=problem.type_of_problem, k=pop_size)
    return P_X

def get_k_best_solutions(type_of_problem, tmp_P, tmp_P_F, k):
    if type_of_problem == 'single-objective':
        idx = np.argsort(tmp_P_F)
        idx = np.flipud(idx)
        tmp_P_ = tmp_P[idx[:k]]
        P_X = []
        for idv in tmp_P_:
            P_X.append(idv.X)
        return np.array(P_X)

    elif type_of_problem == 'multi-objective':
        P_X = []
        n = 0
        idx = np.array(list(range(len(tmp_P_F))))
        while True:
            idx_front_0 = get_front_0(tmp_P_F)
            front_0 = idx[idx_front_0].copy()
            for i in front_0:
                P_X.append(tmp_P[i].X)
                n += 1
                if n == k:
                    return np.array(P_X)
            idx = idx[~idx_front_0]
            tmp_P_F = tmp_P_F[~idx_front_0]

    else:
        raise ValueError()