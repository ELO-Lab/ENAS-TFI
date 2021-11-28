import numpy as np
import random
import torch

from .compare import find_the_better
from benchmark_api.nasbench import wrap_api as api
wrap_api = api.NASBench_()


def check_valid(hash_key, **kwargs):
    """
    - Check if the current solution already exists on the set of checklists.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])

def X2matrices(X):
    """
    - This function in used to convert the vector which used to representation the architecture to 2 matrix (Edges matrix & Operators matrix).
    - This function is used to help for getting hash key in '101' problems.
    """
    IDX_OPS = [1, 3, 6, 10, 15]
    edges_matrix = np.zeros((7, 7), dtype=np.int8)
    for row in range(6):
        idx_list = None
        if row == 0:
            idx_list = [2, 4, 7, 11, 16, 22]
        elif row == 1:
            idx_list = [5, 8, 12, 17, 23]
        elif row == 2:
            idx_list = [9, 13, 18, 24]
        elif row == 3:
            idx_list = [14, 19, 25]
        elif row == 4:
            idx_list = [20, 26]
        elif row == 5:
            idx_list = [27]
        for i, edge in enumerate(idx_list):
            if X[edge] - 1 == 0:
                edges_matrix[row][row + i + 1] = 1

    ops_matrix = ['input']
    for i in IDX_OPS:
        if X[i] == 2:
            ops_matrix.append('conv1x1-bn-relu')
        elif X[i] == 3:
            ops_matrix.append('conv3x3-bn-relu')
        else:
            ops_matrix.append('maxpool3x3')
    ops_matrix.append('output')

    return edges_matrix, ops_matrix


def get_hashKey(arch, problem_name):
    """
    This function is used to get the hash key of architecture. The hash key is used to avoid the existence of duplication in the population.\n
    - *Output*: The hash key of architecture.
    """
    if problem_name == 'NASBench101':
        edges_matrix, ops_matrix = X2matrices(arch)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        hash_key = wrap_api.get_module_hash(model_spec)
    elif problem_name == 'NASBench201':
        hash_key = ''.join(map(str, arch))
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')
    return hash_key

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def get_front_0(F):
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    return r == 0