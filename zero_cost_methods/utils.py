from .predictors import ZeroCostV1, ZeroCostV2
from utils.utils import X2matrices

def get_config_for_zero_cost_predictor(problem, path_data, seed):
    config = {
        'search_space': problem.name,
        'dataset': problem.dataset,
        'root_data': path_data + '/dataset',
        'search': {
                'batch_size': 256,
                'data_size': 25000,
                'cutout': False,
                'cutout_length': 16,
                'cutout_prob': 1.0,
                'train_portion': 0.7,
                'seed': seed
        }
    }
    return config

def get_zero_cost_predictor(config, method_type):
    if method_type == 'grad_norm':
        predictor = ZeroCostV2(config, batch_size=64, method_type='grad_norm')
    elif method_type == 'grasp':
        predictor = ZeroCostV2(config, batch_size=64, method_type='grasp')
    elif method_type == 'jacov':
        predictor = ZeroCostV1(config, batch_size=64, method_type='jacov')
    elif method_type == 'snip':
        predictor = ZeroCostV2(config, batch_size=64, method_type='snip')
    elif method_type == 'fisher':
        predictor = ZeroCostV2(config, batch_size=64, method_type='fisher')
    elif method_type == 'synflow':
        predictor = ZeroCostV2(config, batch_size=64, method_type='synflow')
    else:
        raise ValueError(f'Just supported "grad_norm"; "grasp"; "jacob"; "snip"; and "synflow", not {method_type}.')
    predictor.pre_process()
    return predictor

def modify_input_for_fitting(X, problem_name):
    if problem_name == 'NASBench101':
        edges_matrix, ops_matrix = X2matrices(X)
        X_modified = {'matrix': edges_matrix, 'ops': ops_matrix}
    elif problem_name == 'NASBench201':
        OPS_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        X_modified = f'|{OPS_LIST[X[0]]}~0|+' \
                     f'|{OPS_LIST[X[1]]}~0|{OPS_LIST[X[2]]}~1|+' \
                     f'|{OPS_LIST[X[3]]}~0|{OPS_LIST[X[4]]}~1|{OPS_LIST[X[5]]}~2|'
    elif problem_name == 'NASBench301':
        pass
    return X_modified