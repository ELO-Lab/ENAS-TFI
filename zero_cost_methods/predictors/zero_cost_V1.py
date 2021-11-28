"""
Author: Robin Ru @ University of Oxford
This contains implementations of jacov and snip based on
https://github.com/BayesWatch/nas-without-training (jacov)
and https://github.com/gahaalt/SNIP-pruning (snip)
Note that zerocost_v2.py contains variants of jacov and snip implemented
in subsequent work. However, we find this version of jacov tends to perform
better.
"""


import numpy as np
import torch
import logging
import gc

from zero_cost_methods.predictors.predictor import Predictor
from zero_cost_methods.predictors.utils.build_nets import get_cell_based_tiny_net
from zero_cost_methods.utils_2.utils import get_project_root, get_train_val_loaders
# from zero_cost_methods.predictors.utils_2.build_nets.build_darts_net import NetworkCIFAR
# from zero_cost_methods.search_spaces.darts.conversions import convert_compact_to_genotype

logger = logging.getLogger(__name__)


def get_batch_jacobian(net, x):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob


def eval_score(jacob):
    correlation = np.corrcoef(jacob)
    v, _ = np.linalg.eig(correlation)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


class ZeroCostV1(Predictor):
    def __init__(self, config, batch_size=64, method_type='jacov'):
        super().__init__()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.batch_size = batch_size
        self.method_type = method_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.search_space = config['search_space']
        self.dataset = config['dataset']
        if method_type == 'jacov':
            self.num_classes = 1
        else:
            num_classes_dic = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}
            self.num_classes = num_classes_dic[self.config['dataset']]
        self.train_loader = None

    def pre_process(self):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode='train')

    def query(self, xtest):
        test_set_scores = []
        count = 0
        for test_arch in xtest:
            count += 1
            logger.info("zero cost: {} of {}".format(count, len(xtest)))
            if "nasbench201" in self.search_space:
                ops_to_nb201 = {
                    "AvgPool1x1": "avg_pool_3x3",
                    "ReLUConvBN1x1": "nor_conv_1x1",
                    "ReLUConvBN3x3": "nor_conv_3x3",
                    "Identity": "skip_connect",
                    "Zero": "none",
                }
                # convert the naslib representation to nasbench201
                cell = test_arch.edges[2, 3].op
                edge_op_dict = {
                    (i, j): ops_to_nb201[cell.edges[i, j]["op"].get_op_name]
                    for i, j in cell.edges
                }
                op_edge_list = [
                    "{}~{}".format(edge_op_dict[(i, j)], i - 1)
                    for i, j in sorted(edge_op_dict, key=lambda x: x[1])
                ]
                arch_str = "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
                print(arch_str)
                arch_config = {
                    "name": "infer.tiny",
                    "C": 16,
                    "N": 5,
                    "arch_str": arch_str,
                    "num_classes": self.num_classes,
                }

                network = get_cell_based_tiny_net(
                    arch_config
                )  # create the network from configuration
            # elif "darts" in self.config.search_space:
            #     test_genotype = convert_compact_to_genotype(test_arch.compact)
            #     arch_config = {
            #         "name": "darts",
            #         "C": 32,
            #         "layers": 8,
            #         "genotype": test_genotype,
            #         "num_classes": self.num_classes,
            #         "auxiliary": False,
            #     }
            #     network = NetworkCIFAR(arch_config)

            data_iterator = iter(self.train_loader)
            x, target = next(data_iterator)
            x, target = x.to(self.device), target.to(self.device)

            network = network.to(self.device)

            if self.method_type == "jacov":
                jacobs, labels = get_batch_jacobian(network, x, target)
                # print('done get jacobs')
                jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

                try:
                    score = eval_score(jacobs, labels)
                except Exception as e:
                    print(e)
                    score = -10e8

            elif self.method_type == "snip":
                criterion = torch.nn.CrossEntropyLoss()
                network.zero_grad()
                _, y = network(x)
                loss = criterion(y, target)
                loss.backward()
                grads = [
                    p.grad.detach().clone().abs()
                    for p in network.parameters()
                    if p.grad is not None
                ]

                with torch.no_grad():
                    saliences = [
                        (grad * weight).view(-1).abs()
                        for weight, grad in zip(network.parameters(), grads)
                    ]
                    score = torch.sum(torch.cat(saliences)).cpu().numpy()
                    if hasattr(self, "ss_type") and self.ss_type == "darts":
                        score = -score

            test_set_scores.append(score)
            network, data_iterator, x, target, jacobs, labels = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
            torch.cuda.empty_cache()
            gc.collect()

        return np.array(test_set_scores)

    def query_one_arch(self, arch_str):
        # TODO: Make this function suitable for NAS-Bench-101 and NAS-Bench-301
        arch_config = {
            "name": "infer.tiny",
            "C": 16,
            "N": 5,
            "arch_str": arch_str,
            "num_classes": self.num_classes,
        }
        network = get_cell_based_tiny_net(arch_config)  # create the network from configuration

        data_iterator = iter(self.train_loader)
        x, target = next(data_iterator)
        x, target = x.to(self.device), target.to(self.device)

        network = network.to(self.device)

        if self.method_type == "jacov":
            jacobs = get_batch_jacobian(network, x)
            # print('done get jacobs')
            jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
            try:
                score = eval_score(jacobs)
            except Exception as e:
                print(e)
                score = -10e8

        elif self.method_type == "snip":
            criterion = torch.nn.CrossEntropyLoss()
            network.zero_grad()
            _, y = network(x)
            loss = criterion(y, target)
            loss.backward()
            grads = [
                p.grad.detach().clone().abs()
                for p in network.parameters()
                if p.grad is not None
            ]

            with torch.no_grad():
                saliences = [
                    (grad * weight).view(-1).abs()
                    for weight, grad in zip(network.parameters(), grads)
                ]
                score = torch.sum(torch.cat(saliences)).cpu().numpy()
                if hasattr(self, "ss_type") and self.ss_type == "darts":
                    score = -score

        torch.cuda.empty_cache()
        gc.collect()

        return score

# if __name__ == '__main__':
#     # TODO:
#     arch_int = [0, 2, 3, 4, 0, 2]
#     arch_str = int2str(arch_int)
#     # print(arch_str)
#     arch_config = {
#         'name': 'infer.tiny',
#         'C': 16,
#         'N': 5,
#         'arch_str': arch_str,
#         'num_classes': 1,
#     }
#     """
#         - Input: the 'arch_str' representation of architecture
#     """
#
#     network = get_cell_based_tiny_net(arch_config)
#     print(network)