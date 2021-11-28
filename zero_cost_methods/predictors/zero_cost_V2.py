"""
Original source code: https://github.com/automl/NASLib
"""

import math
import torch
from .predictor import Predictor
from zero_cost_methods.utils_2.utils import get_train_val_loaders
from .utils.models.build_darts_net import NetworkCIFAR
from .utils.models import nasbench2 as nas201_arch
from .utils.models import nasbench1 as nas101_arch
from .utils.models import nasbench1_spec
from .utils.pruners import predictive
from zero_cost_methods.search_spaces.darts.conversions import convert_compact_to_genotype


class ZeroCostV2(Predictor):
    def __init__(self, config, batch_size=64, method_type='synflow'):
        super().__init__()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type

        self.batch_size = batch_size
        self.dataload = 'random'
        self.num_imgs_or_batches = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config

        num_classes_dict = {'CIFAR-10': 10, 'CIFAR-100': 100, 'ImageNet16-120': 120}
        self.num_classes = None
        if self.config['dataset'] in num_classes_dict:
            self.num_classes = num_classes_dict[self.config['dataset']]
        else:
            raise KeyError(f'Not support {self.config["dataset"]} dataset. Just only supported "CIFAR-10"; "CIFAR-100" '
                           f'and ImageNet16-120 datasets')
        self.train_loader = None

    def pre_process(self):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode='train')

    def query_one_arch(self, arch):
        if self.config['search_space'] == 'NASBench101':
            # Required format: 2 matrices (edges matrix; ops matrix)
            spec = nasbench1_spec._ToModelSpec(
                arch['matrix'], arch['ops']
            )
            network = nas101_arch.Network(
                spec,
                stem_out=128,
                num_stacks=3,
                num_mods=3,
                num_classes=self.num_classes
            )
        elif self.config['search_space'] == 'NASBench201':
            # Required format: |{}|+|{}|{}|+|{}|{}|{}|
            network = nas201_arch.get_model_from_arch_str(arch, self.num_classes)
            arch_2 = nas201_arch.get_arch_str_from_model(network)
            if arch != arch_2:
                print(f'Value Error: orig_arch={arch}, convert_arch={arch_2}')
                measure_score = -10e8
                return measure_score
        elif self.config['search_space'] == 'NASBench301':
            # Required format: compact_arch (genotype)
            genotype = convert_compact_to_genotype(arch)
            arch_config = {
                "name": "darts",
                "C": 32,
                "layers": 8,
                "genotype": genotype,
                "num_classes": self.num_classes,
                "auxiliary": False,
            }
            network = NetworkCIFAR(arch_config)
        else:
            raise ValueError(f'Not supporting this search space: {self.config["search_space"]}')
        network = network.to(self.device)
        score = predictive.find_measures(
            network,
            self.train_loader,
            (self.dataload, self.num_imgs_or_batches, self.num_classes),
            self.device,
            measure_names=[self.method_type],
        )
        if math.isnan(score):
            score = -1e8
        torch.cuda.empty_cache()
        return score