# Efficiency Enhancement of Evolutionary Neural Architecture Search via Training-Free Initialization
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong
<!-- In ICONIP 2021. -->
## Setup
- Clone this repo:
```
$ git clone https://github.com/f4nku4n/ENAS_TFI
$ cd ENAS_TFI
```
- Install dependencies:
```
$ pip install -r requirements.txt
```
- Download data in [here](https://drive.google.com/drive/u/0/folders/1j9EJY8xSqjtfsJ1Tgk333hpLMF50wOpa) and put into [*data*](https://github.com/f4nku4n/ENAS_TFI/tree/master/data) folder
## Usage
### Search
- For single-objective NAS problems:
```shell
python main.py  --n_runs 21 --warm_up 0 --nSamples_for_warm_up 0 --problem_name [problem_name] --algorithm_name GA --seed 0
```
- For multi-objective NAS problems:
```shell
python main.py  --n_runs 21 --warm_up 0 --nSamples_for_warm_up 0 --problem_name [problem_name] --algorithm_name NSGA-II --seed 0
```
`--problem_name [problem_name]` receives one of following values:
problem_name               |  NAS Benchmark            |  Type of problem |  Dataset                  |  Objecitve |                
:-------------------------|:-------------------------:|:----------------:|:-------------------------:|:-------------------------:
SO-NAS101|  NAS-Bench-101 | single-objective | CIFAR-10 | validation error|
SO-NAS201-1|  NAS-Bench-201 | single-objective | CIFAR-10| validation error|
SO-NAS201-2|  NAS-Bench-201 | single-objective | CIFAR-100| validation error|
SO-NAS201-3|  NAS-Bench-201 | single-objective | ImageNet16-120| validation error|
MO-NAS101|  NAS-Bench-101 | multi-objective | CIFAR-10| #params & validation error|
MO-NAS201-1|  NAS-Bench-201 | multi-objective | CIFAR-10 | FLOPs & validation error|
MO-NAS201-2|  NAS-Bench-201 | multi-objective | CIFAR-100 | FLOPs & validation error|
MO-NAS201-3|  NAS-Bench-201 | multi-objective | ImageNet16-120 | FLOPs & validation error|

To search with the Warmup method, set `--warm_up 1` and set the number of samples `--nSamples_for_warm_up`. In our experiments, we set `--nSamples_for_warm_up 500`.

To experiment with the different `population_size` or `maximum_number_of_evaluations`, set another value in [main.py](https://github.com/f4nku4n/ENAS_TFI/blob/master/main.py) (for `population_size`) and [factory.py](https://github.com/f4nku4n/ENAS_TFI/blob/master/factory.py) (for `maximum_number_of_evaluations`)
### Evaluate & Visualize
- For single-objective NAS problems:
```shell
python visualize_so.py  --path_results [path_results]
```
- For multi-objective NAS problems:
```shell
python visualize_mo.py  --path_results [path_results]
```
For example: ```python visualize_mo.py  --path_results .\results\MO-NAS101```

***Note:*** `[path_results]` ***must only contains results of experiments are conducted on the same problem.***
<!-- ## Results (in paper)
- Single-objective NAS problems:
![](https://github.com/f4nku4n/ENAS_TFI/blob/master/figs/SONAS(1).png)

- Multi-objective NAS problems:
![](https://github.com/f4nku4n/ENAS_TFI/blob/master/figs/MONAS(1).png) -->

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [How Powerful are Performance Predictors in Neural Architecture Search?](https://github.com/automl/NASLib)
