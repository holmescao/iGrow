# iGrow: A Smart Agriculture Solution to Autonomous Greenhouse Control

iGrow is a smart agriculture solution, for autonomous greenhouse control.

<p align="center"><img src="figs/pipeline.png" width="600"/></p>

> [**iGrow: A Smart Agriculture Solution to Autonomous Greenhouse Control**](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/21440/21189&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=14603680229299879047&ei=daAsZcH0HdaSy9YPjcuG4AI&scisig=AFWwaebpDdnsmDgrjVdht1wQMU66)
>
> Xiaoyan Cao*, Yao Yao*, Lanqing Li, Wanpeng Zhang, Zhicheng An, Zhong Zhang, Li Xiao, Shihui Guo, Xiaoyu Cao, Meihong Wu, Dijun Luo (*= equal contribution)
>
> *[AAAI2022-iGrow](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/21440/21189&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=14603680229299879047&ei=daAsZcH0HdaSy9YPjcuG4AI&scisig=AFWwaebpDdnsmDgrjVdht1wQMU66)*
> 

## Abstract
Agriculture is the foundation of human civilization. However, the rapid increase of the global population poses a challenge to this cornerstone by demanding more food. Modern autonomous greenhouses, equipped with sensors and actuators, provide a promising solution to the problem by empowering precise control for high-efficient food production. However, the optimal control of autonomous greenhouses is challenging, requiring decision-making based on high-dimensional sensory data, and the scaling of production is limited by the scarcity of labor capable of handling this task. With the advances of artificial intelligence (AI), the Internet of things (IoT), and cloud computing technologies, we are hopeful to provide a solution to automate and smarten greenhouse control to address the above challenges. In this paper, we propose a smart agriculture solution named iGrow, for autonomous greenhouse control (AGC): (1) for the first time, we formulate the AGC problem as a Markov decision process (MDP) optimization problem; (2) we design a neural network-based simulator incorporated with the incremental mechanism to simulate the complete planting process of an autonomous greenhouse, which provides a testbed for the optimization of control strategies; (3) we propose a closed-loop bi-level optimization algorithm, which can dynamically re-optimize the greenhouse control strategy with newly observed data during real-world production. We not only conduct simulation experiments but also deploy iGrow in real scenarios, and experimental results demonstrate the effectiveness and superiority of iGrow in autonomous greenhouse simulation and optimal control. Particularly, compelling results from the tomato pilot project in real autonomous greenhouses show that our solution significantly increases crop yield (+10.15%) and net profit (+92.70%) with statistical significance compared to planting experts. Our solution opens up a new avenue for greenhouse production.

## Introduction
This directory contains all data and code needed to fully reproduce results for our paper. The approach is described in *iGrow: A Smart Agriculture Solution to Autonomous Greenhouse Control*.

Note that the materials presented here mainly consists of replications of experiments and results; for additional information (e.g., scenario introduction of real greenhouses) see the `Technical Appendix.pdf` and `Multimedia Appendix.zip`.


## Fully reproduce the experimental results of our paper

### 1. Comparison of baseline and incremental simulators in virtual trajectores

Running the follow command:
```python
python evaluate_simulator.py
```
You will obtain R$^2$ of different variables of two simulators.
The results are stored in `./result/table1/`


### 2. Accuracy of different simulators in the real scenario

Running the follow command:
```python
python vs_simulators.py
```
You will obtain accuracy of `WUR simulator` and `both of our simulators` compared the ground truth in the real trajectory, take planting trajectory of the champion of the 2nd Autonomous Greenhouse Challenge -- *Automatoes* as an example.

The results are stored in `./result/figure3/`


### 3. Performance comparison of different methods on our incremental simulator

Running the follow command:
```python
python vs_methods.py
```
You will obtain economic effectiveness and setpoints simulated by different methods on our incremental simulator.

The results are stored in `./result/figure4/`


### 4. Comparison of the economic effectiveness in the 2nd pilot project.

Running the follow command:
```python
python liaoyang_harvest.py
```

You will obtain the main economic effectiveness curves (including `Crop yield`, `Gains`, `Fruit Prices`) of the control group (planting experts) and the experimental group (iGrow) in the 2nd pilot project.

The results are stored in `./result/figure5/`


### 5. Overall economic effectiveness comparison of the 2nd pilot project.

Running the follow command:
```python
python liaoyang_economic.py
```

You will obtain all of economic indicators of the control group (planting experts) and the experimental group (iGrow) in the 2nd pilot project.

The results are stored in `./result/table2/`


### 6. The pair relationship among four action variables over time

Running the follow command:
```python
python liaoyang_analysis.py
```

You will obtain the pair relationship among four action variables over time of the control group (planting experts) and the experimental group (iGrow) in the 2nd pilot project.

The results are stored in `./result/figureS4toS13/`


## Re-train algorithm (not necessary)

Please set different random number seeds, otherwise the original result will be overwritten.


### 1. Simulator

Due to the size of planting trajectories dataset exceeds the available max limit set by CMT, then we upload a representative subset of this dataset.

Running the follow command:
```python
python train_simulator.py
```

You will obtain the baseline simulator.

The results are stored in `./result/models/baseline/`


### 2. SAC

Running the follow command:
```python
python sac_main.py
```

You will obtain the model of SAC algorithm.

The results are stored in `./SAC/sac_model`


### 3. EGA

Running the follow command:
```python
python ega_main.py
```

You will obtain the model of EGA algorithm.

The results are stored in `./GA/ga_train/policy/`


## High-level overview of source files
In the top-level directory are executable scripts to execute, evaluate, and visualize the experimental results of our paper. 

The relationship of these executable scripts to the results in the paper is as follows:

- `evaluate_simulator.py`: Table 1
- `vs_simulators.py`: Figure 3
- `python vs_methods.py`: Figure 4
- `python liaoyang_harvest.py`: Figure 5
- `python liaoyang_economic.py`: Table 2
- `python liaoyang_analysis.py`: Figure S4 to S13
- `python train_simulator.py`: baseline and incremental simulators
- `python sac_main.py`: The strategy of SAC algorithm
- `python ega_main.py`: The strategy of EGA algorithm

## Other information description

`Figure 1` is the overview of our paper; `Figure 2` shows the the structure of our simulator. They only need to present in the paper.

`Figure S1 to S3` and  `Table S1, S2` belong to the introduction of the pilot scenario, which can bee seen in `Multimedia Appendix.zip`.

## Environment dependencies for running

The code is compatible with `Python 3.6`. Running the follow command for install dependencies are needed to run the source code files:

```python
pip install -r requirements.txt
```

Note that `cuda==10.2`, `Pytorch==1.7.1` and `geatpy==2.5.1`.
