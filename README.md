# commonroad_repairer

This repository contains a software package to solve trajectory repairing problems on [CommonRoad](https://commonroad.in.tum.de) considering traffic rules.

## About Trajectory repairing

_Inspired by Randall Munroe, I describe my research using the 1,200 most common English words from [www.wordfrequency.info](www.wordfrequency.info).

We want our cars to always plan a safe path. But environments change every time. Thus, the path cannot be used as we want or does not follow traffic rules from time to time. One possible solution is to remain part of the path and plan the rest.

## The required Python dependencies

The code is written in Python 3.8 and has been tested on Ubuntu 20.04.

## Installation Guide

We recommend using [Anaconda](https://www.anaconda.com/) to manage your environment so that even if you mess something up, you can always have a safe and clean restart. A guide for managing python environments with Anaconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

After installing Anaconda, create a new environment with:

``` sh
conda create -n repairverse python=3.10 -y
```

Here the name of the environment is called **repairverse**. You may also change this name as you wish. In such case, don't forget to change it in the following commands as well. **Always activate** this environment before you do anything related:

```sh
conda activate repairverse
```

You have to manually install the following packages:

* [commonroad-qp-planner](https://gitlab.lrz.de/yuanfei/commonroad-qp-planner): branch `feature_repairing_intersection` or `feature/repair/miqp`
* [STL CRmonitor](https://gitlab.lrz.de/ge69xek/stl_crmonitor): branch `intersection_mpr` or `feature/repair/miqp`
* [MPR](https://gitlab.lrz.de/cps/commonroad-model-predictive-robustness): branch `fix_intersection_feature` or `feature/repair/miqp`

```sh
# Clone the repository and switch to the desired branch
git clone <package_url>
cd <package_folder>
git checkout <branch_name>

# Install the package in editable mode
pip install -e .
```

Then, install the dependencies with:

```sh
pip install -r requirements.txt
```

This will install related dependencies specified in `requirements.txt`. Or simply install the dependencies listed in `requirements.txt` and add this repository to your python path.

Finally, install this commonroad-repairer package:

```sh
pip install -e .
```

### Optimization license

For using the optimization solvers, e.g., Gurobi, Mosek, it is required to apply for an academic license:

* Mosek: <https://www.mosek.com/products/academic-licenses/>
* Gurobi: <https://www.gurobi.com/academia/academic-program-and-licenses/>
  * `conda install -c gurobi gurobi`
  * `connect to the campus network/use` [eduVPN](https://docs.eduvpn.org/client/linux/installation.html)
  * `grbgetkey xxx` (obtained from the Gurobi website)

## Folder structure

```sh
commonroad-repairer 
├─ config                               # Configurations for traffic rules and QP planner                                        
├─ crrepairer
│  ├─ cut_off
|     ├─ base                           # Base class for detecting cut-off states 
|     ├─ tc                             # Time-To-Comply (with traffic rules)                                              
|     ├─ ttr                            # Time-To-React (collsion avoidance) 
|     ├─ utils                          # Utility functions for detecting cut-off states 
│  ├─ repairer
|     ├─ base                           # Base class for the repairer
|     ├─ smt_repairer                   # Satisfiability modulo theories-based trajectory repairer
|     ├─ visualization                  # Script to visualize the scenario and the repaired results.
│  ├─ smt 
│     ├─ sat_solver
|        ├─ dpll                        # Davis-Putnam-Logemann-Loveland Algorithm
|        ├─ sat_solver                  # SATisfiability solver (DPLL algorirthm-based)
│     ├─ t_solver
|        ├─ qp_planner_repair           # Adaption of the QP-planner 
|        ├─ rule_constraints            # Script to add rule constraints based on the assignments of predicates 
|        ├─ t_solver                    # Theory solver
|     ├─ monitor_wrapper                # Wrapper for traffic rule monitor
├─ evaluation                           # Evaluation with HighD scenarios[1] using converter[2]
├─ scenarios
├─ tests
├─ tutorials
├─ environment.yml   
├─ gitlab-ci.yml   
├─ LICENSE.txt                                       
├─ README.md                                            
└─ setup.py                                      
```

## Minimal Example

A tutorial notebook and an example script can be found in the `tutorial/` folder. For running the examples from the paper
[3], please refer to the folder `examples/`.

* [1] <https://www.highd-dataset.com/>
* [2] <https://commonroad.in.tum.de/dataset-converters>
* [3] [Lin, Yuanfei; Althoff, Matthias: Rule-Compliant Trajectory Repairing using Satisfiability Modulo Theories. 2022 IEEE Intelligent Vehicles Symposium (IV), 2022, 449-456](https://mediatum.ub.tum.de/doc/1657306/akfnem296v88cj0gn6srj86cx.Lin_IV22_final_submission.pdf)

## Citation

```text
@inproceedings{ lin2022repair,
    author = {Lin, Yuanfei and  Althoff, Matthias},
    title = {Rule-Compliant Trajectory Repairing using Satisfiability Modulo Theories},
    booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)},
    year = {2022},
    pages = {449-456},
    doi = {10.1109/IV51971.2022.9827357},
    url = {https://ieeexplore.ieee.org/document/9827357},
    abstract = {Autonomous vehicles must comply with traffic rules. However, most motion planners do not explicitly consider all relevant traffic rules. Once traffic rule violations of an initially-planned trajectory are detected, there is often not enough time to replan the entire trajectory. To solve this problem, we propose to repair the initial trajectory by investigating the satisfiability modulo theories paradigm. This framework makes it efficient to reason whether and how the trajectory can be repaired and, at the same time, determine the part along the trajectory that can remain unchanged. Moreover, the robustness of traffic rule satisfaction is used to formulate a convex optimization problem for generating rule-compliant trajectories. We compare our approach with trajectory replanning and demonstrate its usefulness with traffic scenarios from the CommonRoad benchmark suite and recorded data. The evaluation result shows that rule-compliant trajectory repairing is computationally efficient and widely applicable. },
    keywords = {autonomous driving; traffic rules; motion planning; trajectory repairing},
}
```
