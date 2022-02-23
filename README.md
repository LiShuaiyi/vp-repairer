# commonroad_repairer

This repository contains a software package to solve trajectory repairing problems on [CommonRoad](https://commonroad.in.tum.de) considering traffic rules.

## The required Python dependencies
The code is written in Python 3.7 and has been tested on Ubuntu 20.04. 

You have to mannually install the following packages:
* [commonroad-qp-planner](https://gitlab.lrz.de/yuanfei/commonroad-qp-planner): branch /feature_safe_distance
* [CommonRoad Drivability Checker](https://commonroad.in.tum.de/drivability-checker)>=2021.1
* [STL CRmonitor](https://gitlab.lrz.de/ge69xek/stl_crmonitor): branch /feature_interface

## Installation Guide
We recommend using [Anaconda](https://www.anaconda.com/) to manage your environment so that even if you mess something up, you can always have a safe and clean restart. A guide for managing python environments with Anaconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

After installing Anaconda, create a new environment with:
``` sh
$ conda create -n commonroad-py37 python=3.7 -y
```

Here the name of the environment is called **commonroad-py37**. You may also change this name as you wish. In such case, don't forget to change it in the following commands as well. **Always activate** this environment before you do anything related:

```sh
$ conda activate commonroad-py37
or
$ source activate commonroad-py37
```
Install `Jupyter Notebook` and supplementary modules:
```sh
$ conda install jupyter ipykernel ipywidgets sphinx scipy -y
$ jupyter nbextension install --py widgetsnbextension --user
$ jupyter nbextension enable widgetsnbextension --user --py
```
Then, install the dependencies with:

```sh
$ pip install -r requirements.txt
```
This will install related dependencies specified in `requirements.txt`. 

## Folder structure
```
commonroad-repairer 
├─ config                               # Configurations for traffic rules and QP planner                                        
├─ crrepairer
│  ├─ abstraction
|     ├─ abstractor                     # Abstractor for metric temporal logic formulae
|     ├─ monitor                        # Wrapper for traffic rule monitor
│  ├─ cut_off
|     ├─ base                           # Base class for detecting cut-off states 
|     ├─ simulation                     # Simulation of possible compliant maneuvers 
|     ├─ tc                             # Time-To-Comply                                                  
|     ├─ ttr                            # Time-To-React (collsion avoidance) 
|     ├─ utils                          # Utility functions for detecting cut-off states 
│  ├─ repairer
|     ├─ base                           # Base class for the repairer
|     ├─ smt_repairer                   # Satisfiability modulo theories-based trajectory repairer
|     ├─ visualization                  # Script to visualize the scenario and the repaired results.
│  ├─ sat_solver
|     ├─ dpll                           # Davis-Putnam-Logemann-Loveland Algorithm
|     ├─ sat_solver                     # SATisfiability solver (DPLL algorirthm-based)
│  ├─ t_solver
|     ├─ qp_planner                     # Adaption of the QP-planner 
|     ├─ rule_constraints               # Script to add rule constraints based on the assignments of predicates 
|     ├─ t_solver                       # Theory solver
|     ├─ utils                          # Utility functions for the T-solver
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

[1] https://www.highd-dataset.com/

[2] https://commonroad.in.tum.de/dataset-converters
