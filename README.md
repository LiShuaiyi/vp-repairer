# commonroad_repairer

This repository contains a software package to solve trajectory repairing problems on [CommonRoad](https://commonroad.in.tum.de) considering traffic rules.

## The required Python dependencies
The code is written in Python 3.7 and has been tested on Ubuntu 20.04. 
* matplotlib>=3.3.4
* numpy>=1.19.5
* sympy>=1.8
* setuptools>=50.3.2
* z3-solver>=4.8.12.0
* cvxpy>=1.1.15
* osqp>=0.6.2
* mosek>=9.3.11
* commonroad-io>=2021.1
* commonroad-vehicle-models>=1.0.0
* [commonroad-qp-planner/feature_safe_distance](https://gitlab.lrz.de/yuanfei/commonroad-qp-planner)
* [CommonRoad Drivability Checker>=2021.1/](https://commonroad.in.tum.de/drivability-checker)
* [STL CRmonitor/feature_interface](https://gitlab.lrz.de/ge69xek/stl_crmonitor)

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
├─ evaluate
├─ external
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
