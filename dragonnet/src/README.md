# DragonNet analysis under overlap violations

- The code here is used to investigate the performance of the DragonNet under overlap violations.

- As with the original implementation, you will need to install tensorflow 1.13, sklearn, numpy 1.15, keras 2.2.4 and, pandas 0.24.1

- The "overlap_main.py" script in the "experiment" folder contains the code used to run the simulations. 
It makes use of the original DragonNet implementation found in the "original_dragonnet_code" folder and the modified model in "overlap_models.py", which does not trim or scale the propensity scores.

- The IHDP samples used are readily available in the "ihdp_data" folder and the synthetic data can be generated again with the same settings using the functions in "generate_data.py".


