# Implementation for Bachelor project
## Authors
For networks.py, functions.py, DataGen.ipynb, implementation of baseline -> Joery de Vries (supervisor)
For the rest -> Maria Mihai
## Dependencies
Requires specific flax version (0.8.1)
## How to run
python train_variants.py train_sn --seed 42 --path '.\logs' --setup 2  
train_sn = Variant to run (choose out of train_baseline, train_sn, train_ab, train_snab).
--seed = Randomisation seed
--path = Where to save logs
--setup = Training data set-up (choose out of 1 = noisy data and 2 = different function families).
