# Restricted Boltzmann Machine

This code trains an RBM via contrastive divergence to minimize the KL divergence between the empirical distribution of the MNIST dataset of handwritten digits and the marginal distribution p(v|W,a,b) of the RBM. This code also generates samples from the marginal distribution p(v|W,a,b) of the visible variables via randomly initialized Gibbs sampling chains.


## Usage
To obtain samples from marginal distribution p(v|W,a,b) of the visible variables from pretrained model via randomly initialized Gibbs sampling chains:
```
python3 run.py --epochs 0 --v_marg_steps 5000 --train False --eval True --load True --run_name run_trained --ckpt_epoch 140
```
^these samples will appear in the folder './logdir/run_trained/samples'

To train model that learns very accurate p(v|W,a,b) via very long (CD-k with k=5000) Gibbs sampling chain:
```
python3 run.py --epochs 150 --cd_k 5000 --lr .01 --v_marg_steps 5000 --batch_size 200 --n_train_eval_epochs 5
```

To quickly train model that learns decent p(v|W,a,b) via short (CD-k with k=2) Gibbs sampling chain:
```
python3 run.py --epochs 10 --cd_k 2 --adam False --vb_mean -0.5 --hb_mean -0.2 --lr .001 --v_marg_steps 5000 --batch_size 200 --n_train_eval_epochs 1
```


## Results
These are samples from trained RBM's marginal distribution p(v|W,a,b) of the visible variables obtained via randomly initialized Gibbs sampling chains of length 5000. They are located in the "./images" folder.

Samples from p(v|W,a,b) from model trained using CD-k with k=5000:

![sample_marg_prob_v__cdk5000_gibbs5000_ep140](images/sample_marg_prob_v__cdk5000_gibbs5000_ep140.png)

Samples from p(v|W,a,b) from model trained using CD-k with k=2:

![sample_marg_prob_v__cdk2_gibbs5000_ep9](images/sample_marg_prob_v__cdk2_gibbs5000_ep9.png)


## Future Improvements
ML related:
- Implement version of Adam that converges to optima that generalize better and/or mix the Gibbs Markov chain more rapidly; 
  e.g. "Fixing Weight Decay Regularization in Adam" https://arxiv.org/abs/1711.05101

Software Related:
- Cmd-line parse_args could be split into configs for specific aspects of the program.
- Probability functions (e.g. sample_bernoulli) could be fuctions of probability distribution
  classes such that each class contains unique variations of common functions such as 
  reparameterize, entropy, & log_prob.
- Since gradients of RBM are written manually instead of automatically, they should be unit tested. 
  To do this, one would assert that gradient function one writes returns results that are within a certain 
  range of finite difference approximations of the gradient.

###### Acknowledgements
- Hinton's [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
- Stackexchange reply that suggests using pseudo_log_likelihood to monitor training: https://stats.stackexchange.com/a/114935
- pseudo_log_likelihood derivation from Equation 3.6 of Pawel Budzianowski's "Training Restricted Boltzmann Machines Using High-Temperature Expansions" http://www.mlsalt.eng.cam.ac.uk/foswiki/pub/Main/ClassOf2016/Pawel_Budzianowski_8224891_assignsubmission_file_Budzianowski_Dissertation.pdf
- Hugo Larochelle's free energy derivation: http://info.usherbrooke.ca/hlarochelle/ift725/5_03_free_energy.pdf