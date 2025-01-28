# OAT_Trust

Here's a super quick overview of the repo's functionality.

#### Table of Contents
* [Graphs](#graphs)
* [Configs](#configs)
* [Running Experiments](#configs)


## Graphs

Every graph is grabbed by ```get_graph(key)```, where the key is a string containing graph creation info. Each graph has a separate mapping from keys to graphs, but the following should always be true (especially when making new graph types):

* The key should begin with '[type]_', where the type determines which graph maker is used. E.g.
    * E.g. 'line_20', 'EC_5_.5', 'Rbell_3_15_2'
* All other parameters are separated by underscores, which are parsed in ```get_graph```
* Each has a corresponding ```make_[type](params)``` function which returns the adjacency matrix, with ones added to the diagonal.
* TODO: Adding functionality for specifying malicious agent numbers. Should there be defaults? Should this vary from graph to graph? etc.

## ```calculate_optimal_attack.py```

This file runs the experiment for some config and (optionally) an additional parameter ```H```.

### Configs

Each config should (preferably) be in ./configs or ./finished_configs, though we could add better structure. Configs are .yaml files with the following structure:

```
graph: [string corresponding to graph] e.g. 'Rbell_5_30_2'
x_inits: [initial values for x_L(0); in 'zeros', 'alternating', 'hot', 'cold-hot'] e.g. 'zeros'
m: [number of malicious agents] e.g. 1
eta: [max attack size] e.g. 1
enforce_symmetry: [boolean on whether initial attack is +eta] e.g. False
k: [number of splits for block sampling] e.g. 2
root_dir [string of directory to save experiment]: e.g. '1_28_25_experiments'
H: [(optional) time horizon to evaluate; overriden by -H argument]
```

### Running Experiments

You can run an experiment with 

```
python calculate_optimal_attack.py -c [config name]
```

To compute optimal attacks for a range of horizons, specify ```H_max``` and ```file``` in ```run_single_experiment.sh``` and then run

```
bash ./run_single_experiment.sh
```

All files will be saved to a single .npy file in the specified directory.

To run a batch of experiments, place all configs in a common directory and alter ```run_batch_experiment.sh``` as needed.

## Sampling 

Right now sampling is done in ```samplers.py```, specifically in ```k_block_sampler```. This is the chunk that most-critically needs to be improved.