# Graph Coloring with Physics-Inspired Graph Neural Networks

In this repo we show how to solve graph coloring problems with physics-inspired graph neural networks, as outlined 
in Martin J. A. Schuetz, J. Kyle Brubaker, Zhihuai Zhu, Helmut G. Katzgraber, _Graph Coloring with Physics-Inspired 
Graph Neural Networks_, [arXiv:2202.01606](https://arxiv.org/abs/2202.01606). In `gc_example.ipynb` we solve one 
example COLOR graph problem, but our approach can easily be extended to other problems, for instance the citations 
graphs. For the implementation of the graph neural network layers (GCNConv, SAGEConv, etc) we use the 
open-source ```dgl``` library. 

## Introduction

We show how graph neural networks can be used to solve the canonical graph coloring problem.
We frame graph coloring as a multi-class node classification problem and utilize an unsupervised
training strategy based on the statistical physics Potts model. Generalizations to other multi-class
problems such as community detection, data clustering, and the minimum clique cover problem
are straightforward. We provide numerical benchmark results and illustrate our approach with an
end-to-end application for a real-world scheduling use case within a comprehensive encode-process-
decode framework. Our optimization approach performs on par or outperforms existing solvers,
with the ability to scale to problems with millions of variables.

## Example code

In this notebook ([gc_example.ipynb](gc_example.ipynb)) we show how to solve graph coloring problems 
with physics-inspired graph neural networks, as outlined in M. J. A. Schuetz, J. K. Brubaker, Z. Zhu, H. G. Katzgraber,
_Graph Coloring with Physics-Inspired Graph Neural Networks_, 
[arXiv:2202.01606](https://arxiv.org/abs/2202.01606). Here we focus on one example COLOR graph instance 
(see [data](Data) ) and provide one set of known hyperparameters that will yield a cost of 0 when run on 
a GPU instance. For the implementation of the graph neural network we use the open-source ```dgl``` library. 

## Environment Setup

Please note we have provided a `requirements.txt` file, which defines the environment required to run this code. 
Because some of the packages are not available on default OSX conda channels, we have also provided suggested 
channels to find them on. These can be distilled into a single line as such:

> conda create -n \<environment_name\> python=3.7 --file requirements.txt -c dglteam

We include logic to determine whether to use CUDA (e.g. GPU) or CPU backend, but it's important to note that 
DGL library must be installed with CUDA drivers included or this script will fail. We include comments in 
`requirements.txt` on how to handle this, which we reproduce here for visibility:

```
# For CPU-based DGL installation (default option):
dgl

# For GPU-based DGL installation:
# DGL installation is complicated if you want to include CUDA (i.e. use GPUs)
# First, go to the terminal and look for the CUDA version, i.e. via `nvidia-smi`
# With that version, you can insert into the following command:
# `conda install -c dglteam dgl-cudaXY.Z`
# For instance, if CUDA version = 11.0:
# `conda install -c dglteam dgl-cuda11.0`
#dgl-cuda11.0

```

## Data

To help keep repository size low, we do not include the input dataset. The COLOR dataset can be downloaded from 
this site: https://mat.tepper.cmu.edu/COLOR/instances.html

The direct download link to the `instances.tar` object is here: 
https://mat.tepper.cmu.edu/COLOR/instances/instances.tar

We suggest downloading these under the parent path `data/input/COLOR` and unpacking there, such that you have 
`queen5_5.col` and the path `data/input/COLOR/instances/queen5_5.col`. However, this is left to the user, and the 
parent path to file `queen5_5.col` (or whichever specific problem the user chooses) can be specified in the 
`input_parent` variable in **cell 3**

You can unpack the tar file via any standard utility (i.e. by double-clicking on it) or via command line, such 
as (on linux) `tar -xvf instances.tar`

**Cells 4-6** contain logic to automate the downloading and unpacking of the COLOR datasets. The user can uncomment 
the code in these cells as an alternative to manually executing the above steps.

## Code Execution

Once the virtual environment is established (see above), running the code is straightforward. From the parent folder, 
launch the notebook via 

> conda activate \<environment_name\>
> jupyter notebook gc_example.ipynb

Once in the notebook, run the cells via 

`Cell` > `Run All` 

or 

`Kernel` > `Restart & Run All`

**NOTE:** On a standard laptop (e.g. a 2019 13" MacBook Pro), the full notebook takes ~30-60 seconds to run. 
This should not vary much across hardware, as the code is not parallelized and the problem instance and 
GNN model are small enough to fit in memory.

### Hyperparameter Optimization

In [arXiv:2202.01606](https://arxiv.org/abs/2202.01606) Appendix I, we report the hyperparameters used to achieve 
the performance across problem instances listed in Table I. Most of these parameters were found through hyperparameter 
optimization (HPO) leveraging the [hyperopt library](http://hyperopt.github.io/hyperopt/). We do not include the 
logic to run HPO in our example notebook. For an example of how to run the HPO process, see the example in the
[Getting Started section of Hyperopt's site](http://hyperopt.github.io/hyperopt/#getting-started). It is important to 
note that the results of HPO, and any given model training run, can be sensitive to variables like the random seed of 
the appropriate RNG libraries (i.e. `numpy.random`, `torch.random`) or whether training is run on CPU or GPU. Such 
factors can lead to differences in results across machines if not handled carefully.

Here is the list of key hyperparameters we would suggest tuning to improve results:

`number_epochs` The maximum number of epochs to train the GNN model

`learning_rate` Learning rate for the optimizer (Adam, AdamW, etc)

`dim_embedding` Dimensionality of embedding vector, per input

`hidden_dim` Size of intermediate hidden layer

`dropout` Fraction of nodes to drop out in each layer

`tolerance` Minimum change in loss to be considered an improvement

`patience` How many epochs without improvement before early stopping is triggered

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. 
See the LICENSE file.

The sample code within this documentation is made available under the MIT-0 license. See the LICENSE-SAMPLECODE file.

## Citation

```
@article{Schuetz2022a,
  title={Graph Coloring with Physics-Inspired Graph Neural Networks},
  author={Schuetz, Martin J. A. and Brubaker, J. Kyle and Zhu, Zhihuai and Katzgraber, Helmut G.},
  journal={arXiv preprint arXiv:2202.01606},
  year={2022}
}
```
