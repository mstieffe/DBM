# DBM

This is the code for the DBM algorithm accompanying our recent publications:
- [Adversarial reverse mapping of equilibrated condensed-phase molecular structures](https://iopscience.iop.org/article/10.1088/2632-2153/abb6d4/meta) 
- [Adversarial reverse mapping of condensed-phase molecular structures: Chemical transferability](https://arxiv.org/abs/2101.04996) 
- [Benchmarking coarse-grained models of organic semiconductors via deep backmapping](https://pure.mpg.de/rest/items/item_3432016/component/file_3432020/content)

DBM is a new method based on ML for the reverse-mapping of molecular systems in the condensed-phase. The method is developed to avoid further energy minimization for relaxation and MD simulations for equilibration of the generated FG structures. Moreover, DBM requires little human intervention, since the reinsertion of local details is learned from training examples.

<p align="center">
  <img width="460" height="auto" src="/images/intro.png">
</p>

The generative adversarial approach is used to train DBM. To this end, a training set consisting of pairs of corresponding CG and FG molecular structures is used. While the target of the generator is to reproduce FG configurations, the CG structures are treated as conditional variables for the generative process. The generator reinserts missing degrees of freedom along CG variables and a discriminator compares the generated structures with the training examples. Since the input for the discriminator consists of both, the CG and the FG configuration, the discriminator evaluates not only the quality of the generated FG structure, but also its consistency with the given CG structure. 

A CNN architecture is used for both models that requires a regular discretization of 3D space, which limits scaling to larger spatial structures. Therefore, the generator is combined with an autoregressive approach that reconstructs the FG structure incrementally, i.e. atom by atom. While DBM only learns local correlations, large-scale features are adapted from the CG structure. As such, only local information is required in each step, which makes the method scalable to larger system sizes. In addition, the local environment approach is a key feature for the generalizability of DBM.

![](/images/autoregressive.png)

The order of reconstruction is defined by a traversal of the molecular graph. Since molecular graphs are generally undirected and can be cyclic or acyclic, the depth-first-search algorithm is applied to obtain a ordering for the atoms. In a first step, DBM generates atom positions with no parents and positions of subsequent atoms are based on the atoms generated in previous steps. However, such forward sampling only yields accurate results if the underlying graph structure has a topological order, i.e. a graph traversal in which each node is visited only after all of its dependencies are explored. As such, accurate sampling of molecular structures requires more feedback than a simple forward sampling strategy provides. To this end, a variant of Gibbs sampling is applied, which subsequently refines the initial molecular structures by iteratively resampling the atom positions. Each further iteration still updates one atom at a time, but uses the knowledge of all other atoms.

The potential energy function of the system can be incorporated in the training of DBM to improve its performance and to monitor the training process. Specifically, the potential energy of generated structures is utilized as an additional term in the cost function of the generator. As such, the potential energy acts as a regularizer that steers the optimization towards generating structures with high Boltzmann weight. It thereby effectively accelerates convergence and helps to improve accuracy.

## Installation

The env.yml file can be used to create a Conda environment that includes all the necessary Python packages.

```
conda env create -f env.yml
```

## Usage

The best way to learn how to use Allegro is through the [Colab Tutorial](https://colab.research.google.com/drive/1-DDO77m3ZrT_ZMukLbDl-snjD6EtEYVO?usp=sharing). This will run entirely on Google's cloud virtual machine, you do not need to install or run anything locally.

### Data
In order to use the DBM algorithm, the user will need to provide the following data:
- **Molecular structures:** Snapshots of CG molecular structures with a `.gro` file extension, formatted as described in the [GROMACS manual](https://manual.gromacs.org/archive/5.0.4/online/gro.html). These files should be stored in a directory named `my_dir` inside `./data/reference_snapshots/my_dir/cg`. If the user wants to train a new model, they will also need to provide reference AA structure files, which should be stored inside `./data/reference_snapshots/my_dir/aa` and named identically to their corresponding CG structure file.
- **Topology:** For each residue with the name `res_name` included in the snapshot, the user must provide a corresponding topology file with an `.itp` file extension for both the AA topology and the CG topology. The formatting of the topology file is described in the [GROMACS manual](https://manual.gromacs.org/archive/5.0.4/online/gro.html). These files should be stored inside `./data/aa_top/res_name.itp` and `./data/cg_top/res_name.itp`, respectively.
- **Mapping:** For each residue, a mapping file with a `.map` file extension is needed to describe the correspondence between CG and AA structures. The file should be stored inside `./data/mapping/res_name.map` (see colab tutorial for further information).
- **Forcefield and features:** The feature mapping and energy terms are specified in a `.ff` file inside `./forcefield/` (see colab tutorial for further information).
- **Config:** The model specifications, such as training data, the model name, resolution, and regularizer, are stored in a `config.ini` file (see colab tutorial for further information).

### Training

Train the model using the following command:

```
python train.py config.ini
```
	
### Validation

Once training is done, we can use the model for the validation data (specified in the `config.ini`)

```
python val.py config.ini
```


