# DBM

This is the code for the DBM algorithm accompanying our recent publications:
- [Adversarial reverse mapping of equilibrated condensed-phase molecular structures](https://iopscience.iop.org/article/10.1088/2632-2153/abb6d4/meta) 
- [Adversarial reverse mapping of condensed-phase molecular structures: Chemical transferability](https://arxiv.org/abs/2101.04996) 

DBM is a new method based on ML for the reverse-mapping of molecular systems in the condensed-phase. The method is developed to avoid further energy minimization for relaxation and MD simulations for equilibration of the generated FG structures. Moreover, DBM requires little human intervention, since the reinsertion of local details is learned from training examples.

<p align="center">
  <img width="460" height="auto" src="/images/intro.png">
</p>

![DBM generates Boltzmann-equilibrated atomistic structures conditional on the CG configuration using an adversarial network. It is designed for the backmapping of a condensed-phase molecular systems.](/images/intro.png)

The generative adversarial approach is used to train DBM. To this end, a training set consisting of pairs of corresponding CG and FG molecular structures is used. While the target of the generator is to reproduce FG configurations, the CG structures are treated as conditional variables for the generative process. The generator reinserts missing degrees of freedom along CG variables and a discriminator compares the generated structures with the training examples. Since the input for the discriminator consists of both, the CG and the FG configuration, the discriminator evaluates not only the quality of the generated FG structure, but also its consistency with the given CG structure. 

A CNN architecture is used for both models that requires a regular discretization of 3D space, which limits scaling to larger spatial structures. Therefore, the generator is combined with an autoregressive approach that reconstructs the FG structure incrementally, i.e. atom by atom. While DBM only learns local correlations, large-scale features are adapted from the CG structure. As such, only local information is required in each step, which makes the method scalable to larger system sizes. In addition, the local environment approach is a key feature for the generalizability of DBM.

![](/images/autoregressive.png)

The order of reconstruction is defined by a traversal of the molecular graph. Since molecular graphs are generally undirected and can be cyclic or acyclic, the depth-first-search algorithm is applied to obtain a ordering for the atoms. In a first step, DBM generates atom positions with no parents and positions of subsequent atoms are based on the atoms generated in previous steps. However, such forward sampling only yields accurate results if the underlying graph structure has a topological order, i.e. a graph traversal in which each node is visited only after all of its dependencies are explored. As such, accurate sampling of molecular structures requires more feedback than a simple forward sampling strategy provides. To this end, a variant of Gibbs sampling is applied, which subsequently refines the initial molecular structures by iteratively resampling the atom positions. Each further iteration still updates one atom at a time, but uses the knowledge of all other atoms.

The potential energy function of the system can be incorporated in the training of DBM to improve its performance and to monitor the training process. Specifically, the potential energy of generated structures is utilized as an additional term in the cost function of the generator. As such, the potential energy acts as a regularizer that steers the optimization towards generating structures with high Boltzmann weight. It thereby effectively accelerates convergence and helps to improve accuracy.

## python packages

The env.yml file can be used to create a Conda environment that includes all the necessary Python packages.

```
conda env create -f env.yml
```

## usage

The code is organized as followes:

### data
In order to use the DBM algorithm, the user will need to provide the following data:
- **Molecular structures:** Snapshots of CG molecular structures with a `.gro` file extension, formatted as described in the [GROMACS manual](https://manual.gromacs.org/archive/5.0.4/online/gro.html). These files should be stored in a directory named `my_dir` inside `./data/reference_snapshots/my_dir/cg`. If the user wants to train a new model, they will also need to provide reference AA structure files, which should be stored inside `./data/reference_snapshots/my_dir/aa` and named identically to their corresponding CG structure file.
- **Topology:** For each residue with the name `res_name` included in the snapshot, the user must provide a corresponding topology file with an `.itp` file extension for both the AA topology and the CG topology. The formatting of the topology file is described in the [GROMACS manual](https://manual.gromacs.org/archive/5.0.4/online/gro.html). These files should be stored inside `./data/aa_top/res_name.itp` and `./data/cg_top/res_name.itp`, respectively.
- **Mapping:** For each residue, a mapping file with a `.map` file extension is needed to describe the correspondence between CG and AA structures. The file should be stored inside `./data/mapping/res_name.map`
- **Forcefield and features:** The feature mapping and energy terms are specified in a `.ff` file inside `./forcefield/` (see sample below for further details).
- **Config:** The model specifications, such as training data, the model name, resolution, and regularizer, are stored in a `config.ini` file (see the example below for further details).


### example

In the folllowing, we will train DBM on liquid-phase structures of cumene and octane and then use it for the backmapping of syndiotactic polystyrene (sPS).

#### data preparation

- Snapshots for cumene, octane and sPS be found `./data/reference_snapshots/`. The coarse-grained model of sPS was developed by Fritz *et al*[1]. It represents a polymer as a linear chain, where each monomer is mapped onto two CG beads of different types, denoted A for the chain backbone and B for the phenyl ring. The center of bead A is the center of mass of the CH2 group and the two neighboring CH groups, which are weighted with half of their masses. Bead B is centered at the center of mass of the phenyl group. Cumene is mapped onto three CG beads: Two beads of type A for the backbone, each containing a methyl group and sharing the CH group connected to the phenyl ring, and one bead of type B for the phenyl ring. Octane is mapped onto four beads of type A, where neighboring A beads share a CH2 group.

- We place the toplogy files for all residues used into the corresponding directories. For example, the residue for cumene is named `G065` and we will have one file `G065.itp` inside `./data/aa_top/` as well as `./data/cg_top/`, to define the topologies for the AA as well as the CG structure.

- Next, the mapping file is generated. As an example, the mapping for looks like this:
	```
	[map]
	;atom_index atom_type bead_index bead_type 
	    1  H_AR 2   A
	    2  C_AR 2   A
	    3  C_AR 2   A
	    4  H_AR 2   A
	    5  C_AR 2   A
	    6  H_AR 2   A
	    7  C_AR 2   A
	    8  C_AR 2   A
	    9  H_AR 2   A
	   10  C_AR 2   A
	   11  H_AR 2   A
	   12    C  1   B
	   13    H  1   B
	   14    C  1   B
	   15    H  1   B
	   16    H  1   B
	   17    H  1   B
	   18    C  3   B
	   19    H  3   B
	   20    H  3   B
	   21    H  3   B
	[/map]

	[align]
	;bead_index	fixpoint
	1       3
	2       1
	3	1
	[/align]

	[mult]
	;bead_index	multiples
	1       1
	2       1
	3	1
	[/mult]
	```
	While CG force fields might lead to the sharing of an atom between two neighboring
	beads, the reconstruction of the atom is assigned to only one of the two beads. The mapping is defined between `[map][/map]`. Additionally, we can define a preference axis for each bead to reduce the rotational degrees of freedom. This preference axis can be defined by the position of the central bead and the difference vector to any other bead, which is specified between `[align][/align]`. Furthermore, we can use data augmentation and can increase the number of occurances of a given bead in the training set by integer multiples defined inside `[mult][/mult]`.

- Then we have to define the feature mapping and the energy terms used during training and place it as a `.txt` file inside the directory `./forcefield`. Similar to the three feature channels found for RGB images, we store a number of feature channels in each voxel that represent the presence of other atoms or beads of a certain kind. In our current implementation we made the feature mapping rather flexible such that it can be defined individually by the user. Atom types can be distinguished not only by element but additionally by chemical similarity, i.e., atoms of a given type can be treated as identical in the MD simulation. Furthermore, the user can add channels to distinguish the functional form of interaction to the current atom of interest. Interaction types can include bond, bending angle, torsion, and Lennard-Jones. Similarly, separate channels can be used to encode the different coarse-grained bead types.
The file is oraganized as follows:
	- the name of the feature mapping and the number of exclusions is defined between `[general][/general]`.
	```
	[general]
	; Name      nrexcl
	ff2           2
	[/general]
	```
	- the name of the atom types, their masses, charge und Lennard-Jones parameters are defined inside `[atom_types][/atom_types]`. Atoms of a given type in a local environment of a bead can be displayed in their own channel specified in the *channel* column. Setting the channel number to *-1* deactivates this feature in the local environment representation.

	```
	[atom_types]
	;name	channel   mass      charge       sigma      epsilon
	C	-1       12.0110     0.0000          0.3207      0.3519
	C_AR	-1       12.0110    -0.1150          0.3550      0.2940
	H	-1        1.0080     0.0000          0.2318      0.3180
	H_AR	-1        1.0080     0.1150          0.2420      0.1260
	[/atom_types]
	```

	- bonds, angles and dihedrals can be defined for atom types declared previously. Again, an additional culumn *channels* defines to which channel a given feature is mapped to. Such a feature means, that for a given atom of interest, we would insert all other atoms interacting via the specfic interaction term into the specified channel. Example: Given an angle type `C-C_AR-C_AR` mapped to channel 9. If we want to generate the `C` while the other atoms (the two `C_AR`) being already placed, then both `C_AR` atoms will be drawn into channel nr.9. The idea is, that we split up the local environment into small, incomplete fragments (such as two atoms of an angle) and let the generator place the atom such that it completes the fragment. Note that it is possible to map different features to the *same* channel. In this example, we map a bond feature of `C-H` and `C_AR-H_AR` to the same channel (channel nr. 1), as their equilibrium angle and force constant are very similar. Again, setting the channel number to *-1* deactivates this feature in the local environment representation.

	```
	[bond_types]
	; i     j	channel  func        b0          kb
	C      C	0        1       0.153000      1000.00
	C      H	1        1       0.110000      1000.00
	C_AR   C_AR	2     	 1       0.139000      1000.00
	C_AR   H_AR	1     	 1       0.108000      1000.00
	C      C_AR	3    	 1       0.151000      1000.00
	[/bond_types]

	[angle_types]
	; i     j      k	channel  func       th0         cth
	H      C      H		4       1       109.45       306.40
	C      C      H 	5       1       109.45       448.90
	C      C      C   	6     	1       111.00       530.30
	C_AR   C      H  	7       1       109.45       366.90
	C      C      C_AR 	8   	1       109.45       482.30
	C      C_AR   C_AR  	9   	1       120.00       376.60
	C_AR   C_AR   C_AR  	10  	1       120.00       376.60
	C_AR   C_AR   H_AR  	11  	1       120.00       418.80
	[/angle_types]


	[dihedral_types]
	; i    j     k     l	channel func 
	C      C     C     C	12      1    0.0000   6.0000   3.0000
	H      C     C     C	12      1    0.0000   6.0000   3.0000
	C_AR   C_AR  C_AR  C_AR	13      2    0.0000 167.4000
	C_AR   C_AR  C_AR  C	13      2    0.0000 167.4000
	C_AR   C_AR  C_AR  H_AR	13      2    0.0000 167.4000
	[/dihedral_types]
	```
	- For non-bonded atoms in the environment we can also introduce feature channels specified by the involved atom types.
	```
	[lj_types]
	; i     j       channel
	C       C_AR    14
	C       H_AR    15
	C       C       14
	C       H       15
	C_AR    H_AR    15
	C_AR    C_AR    14
	C_AR    H       15
	H_AR    H_AR    16
	H_AR    H       16
	H       H       16
	[/lj_types]
	```
	- Finally, we introduce feature channels for the CG beads in the environment.
	```
	[bead_types]
	;name	channel
	B	17
	S	18
	[/bead_types]
	```
- Once all of this is done, we can write a `config.ini` file to specify the model parameters:
	```
	[model]
	name = my_model_name # name of the model
	output_dir = my_output_dir # dir for saving
	model_type = regular # network architecture(choose between 'tiny', 'regular' and 'big')
	n_chns = 128 # number of channels in the first conv. layer
	noise_dim = 128 # dimension of noise vector
	sn_gen = 0 # number of iterations for the spectral norm in the generator
	sn_crit = 1 # number of iterations for the spectral norm in the critic
	gp = True # use gradient penalty 

	[forcefield]
	ff_file = ff2.txt # the file specifying the feature mapping/energy terms

	[universe]
	aug = True # use data augmentation 
	align = True # use preference axis to reduce rotation degrees of freedom
	order = dfs # order of reconstruction (choose between dfs, bfs and random), dfs is recommendet
	cutoff = 0.7 # cutoff for the local environment representation in nm
	kick = 0.05 # max random displacement for non-bonded atoms from their CG bead center

	[data]
	train_data = cumene_t350_6, octane_t350_10 # list of snapshot folders used for training
	val_data = sPS_t568_1, cumene_t350_1, octane_t350_1 # list of snapshot folders used for validation

	[training]
	recurrent = True # use recurrent training or not
	n_epoch = 200 # number of epochs
	rand_rot = True # use random rotations
	batchsize = 64
	n_critic = 4 # number of training steps for the critic
	n_checkpoints = 2 # max number of checkpoints
	n_save = 5 # number of epochs between checkpoints

	[prior]
	mode = min # prior/regularizer mode. choose between 'min', 'match' or 'none'
	ratio_bonded_nonbonded = 0.1 # ratio between non-bonded and bonded energy terms in the prior
	weights = 0.0, 0.001, 0.01 # prior weights
	schedule = 10, 20 # schedule for the prior weights (in this example the weight of 0.001 is used after epoch 10)

	[grid]
	resolution = 8 # grid resolution
	length = 1.2 # length spanned by the grid in nm
	sigma = 0.02 # sigma of the gaussian blobs in nm

	[validate]
	n_gibbs = 2 # number of gibbs iterations during backmapping
	```

- Finally, we can train the model
	```
	python train.py config.ini
	```
- Once training is done, we can use the model for the validation data (specified in the `config.ini`)
	```
	python val.py config.ini
	```


