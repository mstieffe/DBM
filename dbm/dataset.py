from torch.utils.data import Dataset
from dbm.util import make_grid_np, rand_rot_mtx, voxelize_gauss
from dbm.stats import *

class DS(Dataset):
    """
    This class defines a PyTorch Dataset for the DBM algorithm.
    It generates a dataset using a Recurrent_Generator instances

    Attributes:
    - data: instance of a Data object that contains the molecular structures and topology
    - elems: a list of dictionaries representing the elements of the molecular structures
    - resolution: the resolution of the grid used for the local environments centered at a bead position
    - delta_s: the grid spacing used for the local environments
    - sigma: the standard deviation of the Gaussian used to voxelize the molecular
             structure in the local environment
    - rand_rot: a boolean indicating whether random rotations should be applied
                during training
    - align: an integer indicating whether the molecules should be aligned during
             generation
    - grid: the grid used for the molecular structures

    Methods:
    - __init__(self, data, cfg, train=True): initializes the dataset with the given
      molecular data and configuration file
    - __len__(self): returns the length of the dataset
    - __getitem__(self, ndx): retrieves an item from the dataset at the given index
    """

    def __init__(self, data, cfg, train=True, verbose=False):
        """
        Initializes the DS_seq instance with the given molecular data and configuration
        file. Uses a series of Recurrent_Generator instances to generate molecular structures
        for various configurations.

        Arguments:
        - data: an instance of a Data object
        - cfg: a configuration file
        - train: a boolean indicating whether the dataset is for training or testing

        Returns:
        None
        """

        self.data = data

        # set properties
        self.resolution = cfg.getint('grid', 'resolution')
        self.delta_s = cfg.getfloat('grid', 'length') / cfg.getint('grid', 'resolution')
        self.sigma = cfg.getfloat('grid', 'sigma')
        self.rand_rot = cfg.getboolean('training', 'rand_rot')
        self.align = int(cfg.getboolean('universe', 'align'))

        # print properties
        if verbose:
            print("Using an {}x{}x{} grid with spacing {}".format(self.resolution, self.resolution, self.resolution, self.delta_s))
            print("The sigma value for the Gauss-blobb representation is {}".format(self.sigma))
            print("Alignment of the local environments is set to {}".format(self.align))
            print("Random rotations around alignment-axis is during training is set to {}".format(self.rand_rot))

        # create the grid
        self.grid = make_grid_np(self.delta_s, self.resolution)

        # create generator objects for the different molecular representations
        generators = []
        generators.append(Recurrent_Generator(data, hydrogens=False, gibbs=False, train=train, rand_rot=False))
        generators.append(Recurrent_Generator(data, hydrogens=False, gibbs=True, train=train, rand_rot=False))
        # if hydrogens are included in the representation, add corresponding generator objects
        if cfg.getboolean('training', 'hydrogens'):
            generators.append(Recurrent_Generator(data, hydrogens=True, gibbs=False, train=train, rand_rot=False))
            generators.append(Recurrent_Generator(data, hydrogens=True, gibbs=True, train=train, rand_rot=False))

        # combine all elements from the different generator objects
        self.elems = []
        for g in generators:
            self.elems += g.all_elems()

    def __len__(self):
        """
        Returns the length of the dataset.

        Args:
            None

        Returns:
            int: The length of the dataset.
        """
        return len(self.elems)

    def __getitem__(self, ndx):
        """
        Returns the dataset element at the given index.

        Args:
            ndx (int): The index of the element to be returned.

        Returns:
            tuple: A tuple containing the following elements:
                - ndarray: Voxelized representation of the target atom.
                - int: Type of the target atom.
                - ndarray: feature vectors describing the atoms in the loc environment.
                - ndarray: replacement vector (one-hot), indicates at which index the generated atom will be inserted in the local environment
                - ndarray: a mask for elements for empty tokens in the sequence (zero padding)
                - ndarray: Voxelized representation of the atoms.
                - ndarray: Voxelized representation of the beads.
                - ndarray: Coordinates of the atoms in the local environment (used to compute energies).
                - tuple: A tuple containing the following elements:
                    - ndarray: Indices of the bonded energy terms.
                    - ndarray: Indices of the angle energy terms.
                    - ndarray: Indices of the dihedral energy terms.
                    - ndarray: Indices of the Lennard-Jones energy terms.
        """

        # generate random rotation matrix for training
        if self.rand_rot:
            R = rand_rot_mtx(self.data.align)
        else:
            R = np.eye(3)

        # get the element at the given index
        d = self.elems[ndx]

        # rotate and create voxelized representation of the target atom, atoms in the environment, and beads
        target_atom = voxelize_gauss(np.dot(d['target_pos'], R.T), self.sigma, self.grid)
        aa_coords = np.dot(d['aa_pos'], R.T).astype('float32')
        atom_grid = voxelize_gauss(aa_coords, self.sigma, self.grid)
        bead_grid = voxelize_gauss(np.dot(d['cg_pos'], R.T), self.sigma, self.grid)

        # create feature representation for beads
        # atom features need to be generated on the fly, as they depend on the previously generated atom
        cg_features = d['cg_feat'][:, :, None, None, None] * bead_grid[:, None, :, :, :]
        # (N_beads, N_chn, 1, 1, 1) * (N_beads, 1, N_x, N_y, N_z)
        cg_features = np.sum(cg_features, 0)
        # (N_chn, N_x, N_y, N_z)

        # define elems (used for iterating over the sequence)
        elems = (target_atom, d['target_type'], d['aa_feat'], d['repl'], d['mask'])
        # define initial (start values for the sequential reconstruction)
        initial = (atom_grid, cg_features, aa_coords)
        # indices for the energy functions, such that we can generate the energy contribution of generated atoms
        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])

        return elems, initial, energy_ndx
