import numpy as np

class AA_Feature():
    """
    AA_Feature is a class that generates atomic feature vectors and energy indices for a local environment.
    The feature vectors are based on the atom type and local AA graph structure. The energy
    index is used to calculate the potential energy between atoms.
    Attributes:
        loc_env (LocalEnvironment): An instance of the LocalEnvironment class that describes the local
                                    environment of the top atom.
        top (Topology): An instance of the Topology class that describes the local graph structure of a given atom,
        which is defined in top.

    Methods:
        featvec(key='all', inter=True): Generates the atomic feature vector for the local env atoms based on the
                                        specified key and interatomic contributions.
        energy_ndx(key='all'): Generates the energy index dictionary for the atom based on the specified key.
        bond_ndx(key='all'): Generates the bond index for the atom based on the specified key.
        angle_ndx(key='all'): Generates the angle index for the atom based on the specified key.
        dih_ndx(key='all'): Generates the dihedral index for the atom based on the specified key.
        dih_rb_ndx(key='all'): Generates the Ryckaert-Bellemans dihedral index for the atom based on the
                               specified key.
        lj_ndx(key='all', inter=True): Generates the Lennard-Jones index for the atom based on the specified
                                       key and interatomic contributions.
    """

    def __init__(self, loc_env, top):
        """
        Initializes the AA_Feature class.
        Args:
            loc_env (LocalEnvironment): An instance of the LocalEnvironment class that describes the local
                                        environment of the top atom.
            top (Topology): An instance of the Topology class that describes the local graph of the atom.
        """
        self.loc_env = loc_env
        self.top = top

        self.fv_init = self.featvec(key='predecessor')
        self.energy_ndx_init = self.energy_ndx(key='predecessor')

        #for init step without any interatomic contributions
        self.fv_init_intra = self.featvec(key='predecessor', inter=False)

        if top.atom.type.mass >= 2.0:
            self.fv_gibbs = self.featvec(key='heavy')
            self.energy_ndx_gibbs = self.energy_ndx(key='heavy')
        else:
            self.fv_gibbs = self.featvec(key='all')
            self.energy_ndx_gibbs = self.energy_ndx(key='all')

        self.repl = np.ones(len(self.loc_env.atoms), dtype=bool)
        self.repl[self.loc_env.atoms_index_dict[self.top.atom]] = False


    def featvec(self, key='all', inter=True):
        """
        Generates the atomic feature vector for the atom based on the specified key and interatomic contributions.
        Args:
            key (str): A string indicating the key to specify the type of feature vector to generate. Default is 'all'.
            inter (bool): A boolean indicating whether to include interatomic contributions in the feature vector.
                          Default is True.
        Returns:
            atom_featvec (numpy.ndarray): A numpy array representing the atomic feature vector.
        """

        # create a numpy array of zeros with dimensions (number of atoms in the local environment, number of feature channels)
        atom_featvec = np.zeros((len(self.loc_env.atoms), self.top.ff.n_channels))

        # loop over all atoms in the local environment and set the value in the corresponding channel to 1
        # based on the atom's type channel (if it exists)
        for index in range(0, len(self.loc_env.atoms)):
            if self.loc_env.atoms[index].type.channel >= 0:
                atom_featvec[index, self.loc_env.atoms[index].type.channel] = 1

        # loop over all bonds in the specified key and set the value in the corresponding channel to 1
        # based on the bond's type channel (if it exists)
        for bond in self.top.bonds[key]:
            if bond.type.channel >= 0:
                indices = self.loc_env.get_indices(bond.atoms)
                atom_featvec[indices, bond.type.channel] = 1

        # loop over all angles in the specified key and set the value in the corresponding channel to 1
        # based on the angle's type channel (if it exists)
        for angle in self.top.angles[key]:
            if angle.type.channel >= 0:
                indices = self.loc_env.get_indices(angle.atoms)
                atom_featvec[indices, angle.type.channel] = 1

        # loop over all dihedrals in the specified key and set the value in the corresponding channel to 1
        # based on the dihedral's type channel (if it exists)
        for dih in self.top.dihs[key] + self.top.dihs_rb[key]:
            #print(dih.type.name)
            if dih.type.channel >= 0:
                indices = self.loc_env.get_indices(dih.atoms)
                atom_featvec[indices, dih.type.channel] = 1

        # if interatomic contributions are not requested, append 'intra_' to the key to select only intra-molecular interactions
        if not inter:
            key = 'intra_' + key

        # loop over all Lennard-Jones interactions in the specified key and set the value in the corresponding channel to 1
        # based on the Lennard-Jones interaction's type channel (if it exists)
        for lj in self.top.ljs[key]:
            if lj.type.channel >= 0:
                indices = self.loc_env.get_indices(lj.atoms)
                atom_featvec[indices, lj.type.channel] = 1

        # set the value of the atom's feature vector with itself to 0
        atom_featvec[self.loc_env.atoms_index_dict[self.top.atom], :] = 0
        return atom_featvec


    def energy_ndx(self, key='all'):
        # A dictionary that stores indices for different types of interactions
        d = {'bonds': self.bond_ndx(key),
                 'angles': self.angle_ndx(key),
                 'dihs': self.dih_ndx(key),
                 'dihs_rb': self.dih_rb_ndx(key),
                 'ljs': self.lj_ndx(key),
                 'ljs_intra': self.lj_ndx(key, inter=False)}
        return d

    def bond_ndx(self, key='all'):
        # List that stores a tupel that contains a Bond together with the indices of the involved atoms
        indices = []
        for bond in self.top.bonds[key]:
            indices.append(tuple([self.top.ff.bond_index_dict[bond.type],
                            self.loc_env.atoms_index_dict[bond.atoms[0]],
                            self.loc_env.atoms_index_dict[bond.atoms[1]]]))
        return indices

    def angle_ndx(self, key='all'):
        # List that stores a tupel that contains a Angle together with the indices of the involved atoms
        indices = []
        for angle in self.top.angles[key]:
            indices.append(tuple([self.top.ff.angle_index_dict[angle.type],
                            self.loc_env.atoms_index_dict[angle.atoms[0]],
                            self.loc_env.atoms_index_dict[angle.atoms[1]],
                            self.loc_env.atoms_index_dict[angle.atoms[2]]]))
        return indices

    def dih_ndx(self, key='all'):
        # List that stores a tupel that contains a Dihedrals together with the indices of the involved atoms
        indices = []
        for dih in self.top.dihs[key]:
            indices.append(tuple([self.top.ff.dih_index_dict[dih.type],
                            self.loc_env.atoms_index_dict[dih.atoms[0]],
                            self.loc_env.atoms_index_dict[dih.atoms[1]],
                            self.loc_env.atoms_index_dict[dih.atoms[2]],
                            self.loc_env.atoms_index_dict[dih.atoms[3]]]))
        return indices

    def dih_rb_ndx(self, key='all'):
        # List that stores a tupel that contains a RB Dihedrals together with the indices of the involved atoms
        indices = []
        for dih in self.top.dihs_rb[key]:
            indices.append(tuple([self.top.ff.dih_rb_index_dict[dih.type],
                            self.loc_env.atoms_index_dict[dih.atoms[0]],
                            self.loc_env.atoms_index_dict[dih.atoms[1]],
                            self.loc_env.atoms_index_dict[dih.atoms[2]],
                            self.loc_env.atoms_index_dict[dih.atoms[3]]]))
        return indices

    def lj_ndx(self, key='all', inter=True):
        # List that stores a tupel that contains a LJ interaction together with the indices of the involved atoms
        if not inter:
            key = 'intra_' + key
        indices = []
        for lj in self.top.ljs[key]:
            indices.append(tuple([self.top.ff.lj_index_dict[lj.type],
                            self.loc_env.atoms_index_dict[lj.atoms[0]],
                            self.loc_env.atoms_index_dict[lj.atoms[1]]]))
        return indices


class CG_Feature():

    def __init__(self, loc_env, ff):

        self.loc_env = loc_env
        self.ff = ff

        self.fv = self.featvec()

        #self.chn_fv = self.chn_featvec()

    def featvec(self):
        bead_featvec = np.zeros((len(self.loc_env.beads), self.ff.n_channels))
        for index in range(0, len(self.loc_env.beads)):
            bead_featvec[index, self.loc_env.beads[index].type.channel] = 1
        for b in self.loc_env.beads_intra:
            bead_featvec[self.loc_env.beads.index(b), -1] = 1
        return bead_featvec

