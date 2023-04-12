import numpy as np
import math

class Recurrent_Generator():
    """
    a class representing a recurrent generator, which traverses the molecular system, atom by atom, and returns
    feature vectors for each atom that stores the condition to find a position for the given atom,
    and energy indices, that store the information of interactions involved with the current atom to
    compute corresponding energies
    """
    def __init__(self, data, train=False, hydrogens=False, gibbs=False, rand_rot=False, pad_seq=True, ref_pos=True, inter=True):

        """
        Initialize the Recurrent_Generator class.
        Parameters:
            data (Data): An object of class Data that holds the data.
            train (bool): A flag indicating whether to use the training set or validation set.
            hydrogens (bool): A flag indicating whether to include hydrogens or not.
            gibbs (bool): A flag indicating whether to return data for gibbs sampling or not.
            rand_rot (bool): A flag indicating whether to randomly rotate the structure.
            pad_seq (bool): A flag indicating whether to pad the sequences or not.
            ref_pos (bool): A flag indicating whether to use reference positions or not.
            inter (bool): A flag indicating whether to use inter-atomic terms or not.
        """

        # Set instance variables
        self.data = data
        self.train = train
        self.hydrogens = hydrogens
        self.gibbs = gibbs
        self.rand_rot = rand_rot
        self.pad_seq = pad_seq
        self.ref_pos = ref_pos
        self.inter = inter

        # Set the samples to use
        if train:
            self.samples = self.data.samples_train
        else:
            self.samples = self.data.samples_val

    def __iter__(self):
        """
        returns dictionary containing AA and CG positions, Aa and CG feature vectors and energy indices
        """
        # Iterate through each sample
        for sample in self.samples:

            # Generate sequence of beads to visit
            bead_seq = sample.gen_bead_seq(train=self.train)

            # Choose dict for atoms in a given bead (heavy or hydrogens)
            if self.hydrogens:
                atom_seq_dict = sample.aa_seq_hydrogens
            else:
                atom_seq_dict = sample.aa_seq_heavy

            # Loop through all beads in the sequence
            for bead in bead_seq:
                d = {}
                d['sample'] = sample
                d['bead'] = bead

                # Get CG features and local environment for the bead
                cg_f = sample.cg_features[bead]
                loc_env = sample.loc_envs[bead]

                # Pad CG features
                d["cg_feat"] = np.array(self.pad2d(cg_f.fv, self.data.max['beads_loc_env']), dtype=np.float32)
                # Pad CG positions
                d["cg_pos"] = np.array(self.pad2d(loc_env.bead_positions(), self.data.max['beads_loc_env']), dtype=np.float32)

                # Pad atomic positions
                if self.ref_pos:
                    d["aa_pos"] = self.pad2d(loc_env.atom_positions_ref(), self.data.max['atoms_loc_env'])
                else:
                    d["aa_pos"] = self.pad2d(loc_env.atom_positions(), self.data.max['atoms_loc_env'])

                # Check if the atom_seq_dict is empty
                if not atom_seq_dict[bead]:
                    continue

                target_pos, target_type, aa_feat, repl = [], [], [], []
                bonds_ndx, angles_ndx, dihs_ndx, dihs_rb_ndx, ljs_ndx = [], [], [], [], []

                # Iterate through each atom in the bead
                for atom in atom_seq_dict[bead]:

                    # Get feature vector
                    aa_f = sample.aa_features[atom]

                    # Get target position (reference position)
                    t_pos = loc_env.rot(np.array([atom.ref_pos]))
                    target_pos.append(t_pos)

                    # Get target aton type and encode it as a one-hot vector
                    t_type = np.zeros(self.data.ff.n_atom_chns)
                    t_type[atom.type.index] = 1
                    target_type.append(t_type)


                    # get feature vectors and energy indices
                    if self.gibbs:
                        # use full information if gibbs flag is set to true
                        if self.inter:
                            # include inttermolecular atoms
                            atom_featvec = self.pad2d(aa_f.fv_gibbs, self.data.max['atoms_loc_env'])
                            ljs_ndx += aa_f.energy_ndx_gibbs['ljs']

                        else:
                            # include only intra molecular atoms
                            atom_featvec = self.pad2d(aa_f.featvec(key='all', inter=False), self.data.max['atoms_loc_env'])
                            ljs_ndx += aa_f.energy_ndx_gibbs['ljs_intra']

                        # energy indices for gibbs step
                        bonds_ndx += aa_f.energy_ndx_gibbs['bonds']
                        angles_ndx += aa_f.energy_ndx_gibbs['angles']
                        dihs_ndx += aa_f.energy_ndx_gibbs['dihs']
                        dihs_rb_ndx += aa_f.energy_ndx_gibbs['dihs_rb']

                    else:
                        # use only information of predecessors otherwise
                        if self.inter:
                            # include inttermolecular atoms
                            atom_featvec = self.pad2d(aa_f.fv_init, self.data.max['atoms_loc_env'])
                            ljs_ndx += aa_f.energy_ndx_init['ljs']
                        else:
                            # include only intra molecular atoms
                            atom_featvec = self.pad2d(aa_f.featvec(key='predecessor', inter=False), self.data.max['atoms_loc_env'])
                            ljs_ndx += aa_f.energy_ndx_init['ljs_intra']

                        # energy indices for initial step
                        bonds_ndx += aa_f.energy_ndx_init['bonds']
                        angles_ndx += aa_f.energy_ndx_init['angles']
                        dihs_ndx += aa_f.energy_ndx_init['dihs']
                        dihs_rb_ndx += aa_f.energy_ndx_init['dihs_rb']

                    # Append AA feature vector
                    aa_feat.append(atom_featvec)

                    # Replace vector: marks the index of the target atom in "aa_pos" (needed for recurrent training), i.e.
                    # to insert the generated atom into correct position in the list local env atoms
                    r = self.pad1d(aa_f.repl, self.data.max['atoms_loc_env'], value=True)
                    repl.append(r)

                # Padding
                d["bonds_ndx"] = np.array(self.pad_energy_ndx(bonds_ndx, self.data.max['bonds_per_bead']), dtype=np.int64)
                d["angles_ndx"] = np.array(self.pad_energy_ndx(angles_ndx, self.data.max['angles_per_bead'], tuple([-1, 1, 2, 3])), dtype=np.int64)
                d["dihs_ndx"] = np.array(self.pad_energy_ndx(dihs_ndx, self.data.max['dihs_per_bead'], tuple([-1, 1, 2, 3, 4])), dtype=np.int64)
                d["dihs_rb_ndx"] = np.array(self.pad_energy_ndx(dihs_rb_ndx, self.data.max['dihs_rb_per_bead'], tuple([-1, 1, 2, 3, 4])), dtype=np.int64)
                d["ljs_ndx"] = np.array(self.pad_energy_ndx(ljs_ndx, self.data.max['ljs_per_bead']), dtype=np.int64)

                if self.pad_seq:
                    for n in range(0, self.data.max['seq_len'] - len(atom_seq_dict[bead])):
                        target_pos.append(np.zeros((1, 3)))
                        target_type.append(target_type[-1])
                        aa_feat.append(np.zeros(aa_feat[-1].shape))
                        repl.append(np.ones(repl[-1].shape, dtype=bool))
                d["target_pos"] = np.array(target_pos, dtype=np.float32)
                d["target_type"] = np.array(target_type, dtype=np.float32)
                d["aa_feat"] = np.array(aa_feat, dtype=np.float32)
                d["repl"] = np.array(repl, dtype=np.bool)

                # Mask for sequences < max_seq_len, required to exclude padded tokens from the training
                mask = np.zeros(self.data.max['seq_len'])
                mask[:len(atom_seq_dict[bead])] = 1
                d["mask"] = np.array(mask, dtype=np.float32)

                # Apply random rotation to the structure if requested
                if self.rand_rot:
                    rot_mat = self.rand_rot_mat()
                    d["target_pos"] = np.dot(d["target_pos"], rot_mat)
                    d["aa_pos"] = np.dot(d["aa_pos"], rot_mat)
                    d["cg_pos"] = np.dot(d["cg_pos"], rot_mat)

                d['loc_env'] = loc_env
                d['atom_seq'] = atom_seq_dict[bead]

                yield d

    def all_elems(self):
        # Return all elements of the generator
        g = iter(self)
        elems = []
        for e in g:
            elems.append(e)
        return elems


    def rand_rot_mat(self):
        # Generate random rotation matrix

        #rotation axis
        if self.data.align:
            v_rot = np.array([0.0, 0.0, 1.0])
        else:
            phi = np.random.uniform(0, np.pi * 2)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v_rot = np.array([x, y, z])

        #rotation angle
        theta = np.random.uniform(0, np.pi * 2)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        rot_mat = rot_mat.astype('float32')

        return rot_mat

    def pad1d(self, vec, max, value=0):
        # Pad 1D array
        vec = np.pad(vec, (0, max - len(vec)), 'constant', constant_values=(0, value))
        return vec

    def pad2d(self, vec, max, value=0):
        # Pad 2D array
        vec = np.pad(vec, ((0, max - len(vec)), (0, 0)), 'constant', constant_values=(0, value))
        return vec

    def pad_energy_ndx(self, ndx, max, value=tuple([-1, 1, 2])):
        # Pad energy indices

        # Remove dupicates
        ndx = list(set(ndx))
        # Pad
        for n in range(0, max - len(ndx)):
            ndx.append(tuple(value))
        return ndx