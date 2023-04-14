import mdtraj as md
from mdtraj.core.element import *
from dbm.ff import Energy_NP
from dbm.features import AA_Feature, CG_Feature
from dbm.loc_env import Local_Env
from dbm.top import Top
from dbm.mol import Atom, Bead, Mol
from dbm.box import Box
from dbm.util import read_between
import re

np.set_printoptions(threshold=np.inf)

class Universe():
    """
    This class represents a universe of molecules.
    It initializes instances of molecules, beads and atoms, based on configuration and force field parameters.
    """

    def __init__(self, cfg, path_dict, ff):

        """
        Initializes a new instance of the Universe class.

        :param cfg: configuration parameters
        :param path_dict: dictionary containing file paths
        :param ff: force field parameters
        """

        self.name = path_dict['file_name']
        print("processing file ", self.name)

        # Set parameters
        self.cfg = cfg
        self.aug = int(cfg.getboolean('universe', 'aug'))
        self.align = int(cfg.getboolean('universe', 'align'))
        self.order = cfg.get('universe', 'order')
        self.cutoff_sq = cfg.getfloat('universe', 'cutoff') ** 2
        self.kick = cfg.getfloat('universe', 'kick')

        ## Set the forcefield
        self.ff = ff

        # Use mdtraj to load xyz information
        # Add element to mdtraj (otherwise some will be identified as VS - virtual sides)
        for atype in list(self.ff.atom_types.values()) + list(self.ff.bead_types.values()):
            if atype.name.upper() not in Element._elements_by_symbol:
                _ = Element(0, atype.name, atype.name, 0.0, 0.0)

        # Load cg and aa structure files
        cg = md.load(str(path_dict['cg_path']))
        if path_dict['aa_path']:
            aa = md.load(str(path_dict['aa_path']))

        # Set the number of molecules in the file
        self.n_mol = cg.topology.n_residues

        # Create Box object to capture the dimensions of simulation box and handle periodic boundary conditions
        self.box = Box(path_dict['cg_path'], cfg.getfloat('universe', 'cutoff'))
        self.subbox_dict = self.box.empty_subbox_dict()

        # Go through all molecules in cg file and initialize instances of mols, beads and atoms
        self.atoms, self.beads, self.mols = [], [], []
        for res in cg.topology.residues:

            # Create Mol object and add it to the universe
            self.mols.append(Mol(res.name))

            # Set files for aa topology, cg topology and mapping from the residue name
            aa_top_file = path_dict['data_dir'] / "aa_top" / (res.name + ".itp")
            cg_top_file = path_dict['data_dir'] / "cg_top" / (res.name + ".itp")
            map_file = path_dict['data_dir'] / "mapping" / (res.name + ".map")

            # Loop through all beads in the residue
            beads = []
            for bead in res.atoms:
                # apply periodic boundary conditions
                pos = self.box.move_inside(cg.xyz[0, bead.index])

                # Create bead
                beads.append(Bead(self.mols[-1],
                                  pos,
                                  self.box.subbox(pos),
                                  self.ff.bead_types[bead.element.symbol]))
                # Add bead to the molecule
                self.mols[-1].add_bead(beads[-1])

                # Add bead to the subbox dictionary
                self.subbox_dict[self.box.subbox(pos)].append(beads[-1])

            # Use the mapping file to generate corresponding atoms
            atoms = []
            for line in read_between("[map]", "[", map_file):
                splitted_line = list(filter(None, re.split("\s+", line)))
                type_name = splitted_line[1]
                bead = beads[int(splitted_line[2]) - 1]
                atoms.append(Atom(bead,
                                  self.mols[-1],
                                  self.ff.atom_types[type_name]))

                # if there is a corresponding AA structure file, load the atom positions as a reference position
                if path_dict['aa_path']:
                    # apply PBC and describe its position relative to its corresponding bead
                    atoms[-1].ref_pos = self.box.diff_vec(aa.xyz[0, atoms[-1].index] - atoms[-1].bead.center)
                # add atom to bead
                bead.add_atom(atoms[-1])
                # add atom to molecule
                self.mols[-1].add_atom(atoms[-1])

            # set mol_index back to 0 for the next molecule
            Atom.mol_index = 0

            # Equip molecule object with AA and CG toplogy information
            self.mols[-1].add_aa_top(aa_top_file, self.ff)
            self.mols[-1].add_cg_top(cg_top_file)

            # Add atoms and beads to universe
            self.beads += beads
            self.atoms += atoms

            # Check if there is information for the CG graph travsersal provided in the mapping file
            # First, check if there is a source for the CG graph traversal given
            for line in read_between("[source]", "[", map_file):
                b_ndx = list(filter(None, re.split("\s+", line)))[0]
                if int(b_ndx) > len(self.mols[-1].beads):
                    msg = 'Index for the source of the CG graph traversal provided in the mapping file is '\
                          'not included in the molecule'
                    raise Exception(msg)
                else:
                    self.mols[-1].cg_source = self.mols[-1].beads[int(b_ndx) - 1]
            # Second, check if there is an order of reconstruction provided
            for line in read_between("[order]", "[", map_file):
                ordering = list(filter(None, re.split("\s+", line)))
                if len(set(ordering)) != len(self.mols[-1].beads):
                    msg = 'Order of reconstruction provided in mapping file does not match number of beads'
                    raise Exception(msg)
                else:
                    try:
                        self.mols[-1].cg_order = [self.mols[-1].beads[int(b_ndx) - 1] for b_ndx in ordering]
                    except:
                        msg = 'Indices for the order of reconstruction provided in the mapping file do match ' \
                              'with the beads included in the molecule'
                        raise Exception(msg)

            # If Local Environments should be aligned, read the alignment information
            if self.align:
                for line in read_between("[align]", "[", map_file):
                    try:
                        splitted_line = list(filter(None, re.split("\s+", line)))
                        b_index, fp_index = splitted_line[0], splitted_line[1]
                        if int(b_index) > len(self.mols[-1].beads) or int(fp_index) > len(self.mols[-1].beads):
                            raise Exception('Indices in algn section do not match the molecular structure!')
                        # each bead gets its own fixpoint, which is indicated by the index of another bead
                        self.mols[-1].beads[int(b_index) - 1].fp = self.mols[-1].beads[int(fp_index) - 1]
                    except:
                        raise Exception('sth. wrong with aligment section')

            # If data augmentation is used (some beads are shown more often than others)
            if self.aug:
                for line in read_between("[mult]", "[", map_file):
                    try:
                        splitted_line = list(filter(None, re.split("\s+", line)))
                        b_index, m = splitted_line[0], splitted_line[1]
                        if int(b_index) > len(self.mols[-1].beads) or int(m) < 0:
                            raise Exception('Invalid number of multiples!')
                        self.mols[-1].beads[int(b_index) - 1].mult = int(m)
                    except:
                        raise Exception('sth. wrong with augmentation section')

        # Reset Atom, Bead and Mol index for nex universe
        Atom.index = 0
        Bead.index = 1
        Mol.index = 1

        # set total number of atoms in the universe
        self.n_atoms = len(self.atoms)

        # Generate local envs and features
        self.loc_envs, self.cg_features, self.aa_seq_heavy, self.aa_seq_hydrogens = {}, {}, {}, {}
        self.tops, self.aa_features = {}, {}

        # Loop through all molecules in universe
        for mol in self.mols:

            # Generate sequence used for traversing the molecular graph and generate dictionaries that contain the
            # sequence for all the atoms in a given bead, and all the predecessors for a given atom
            cg_seq, dict_aa_seq_heavy, dict_aa_seq_hydrogens, dict_aa_predecessors = mol.aa_seq(order=self.order,
                                                                                                train=False)

            # Add dictionaries to the universe
            self.aa_seq_heavy = {**self.aa_seq_heavy, **dict_aa_seq_heavy}
            self.aa_seq_hydrogens = {**self.aa_seq_hydrogens, **dict_aa_seq_hydrogens}

            # Loop through the CG sequence and generate a Loc Env for each bead
            for bead, _ in cg_seq:
                # Get all beads within cutoff range around current bead
                env_beads = self.get_loc_beads(bead)
                # Generate Local Env
                self.loc_envs[bead] = Local_Env(bead, env_beads, self.box)
                # Generate CG features
                self.cg_features[bead] = CG_Feature(self.loc_envs[bead], self.ff)

                # Loop through all heavy atoms in the current bead
                for atom in dict_aa_seq_heavy[bead]:
                    # Generate toplogy object for each atom
                    self.tops[atom] = Top(atom, self.loc_envs[bead], dict_aa_predecessors[atom], self.ff)
                    # Generate features for each atom
                    self.aa_features[atom] = AA_Feature(self.loc_envs[bead], self.tops[atom])
                # Loop through all hydrogens atoms in the current bead
                for atom in dict_aa_seq_hydrogens[bead]:
                    # Generate toplogy object for each atom
                    self.tops[atom] = Top(atom, self.loc_envs[bead], dict_aa_predecessors[atom], self.ff)
                    # Generate features for each atom
                    self.aa_features[atom] = AA_Feature(self.loc_envs[bead], self.tops[atom])

        # Create an Energy_NP object (to compute energies using numpy)
        self.energy = Energy_NP(self.tops, self.box)

        # Add random displacement for the atoms around their CG bead
        self.kick_atoms()

    def gen_bead_seq(self, train=False):
        # Method to generate a CG sequence for the whole universe
        bead_seq = []
        mols = self.mols[:]
        np.random.shuffle(mols)  # order of molecules should not matter
        for mol in mols:
            # Obtain bead order for each molecule
            bead_seq += list(zip(*mol.cg_seq(order=self.order, train=train)))[0]
        return bead_seq

    def kick_atoms(self):
        # This method perturbs the positions of all atoms in the universe.
        for a in self.atoms:
            a.pos = np.random.normal(-self.kick, self.kick, 3)

    def get_loc_beads(self, bead):
        # This method returns a list of beads that are within a certain
        # distance (specified by cutoff_sq) of a given bead

        # Use the subbox dictionary to make search more efficient
        subboxes = self.box.nn_subboxes(bead.subbox)
        nn_beads = []
        # Only lood into neighbouring subboxes
        for sb in subboxes:
            nn_beads += self.subbox_dict[sb]

        # Center the given beads around the current bead
        centered_positions = np.array([b.center for b in nn_beads]) - bead.center
        centered_positions = np.array([self.box.diff_vec(pos) for pos in centered_positions])

        # Compute distances
        centered_positions_sq = [r[0] * r[0] + r[1] * r[1] + r[2] * r[2] for r in centered_positions]

        # Apply cutoff
        indices = np.where(np.array(centered_positions_sq) <= self.cutoff_sq)[0]
        return [nn_beads[i] for i in indices]

    def write_gro_file(self, filename, ref=False):
        # Method to write the universe information to a .gro file

        with open(filename, 'w') as f:
            f.write('{:s}\n'.format(self.name))
            f.write('{:5d}\n'.format(self.n_atoms))

            n = 1
            for mol in self.mols:
                for a in mol.atoms:
                    # Decide whether we write the actual Atom position (backmapped position) or reference position
                    if ref:
                        pos = a.ref_pos
                    else:
                        pos = a.pos
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                        a.mol.index,
                        a.mol.name,
                        a.type.name + str(a.mol.atoms.index(a) + 1),
                        n % 100000,
                        pos[0] + a.bead.center[0],
                        pos[1] + a.bead.center[1],
                        pos[2] + a.bead.center[2],
                        0, 0, 0))
                    n = n + 1

            f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                self.box.dim[0][0],
                self.box.dim[1][1],
                self.box.dim[2][2],
                self.box.dim[1][0],
                self.box.dim[2][0],
                self.box.dim[0][1],
                self.box.dim[2][1],
                self.box.dim[0][2],
                self.box.dim[1][2]))
