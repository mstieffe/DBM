import numpy as np
import networkx as nx
import itertools
from dbm.util import read_between
import re

class Atom():
    """
    This class represents an atom.
    """

    index = 0  # Class attribute to keep track of the index of each Atom object created
    mol_index = 0  # Class attribute to keep track of the index of each Atom object created within a molecule

    def __init__(self, bead, mol, type, ref_pos=None):
        """
        Initializes a new Atom object with a bead, molecule, type and reference position.

        Args:
            bead (Bead): The bead object to which the atom belongs.
            mol (int): The index of the molecule to which the atom belongs.
            type (str): The type of the atom.
            ref_pos (numpy array, optional): The reference position of the atom. Defaults to None.

        Attributes:
            index (int): The index of the atom object.
            mol_index (int): The index of the atom object within a molecule.
            bead (Bead): The bead object to which the atom belongs.
            mol (int): The index of the molecule to which the atom belongs.
            pos (numpy array): The position of the atom.
            type (str): The type of the atom.
            ref_pos (numpy array): The reference position of the atom.

        Returns:
            None
        """
        self.index = Atom.index
        Atom.index += 1
        self.mol_index = Atom.mol_index
        Atom.mol_index += 1
        self.bead = bead
        self.mol = mol
        self.pos = np.zeros(3)
        self.type = type
        if ref_pos is None:
            self.ref_pos = np.zeros(3)
        else:
            self.ref_pos = ref_pos

class Bead():
    """
    This class represents a CG bead.
    """
    index = 1  # Class attribute to keep track of the index of each Bead object created

    def __init__(self, mol, center, subbox, type, index=None, atoms=None, fp=None, mult=1):
        """
        Initializes a new Bead object with a molecule, center, subbox, type, index, atoms, fp and multiplicity.

        Args:
            mol (int): The index of the molecule to which the bead belongs.
            center (numpy array): The center of the bead.
            subbox (int): The index of the subbox in which the bead is located.
            type (str): The type of the bead.
            index (int, optional): The index of the bead object. Defaults to None.
            atoms (list, optional): The list of atoms to which the bead belongs. Defaults to None.
            fp (bead, optional): The index of the fixpoint for alignment of the local environment. Defaults to None.
            mult (int, optional): The multiplicity of the bead. Defaults to 1.

        Attributes:
            index (int): The index of the bead object.
            mol (int): The index of the molecule to which the bead belongs.
            center (numpy array): The center of the bead.
            subbox (int): The index of the subbox in which the bead is located.
            type (str): The type of the bead.
            atoms (list): The list of atoms to which the bead belongs.
            loc_beads (list): The list of beads in the local environment of the bead.
            loc_atoms (list): The list of atoms in the local environment of the bead.
            fp (bead): a bead object used as fixpoint for the alignment of the local environment
            mult (int): The multiplicity of the bead for data augmentation.

        Returns:
            None
        """
        if index:
            self.index = index
        else:
            self.index = Bead.index
            Bead.index += 1
        self.mol = mol
        self.center = center
        self.subbox = subbox
        self.type = type
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        self.loc_beads = None
        self.loc_atoms = None

        self.fp = fp
        self.mult = mult

    def add_atom(self, atom):
        self.atoms.append(atom)


class Mol():
    """
    A class to represent a molecule.

    Attributes:
    -----------
    name : str
        The name of the molecule.
    beads : list, optional
        A list of beads in the molecule. Default is None.
    atoms : list, optional
        A list of atoms in the molecule. Default is None.
    G : NetworkX Graph, optional
        A NetworkX graph representing the molecule's atoms and bonds. Default is None.
    G_heavy : NetworkX Graph, optional
        A NetworkX graph representing the molecule's heavy atoms and bonds. Default is None.
    hydrogens : list, optional
        A list of hydrogen atoms in the molecule. Default is None.
    bonds : list
        A list of bonds in the molecule. Default is an empty list.
    angles : list
        A list of angles in the molecule. Default is an empty list.
    dihs : list
        A list of dihedrals in the molecule. Default is an empty list.
    dihs_rb : list
        A list of R-B dihedrals in the molecule. Default is an empty list.
    excls : list
        A list of excluded atom pairs in the molecule. Default is an empty list.
    pairs : list
        A list of bonded atom pairs in the molecule. Default is an empty list.
    cg_edges : list
        A list of coarse-grained edges in the molecule. Default is an empty list.
    fp : dict
        A dictionary of fixpoints for the beads for local alignment. Default is an empty dictionary.

    Methods:
    --------
    add_bead(bead):
        Adds a bead to the molecule.
    add_atom(atom):
        Adds an atom to the molecule.
    add_aa_top(top_file, ff):
        Adds all-atom topology information to the molecule.
    add_cg_top(top_file):
        Adds coarse-grained topology information to the molecule.
    make_aa_graph():
        Creates a NetworkX graph representing the molecule's atoms and bonds.
    make_cg_graph():
        Creates a NetworkX graph representing the molecule's beads and coarse-grained edges.
    """

    index = 1
    def __init__(self, name, beads = None, atoms = None):
        self.name = name
        self.index = Mol.index
        Mol.index += 1
        if beads is None:
            self.beads = []
        else:
            self.beads = beads
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

        self.G = None
        self.G_heavy = None
        self.hydrogens = None

        self.bonds = []
        self.angles = []
        self.dihs = []
        self.dihs_rb = []
        self.excls = []
        self.pairs = []

        self.cg_edges = []

        self.fp = {}

    def add_bead(self, bead):
        # Add a bead to the molecule
        self.beads.append(bead)

    def add_atom(self, atom):
        # Add an atom to the molecule
        self.atoms.append(atom)

    def add_aa_top(self, top_file, ff):
        """
        Adds all-atom topology information to the molecule.

        Parameters:
        -----------
        top_file : str
            Path to the topology file.
        ff : object
            The forcefield object to use.
        """
        for line in read_between("[bonds]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            if len(splitted_line) >= 2:
                index1 = int(splitted_line[0]) - 1
                index2 = int(splitted_line[1]) - 1
                bond = ff.make_bond([self.atoms[index1], self.atoms[index2]])
                if bond:
                    self.add_bond(bond)


        for line in read_between("[angles]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            if len(splitted_line) >= 3:
                index1 = int(splitted_line[0]) - 1
                index2 = int(splitted_line[1]) - 1
                index3 = int(splitted_line[2]) - 1
                angle = ff.make_angle([self.atoms[index1], self.atoms[index2], self.atoms[index3]])
                if angle:
                    self.add_angle(angle)

        for line in read_between("[dihedrals]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            if len(splitted_line) >= 4:
                index1 = int(splitted_line[0]) - 1
                index2 = int(splitted_line[1]) - 1
                index3 = int(splitted_line[2]) - 1
                index4 = int(splitted_line[3]) - 1
                dih = ff.make_dih([self.atoms[index1], self.atoms[index2], self.atoms[index3], self.atoms[index4]])
                if dih:
                    try:
                        if dih.type.func == 3:
                            self.add_dih_rb(dih)
                        else:
                            self.add_dih(dih)
                    except:
                        self.add_dih(dih)

        for line in read_between("[exclusions]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            if len(splitted_line) >= 2:
                index1 = int(splitted_line[0]) - 1
                #index2 = int(splitted_line[1]) - 1
                #self.add_excl([self.atoms[index1], self.atoms[index2]])
                for ndx in splitted_line[1:]:
                    self.add_excl([self.atoms[index1], self.atoms[int(ndx)-1]])

        for line in read_between("[pairs]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            if len(splitted_line) >= 2:
                index1 = int(splitted_line[0]) - 1
                index2 = int(splitted_line[1]) - 1
                self.add_pair([self.atoms[index1], self.atoms[index2]])

        self.make_aa_graph()

    def add_cg_top(self, top_file):
        """
        Adds CG topology information to the molecule.

        Parameters:
        -----------
        top_file : str
            The name of the topology file.
        """
        for line in read_between("[bonds]", "[", top_file):
            splitted_line = list(filter(None, re.split("\s+", line)))
            index1 = int(splitted_line[0]) - 1
            index2 = int(splitted_line[1]) - 1
            self.add_cg_edge([self.beads[index1], self.beads[index2]])

        self.make_cg_graph()

    def add_bond(self, bond):
        # Adds AA bond
        self.bonds.append(bond)

    def add_angle(self, angle):
        # Adds AA angle
        self.angles.append(angle)

    def add_dih(self, dih):
        # Adds AA dihedrals
        self.dihs.append(dih)

    def add_dih_rb(self, dih):
        # Adds AA RB dihedral
        self.dihs_rb.append(dih)

    def add_excl(self, excl):
        # Adds non-bonded exclusion
        self.excls.append(excl)

    def add_pair(self, pair):
        # Adds nonbonded pair
        self.pairs.append(pair)

    def add_cg_edge(self, edge):
        # Adds CG bond
        self.cg_edges.append(edge)

    def add_fp(self, bead_ndx, fp_ndx):
        # Adds a CG fixpoint for local alignment
        self.fp[self.beads[bead_ndx]] = self.beads[fp_ndx]

    def make_aa_graph(self):
        # Uses networkX to generate a molecular AA graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.atoms)
        edges = [bond.atoms for bond in self.bonds]
        self.G.add_edges_from(edges)

        heavy_atoms = [a for a in self.atoms if a.type.mass >= 2.0]
        heavy_edges = [e for e in edges if e[0].type.mass >= 2.0 and e[1].type.mass >= 2.0]
        self.G_heavy = nx.Graph()
        self.G_heavy.add_nodes_from(heavy_atoms)
        self.G_heavy.add_edges_from(heavy_edges)

        self.hydrogens = [a for a in self.atoms if a.type.mass < 2.0]

    def make_cg_graph(self):
        # Uses networkX to generate a molecular CG graph
        self.G_cg = nx.Graph()
        self.G_cg.add_nodes_from(self.beads)
        self.G_cg.add_edges_from(self.cg_edges)

    def cg_seq(self, order="dfs", train=True):
        # Generate a sequence of CG beads, i.e. a traversal through the CG graph

        # Breadth-first search order
        if order == "bfs":
            # Generate the bead sequence by performing a BFS traversal of the CG graph
            edges = list(nx.bfs_edges(self.G_cg, np.random.choice(self.beads)))
            beads = [edges[0][0]] + [e[1] for e in edges]
            # Random search order
        elif order == "random":
            # Generate the bead sequence by randomly choosing neighbors until all beads are included
            beads = [np.random.choice(self.beads)]
            pool = []
            for n in range(1, len(self.beads)):
                pool += list(nx.neighbors(self.G_cg, beads[-1]))
                pool = list(set(pool))
                next = np.random.choice(pool)
                while next in beads:
                    next = np.random.choice(pool)
                pool.remove(next)
                beads.append(next)
        # If order is not bfs or random, use Depth-first order as default
        else:
            beads = list(nx.dfs_preorder_nodes(self.G_cg))

        # Augment data for undersampled beads
        seq = []
        for n in range(0, len(beads)):
            if train:
                # Repeat the current bead and its parent beads (if any) by the multiplicity of the current bead
                seq += [(beads[n], beads[:n])]*beads[n].mult
            else:
                # Add the current bead and its parent beads (if any) to the sequence
                seq.append((beads[n], beads[:n]))

        # Shuffle sequence if training
        if train:
            np.random.shuffle(seq)

        return seq

    def aa_seq(self, order="dfs", train=True):
        # Generate a sequence of atoms, i.e. a traversal through the AA graph

        # Create a list of heavy atoms in the molecule
        mol_atoms_heavy = [a for a in self.atoms if a.type.mass >= 2.0]

        # Create empty dictionaries to store atom sequences and predecessors
        atom_seq_dict_heavy = {}
        atom_seq_dict_hydrogens = {}
        atom_predecessors_dict = {}

        # Get the sequence of CG beads from the cg_seq() function
        cg_seq = self.cg_seq(order=order, train=train)

        # Loop through each CG bead in the sequence
        for bead, predecessor_beads in cg_seq:

            # Get the atoms in the CG bead
            bead_atoms = bead.atoms

            # Separate the heavy atoms and hydrogens in the CG bead
            heavy_atoms = [a for a in bead_atoms if a.type.mass >= 2.0]
            hydrogens = [a for a in bead_atoms if a.type.mass < 2.0]

            # Get the atoms in the predecessor beads
            predecessor_atoms = list(itertools.chain.from_iterable([b.atoms for b in set(predecessor_beads)]))
            predecessor_atoms_heavy = [a for a in predecessor_atoms if a.type.mass >= 2.0]
            predecessor_atoms_hydrogens = [a for a in predecessor_atoms if a.type.mass < 2.0]

            # Find the start atom for the sequence
            psble_start_nodes = []
            n_heavy_neighbors = []
            for a in heavy_atoms:
                n_heavy_neighbors.append(len(list(nx.all_neighbors(self.G_heavy, a))))
                for n in nx.all_neighbors(self.G_heavy, a):
                    if n in predecessor_atoms_heavy:
                        psble_start_nodes.append(a)
            if psble_start_nodes:
                # just take first one...
                start_atom = psble_start_nodes[0]
            else:
                start_atom = heavy_atoms[np.array(n_heavy_neighbors).argmin()]


            # Generate the sequence of heavy atoms in the bead
            if order == "bfs":
                edges = list(nx.bfs_edges(self.G.subgraph(heavy_atoms), start_atom))
                atom_seq = [start_atom] + [e[1] for e in edges]
            elif order == "random":
                atom_seq = [start_atom]
                pool = []
                for n in range(1, len(heavy_atoms)):
                    pool += list(nx.neighbors(self.G.subgraph(heavy_atoms), atom_seq[-1]))
                    pool = list(set(pool))
                    next = np.random.choice(pool)
                    while next in atom_seq:
                        next = np.random.choice(pool)
                    pool.remove(next)
                    atom_seq.append(next)
            else:
                atom_seq = list(nx.dfs_preorder_nodes(self.G.subgraph(heavy_atoms), start_atom))

            # Randomly shuffle the hydrogens and add them to the end of the sequence
            np.random.shuffle(hydrogens)
            #atom_seq = atom_seq + hydrogens

            # Add the atom sequence and predecessor lists to the dictionaries
            for n in range(0, len(atom_seq)):
                atom_predecessors_dict[atom_seq[n]] = predecessor_atoms_heavy + atom_seq[:n]
            for n in range(0, len(hydrogens)):
                atom_predecessors_dict[hydrogens[n]] = mol_atoms_heavy + predecessor_atoms_hydrogens + hydrogens[:n]
            atom_seq_dict_heavy[bead] = atom_seq
            atom_seq_dict_hydrogens[bead] = hydrogens

        # Return the CG sequence, heavy atom sequence dictionary, hydrogen sequence dictionary, and predecessor dictionary
        return cg_seq, atom_seq_dict_heavy, atom_seq_dict_hydrogens, atom_predecessors_dict


