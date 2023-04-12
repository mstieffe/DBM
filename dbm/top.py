import itertools
import networkx as nx
#from dbm.mol import *

class Top():

    """
    A class to calculate various molecular properties of a single atom within a molecular system.

    Public methods:
    filter_heavy(atoms) -- returns a list of all atoms with mass >= 2.0 in a given list of atoms
    filter_predecessors(atoms) -- returns a list of all atoms in a given list of atoms that are predecessors of the current atom
    get_bonds() -- returns a dictionary of bond objects containing various types of bonds involving the current atom
    get_angles() -- returns a dictionary of angle objects containing various types of angles involving the current atom
    get_dihs() -- returns a dictionary of dihedral objects containing various types of dihedrals involving the current atom
    get_dihs_rb() -- returns a dictionary of RB dihedral objects containing various types of RB dihedrals involving the current atom
    get_ljs() -- returns a dictionary of LJ objects containing various types of LJ interactions involving the current atom

    Properties:
    atom -- the current atom
    loc_env -- the local environment of the current atom
    predecessors -- the predecessors of the current atom
    ff -- the force field used for the calculation
    bonds -- a dictionary of bond objects containing various types of bonds involving the current atom
    angles -- a dictionary of angle objects containing various types of angles involving the current atom
    dihs -- a dictionary of dihedral objects containing various types of dihedrals involving the current atom
    dihs_rb -- a dictionary of RB dihedral objects containing various types of RB dihedrals involving the current atom
    ljs -- a dictionary of LJ objects containing various types of LJ interactions involving the current atom
    """

    def __init__(self, atom, loc_env, predecessors, ff):
        """
        Initializes a Top object.

        Arguments:
        atom -- the current atom
        loc_env -- the local environment of the current atom
        predecessors -- the predecessors of the current atom
        ff -- the force field used for the calculation
        """
        self.atom = atom
        self.loc_env = loc_env
        self.predecessors = predecessors
        self.ff = ff

        # calculate various types of molecular properties involving the current atom
        self.bonds = self.get_bonds()
        self.angles = self.get_angles()
        self.dihs = self.get_dihs()
        self.dihs_rb = self.get_dihs_rb()
        self.ljs = self.get_ljs()

    def filter_heavy(self, atoms):
        """
        Filters out all atoms in a given list of atoms that have mass < 2.0.

        Arguments:
        atoms -- the list of atoms to filter

        Returns:
        a list of all atoms with mass >= 2.0 in the given list of atoms
        """
        return [a for a in atoms if a.type.mass >= 2.0]

    def filter_predecessors(self, atoms):
        """
        Filters out all atoms in a given list of atoms that are not predecessors of the current atom.

        Arguments:
        atoms -- the list of atoms to filter

        Returns:
        a list of all atoms in the given list of atoms that are predecessors of the current atom
        """
        return [a for a in atoms if a in self.predecessors]

    def get_bonds(self):
        """
        Returns a dictionary of all bonds, heavy bonds, and predecessor bonds for the current atom.

        Returns
        -------
        Dict[str, List of Bonds]
            A dictionary of bonds categorized by type.
        """
        bonds = {'all': [], 'heavy': [], 'predecessor': []}
        for bond in self.atom.mol.bonds:
            if self.atom in bond.atoms:
                bond.atoms.remove(self.atom) #current atom should always be the first element in the atom list
                bond.atoms = [self.atom] + bond.atoms
                bonds['all'].append(bond)
                if len(self.filter_heavy(bond.atoms)) == 2:
                    bonds['heavy'].append(bond)
                if len(self.filter_predecessors(bond.atoms)) == 1:
                    bonds['predecessor'].append(bond)
        return bonds

    def get_angles(self):
        """
        Returns a dictionary of all angles, heavy angles, and predecessor angles for the current atom.

        Returns
        -------
        Dict[str, List of Angles]
            A dictionary of angles categorized by type.
        """
        angles = {'all': [], 'heavy': [], 'predecessor': []}
        for angle in self.atom.mol.angles:
            if self.atom in angle.atoms:
                angles['all'].append(angle)
                if len(self.filter_heavy(angle.atoms)) == 3:
                    angles['heavy'].append(angle)
                if len(self.filter_predecessors(angle.atoms)) == 2:
                    angles['predecessor'].append(angle)
        return angles

    def get_dihs(self):
        """
        Returns a dictionary of all dihedrals, heavy dihedrals, and predecessor dihedrals for the current atom.

        Returns
        -------
        Dict[str, List of Dihs]
            A dictionary of dihedrals categorized by type.
        """
        dihs = {'all': [], 'heavy': [], 'predecessor': []}
        for dih in self.atom.mol.dihs:
            if self.atom in dih.atoms:
                dihs['all'].append(dih)
                if len(self.filter_heavy(dih.atoms)) == 4:
                    dihs['heavy'].append(dih)
                if len(self.filter_predecessors(dih.atoms)) == 3:
                    dihs['predecessor'].append(dih)
        return dihs

    def get_dihs_rb(self):
        """
        Returns a dictionary of all RB dihedrals, heavy RB dihedrals, and predecessor RB dihedrals for the current atom.

        Returns
        -------
        Dict[str, List of Dih_RB]
            A dictionary of RB dihedrals categorized by type.
        """
        dihs = {'all': [], 'heavy': [], 'predecessor': []}
        for dih in self.atom.mol.dihs_rb:
            if self.atom in dih.atoms:
                dihs['all'].append(dih)
                if len(self.filter_heavy(dih.atoms)) == 4:
                    dihs['heavy'].append(dih)
                if len(self.filter_predecessors(dih.atoms)) == 3:
                    dihs['predecessor'].append(dih)
        return dihs

    def get_ljs(self):
        """
        Returns a dictionary of LJ interactions for the current atom.

        Returns
        -------
        Dict[str, List of LJ interactions]
            A dictionary of LJ pairs categorized by type.
        """

        # Create an empty dictionary to store LJ interactions.
        ljs = {}

        # Find excluded atoms (i.e., atoms that do not have a nonbonded interaction with the current atom)
        # in the molecular system.
        excl_atoms = []
        for excl in self.atom.mol.excls:
            if self.atom in excl:
                excl_atoms.append(excl)
        # Create a list of unique excluded atoms.
        excl_atoms = list(set(itertools.chain.from_iterable(excl_atoms)))
        # Remove the current atom from the list of excluded atoms, if it exists.
        if self.atom in excl_atoms: excl_atoms.remove(self.atom)

        # Find pair atoms (i.e., atoms that have a nonbonded interaction with the current atom) in the molecular system.
        pair_atoms = []
        for pair in self.atom.mol.pairs:
            if self.atom in pair:
                pair_atoms.append(pair)
        # Create a list of unique pair atoms.
        pair_atoms = list(set(itertools.chain.from_iterable(pair_atoms)))
        # Remove the current atom from the list of pair atoms, if it exists.
        if self.atom in pair_atoms: pair_atoms.remove(self.atom)

        # Find bonded atoms up to a certain cutoff distance (n_excl) from the current atom.
        lengths, paths = nx.single_source_dijkstra(self.atom.mol.G, self.atom, cutoff=self.ff.n_excl)
        n_excl_atoms = list(set(itertools.chain.from_iterable(paths.values())))

        # Combine the list of excluded atoms and bonded atoms up to n_excl to get a new list of excluded atoms.
        excl_atoms = n_excl_atoms + excl_atoms

        # Find LJ interactions for all intra-molecular atoms that are not excluded.
        lj_atoms_intra = list(set(self.loc_env.atoms_intra) - set(excl_atoms))

        # Add pair atoms to the list of intra-molecular atoms for LJ interaction.
        lj_atoms_intra = set(lj_atoms_intra + pair_atoms)

        # Generate all intra-molecular LJ interactions.
        ljs['intra_all'] = self.ff.make_ljs(self.atom, lj_atoms_intra)

        # Find intra-molecular LJ interactions for atoms that only contain heavy elements, and for atoms
        # are in the list of predecessors
        ljs['intra_heavy'], ljs['intra_predecessor'] = [], []
        for lj in ljs['intra_all']:
            if len(self.filter_heavy(lj.atoms)) == 2:
                ljs['intra_heavy'].append(lj)
            if len(self.filter_predecessors(lj.atoms)) == 1:
                ljs['intra_predecessor'].append(lj)

        # Find inter-molecular LJ interactions.
        ljs['inter'] = self.ff.make_ljs(self.atom, self.loc_env.atoms_inter)
        # Find inter-molecular LJ interactions for atoms that only contain heavy elements.
        ljs['inter_heavy'] = []
        for lj in ljs['inter']:
            if len(self.filter_heavy(lj.atoms)) == 2:
                ljs['inter_heavy'].append(lj)

        # Combine all intra- and inter-molecular LJ interactions.
        ljs['all'] = ljs['intra_all'] + ljs['inter']

        # Combine all intra-molecular heavy-atom LJ interactions and inter-molecular heavy-atom LJ interactions.
        ljs['heavy'] = ljs['intra_heavy'] + ljs['inter']

        # Combine all intra-molecular predecessor LJ interactions and inter-molecular LJ interactions.
        ljs['predecessor'] = ljs['intra_predecessor'] + ljs['inter']

        return ljs