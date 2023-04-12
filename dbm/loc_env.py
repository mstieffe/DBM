import math
from dbm.mol import *


class Local_Env():

    """
    Local_Env is a class that represents a local environment around a given bead in a molecular system.

    Attributes:
    - bead: a Bead object that represents the central bead of the local environment.
    - beads: a list of Bead objects that are in the local environment.
    - mol: a Molecule object that represents the entire molecule that the bead belongs to.
    - beads_intra: a list of Bead objects that are in the same molecule as the central bead.
    - beads_inter: a list of Bead objects that are in different molecules than the central bead.
    - atoms: a list of Atom objects that are in the local environment.
    - atoms_intra: a list of Atom objects that are in the same molecule as the central bead.
    - atoms_intra_heavy: a list of Atom objects that are in the same molecule as the central bead and have a mass of at least 2.0.
    - atoms_inter: a list of Atom objects that are in different molecules than the central bead.
    - box: a Box object that represents the simulation box the molecule is in.
    - rot_mat: a 3x3 numpy array representing the rotation matrix used to align the local environment to a reference frame.
    - atoms_index_dict: a dictionary that maps Atom objects to their indices
    - beads_index_dict: a dictionary that maps Bead objects to their indices

    Methods:
    - get_indices(atoms): given a list of Atom objects, returns a list of their corresponding indices in the local env atoms list.
    - get_cg_indices(beads): given a list of Bead objects, returns a list of their corresponding indices in the local env bead list.
    - rot(positions): given a list of coordinates, rotate the coordinates for local alignment
    - rot_back(positions): given a list of coordinates, rotate the coordinates back to original orientation
    - bead_positions(): compute bead positions in the local environment relative to the central bead
    - atom_positions(): compute atom positions in the local environment relative to the central bead
    - atom_positions_ref(): compute reference atom positions in the local environment relative to the central bead
    - rotation_mtx(bead): Given a bead, which is used as fixpoint, compute rotation matrix
    to align the local environment of the system to the z-axis (align edge of central bead with fixpoint bead to z-axis)
    """

    def __init__(self, bead, beads, box):

        self.bead = bead
        self.mol = bead.mol
        self.beads = beads
        self.beads_intra = [b for b in self.beads if b in self.mol.beads]
        self.beads_inter = list(set(self.beads)-set(self.beads_intra))

        atoms = []
        for b in self.beads:
            atoms += b.atoms
        self.atoms = atoms
        self.atoms_intra = [a for a in self.atoms if a in self.mol.atoms]
        self.atoms_intra_heavy = [a for a in self.atoms_intra if a.type.mass >= 2.0]
        self.atoms_inter = list(set(self.atoms)-set(self.atoms_intra))

        self.box = box

        # Alignment of the local environment using a fixpoint
        if self.bead.fp is None:
            self.rot_mat = np.identity(3)
        elif type(self.bead.fp) is Bead:
            if self.bead.index < self.bead.fp.index:
                fp = self.box.diff_vec(self.bead.fp.center - self.bead.center)
            else:
                fp = self.box.diff_vec(self.bead.center - self.bead.fp.center)
            self.rot_mat = self.rot_mat(fp)
        else:
            raise Exception('something wrong with the alignment!')

        self.atoms_index_dict = dict(zip(self.atoms, range(0, len(self.atoms))))
        self.beads_index_dict = dict(zip(self.beads, range(0, len(self.beads))))

    def get_indices(self, atoms):
        # Return indices for given atoms in the list of local environment atoms
        indices = []
        for a in atoms:
            indices.append(self.atoms_index_dict[a])
        return indices

    def get_cg_indices(self, beads):
        # Return indices for given beads in the list of local environment beads
        indices = []
        for b in beads:
            indices.append(self.beads_index_dict[b])
        return indices

    def rot(self, pos):
        # rotate positions for local alignment
        return np.dot(pos, self.rot_mat)

    def rot_back(self, pos):
        # rotate positions back to original orientation
        return np.dot(pos, self.rot_mat.T)

    def bead_positions(self, kick=0.0):
        # Compute bead positions in the local environment relative to the central bead

        # Get relative positions of beads with respect to the center bead
        positions = [b.center - self.bead.center for b in self.beads]

        # Apply box boundary conditions
        positions = [self.box.diff_vec(pos) for pos in positions]

        # Rotate positions for local alignment
        positions = self.rot(positions)

        # Add random noise to positions (typically not used)
        positions = positions + np.random.normal(-kick, kick, positions.shape)

        return positions


    def atom_positions(self):
        # Compute atom positions in the local environment relative to the central bead

        # Get absolute positions of atoms in the system
        positions = [a.pos + a.bead.center - self.bead.center for a in self.atoms]

        # Apply box boundary conditions
        positions = [self.box.diff_vec(pos) for pos in positions]

        # Rotate positions for local alignment
        positions = self.rot(positions)

        return positions

    def atom_positions_ref(self):
        # Compute reference atom positions in the local environment relative to the central bead

        # Get absolute positions of atoms in the system

        positions = [a.ref_pos + a.bead.center - self.bead.center for a in self.atoms]

        # Apply box boundary conditions
        positions = [self.box.diff_vec(pos) for pos in positions]

        # Rotate positions for local alignment
        positions = self.rot(positions)

        return positions

    def rot_mat(self, fixpoint, distortion=None):
        # Compute rotation matrix to align the local environment of the system to the z-axis

        # Define a vector pointing towards the z-axis
        v1 = np.array([0.0, 0.0, 1.0])

        # Apply distortion to the target axis (if provided)(typically not used)
        if distortion:
            #distortion in RAD, example: distortion = pi/9 (20°) -> target axis will be shifted at max. 20° away from (0,0,1)
            alpha = np.random.uniform(2*np.pi)
            random_len = np.random.uniform(1.0)*np.tan(distortion)
            v1 = np.array([np.cos(alpha)*random_len, np.sin(alpha)*random_len, 1.0])

        # Define the fixpoint vector (target direction of rotation)
        v2 = fixpoint

        # Compute the rotation axis
        v_rot = np.cross(v1, v2)
        v_rot =  v_rot / np.linalg.norm(v_rot)

        # Compute the rotation angle
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        theta = np.arctan2(sinang, cosang)

        # Compute the rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return rot_mat



