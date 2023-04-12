import torch
from scipy import constants
import numpy as np

class Energy():

    """
        A class for computing potential energy of a molecular fragment.

    Attributes:
        ff (ForceField): The force field object to be used for energy calculations.
        device (torch.device): The device to be used for computation.
        bond_params (torch.Tensor): Bond parameters for the force field.
        angle_params (torch.Tensor): Angle parameters for the force field.
        dih_params (torch.Tensor): Dihedral parameters for the force field.
        dih_rb_params (torch.Tensor): Ryckaert-Bellemans dihedral parameters for the force field.
        lj_params (torch.Tensor): Lennard-Jones parameters for the force field.
        atom_mass (torch.Tensor): Atomic masses for the force field.
        bond_min_dist (torch.Tensor): Minimum distance between atoms for bond interactions.
        lj_min_dist (torch.Tensor): Minimum distance between atoms for Lennard-Jones interactions.
        avogadro_const (torch.Tensor): Avogadro constant.
        boltzmann_const (torch.Tensor): Boltzmann constant.
        pi (torch.Tensor): The value of pi.
        n_bond_class (int): The number of bond classes in the force field.
        n_angle_class (int): The number of angle classes in the force field.
        n_dih_class (int): The number of dihedral classes in the force field.
        n_lj_class (int): The number of Lennard-Jones classes in the force field.
    """
    def __init__(self, ff, device):
        """
        Initializes the Energy object.

        Args:
            ff (ForceField): The force field object to be used for energy calculations.
            device (torch.device): The device to be used for computation.
        """
        self.ff = ff
        self.device=device
        self.bond_params = torch.tensor(self.ff.bond_params(), dtype=torch.float32, device=device)
        self.angle_params = torch.tensor(self.ff.angle_params(), dtype=torch.float32, device=device)
        self.dih_params = torch.tensor(self.ff.dih_params(), dtype=torch.float32, device=device)
        self.dih_rb_params = torch.tensor(self.ff.dih_rb_params(), dtype=torch.float32, device=device)
        self.lj_params = torch.tensor(self.ff.lj_params(), dtype=torch.float32, device=device)
        self.atom_mass = torch.tensor([[[atype.mass for atype in self.ff.atom_types.values()]]], dtype=torch.float32, device=device)  # (1, 1, n_atomtypes)

        self.bond_min_dist = torch.tensor(0.01, dtype=torch.float32, device=device)
        self.lj_min_dist = torch.tensor(0.01, dtype=torch.float32, device=device)

        self.avogadro_const = torch.tensor(constants.value(u'Avogadro constant'), dtype=torch.float32, device=device)
        self.boltzmann_const = torch.tensor(constants.value(u'Boltzmann constant'), dtype=torch.float32, device=device)
        self.pi = torch.tensor(np.pi, dtype=torch.float32, device=device)

        self.n_bond_class = len(self.ff.bond_params())
        self.n_angle_class = len(self.ff.angle_params())
        self.n_dih_class = len(self.ff.dih_params())
        self.n_lj_class = len(self.ff.lj_params())

    def convert_to_joule(self, energy):
        """
        Converts energy from kJ/mol to J.

        Args:
        - energy (torch.Tensor): the energy in kJ/mol

        Returns:
        - torch.Tensor: the energy in J
        """
        return energy * 1000.0 / self.avogadro_const

    def all(self, coords, energy_ndx, nonbonded="lj"):
        """
        Calculates the bond, angle, dihedral and non-bonded (Lennard-Jones or quadratic repulsion) energies for a given set of atomic coordinates.

        Args:
        - coords (torch.Tensor): the atomic coordinates (size=[batch_size, num_atoms, 3])
        - energy_ndx (tuple): a tuple containing the indices of the atoms involved in each type of energy (bond, angle, dihedral and non-bonded)
        - nonbonded (str): the type of non-bonded energy to calculate (default is "lj")

        Returns:
        - tuple of torch.Tensors: the bond, angle, dihedral and non-bonded energies
        """
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        if bond_ndx.size()[1]:
            b_energy = self.bond(coords, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_energy = self.angle(coords, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.dih(coords, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            if nonbonded == "lj":
                l_energy = self.lj(coords, lj_ndx)
            else:
                l_energy = self.quadratic_repl(coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)

        return b_energy, a_energy, d_energy, l_energy

    def batch_mean(self, aa_coords, energy_ndx):
        """
        Calculates the mean bond, angle, dihedral and non-bonded energies for a given batch of atomic coordinates.

        Args:
        - aa_coords (torch.Tensor): the atomic coordinates (size=[batch_size, num_atoms, 3])
        - energy_ndx (tuple): a tuple containing the indices of the atoms involved in each type of energy (bond, angle, dihedral and non-bonded)

        Returns:
        - tuple of torch.Tensors: the mean bond, angle, dihedral and non-bonded energies
        """
        fb, fa, fd, fl = self.all(aa_coords, energy_ndx)

        b_loss = torch.mean(fb)
        a_loss = torch.mean(fa)
        d_loss = torch.mean(fd)
        l_loss = torch.mean(fl)

        return b_loss, a_loss, d_loss, l_loss

    def match_loss(self, real_coords, fake_coords, energy_ndx):
        """
        Compute the loss function for matching energies of real and fake structures.

        Parameters:
        real_coords (tensor): Real coordinates tensor.
        fake_coords (tensor): Fake coordinates tensor.
        energy_ndx (tensor): Energy index tensor.

        Returns:
        Tuple of bond loss, angle loss, dihedral loss, and Lennard-Jones loss tensors.
        """
        rb, ra, rd, rl = self.all(real_coords, energy_ndx)
        fb, fa, fd, fl = self.all(fake_coords, energy_ndx)

        b_loss = torch.mean(torch.abs(rb - fb))
        a_loss = torch.mean(torch.abs(ra - fa))
        d_loss = torch.mean(torch.abs(rd - fd))
        l_loss = torch.mean(torch.abs(rl - fl))

        return b_loss, a_loss, d_loss, l_loss

    def get_forces(self, x, energy_ndx):
        """
        Compute the forces for a given set of coordinates and energy index.

        Parameters:
        x (tensor): Coordinates tensor.
        energy_ndx (tensor): Energy index tensor.

        Returns:
        Tensor of force values.
        """
        x = x.requires_grad_(True)
        b_energy, angle_energy, dih_energy, lj_energy = self.all(x, energy_ndx)
        energy = b_energy + angle_energy + dih_energy + lj_energy

        return -torch.autograd.grad(energy, x, torch.ones_like(energy), create_graph=True, retain_graph=True)[0]

    def bond(self, atoms, indices):
        """
        Compute bond energy for given atoms and bond indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of bond indices.

        Returns:
        Tensor of bond energy values.
        """
        ndx1 = indices[:, :, 1]  # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        param = self.bond_params[param_ndx]

        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - a_0
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)

        return en

    def angle(self, atoms, indices):
        """
        Compute angle energy for given atoms and angle indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of angle indices.

        Returns:
        Tensor of angle energy values.
        """
        ndx1 = indices[:, :, 1]  # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])

        vec1 = pos1 - pos2
        vec2 = pos3 - pos2

        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        # norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        # a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2
        # en = en**2
        # en = en * f_c
        # en = en / 2.0
        en = torch.sum(en, dim=1)
        return en

    def dih(self, atoms, indices):
        """
        Compute dihedral energy for given atoms and dihedral indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of dihedral indices.

        Returns:
        Tensor of dihedral energy values.
        """
        ndx1 = indices[:, :, 1]  # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]
        func_type = param[:, :, 2].type(torch.int32)
        # mult = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])

        vec1 = pos2 - pos1
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3

        plane1 = torch.cross(vec1, vec2)
        plane2 = torch.cross(vec2, vec3)

        norm1 = plane1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)

        norm2 = plane2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)

        dot = plane1 * plane2
        dot = torch.sum(dot, dim=2)

        norm = norm1 * norm2  # + 1E-20
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        # a = torch.clamp(a, -1.0, 1.0)

        a = torch.acos(a)

        a = torch.where(func_type == 1, 3*a, a)

        en = a - a_0

        en = torch.where(func_type == 1, (torch.cos(en) + 1.0) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)
        return en

    def dih_rb(self, atoms, indices):
        """
        Compute dihedral energy (Ryckard-Belleman) for given atoms and dihedral indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of dihedral indices.

        Returns:
        Tensor of dihedral energy values.
        """
        ndx1 = indices[:, :, 1]  # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param = self.dih_rb_params[param_ndx]
        f1 = param[:, :, 0]
        f2 = param[:, :, 1]
        f3 = param[:, :, 2]
        f4 = param[:, :, 3]
        f5 = param[:, :, 4]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])

        vec1 = pos2 - pos1
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3

        plane1 = torch.cross(vec1, vec2)
        plane2 = torch.cross(vec2, vec3)

        norm1 = plane1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)

        norm2 = plane2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)

        dot = plane1 * plane2
        dot = torch.sum(dot, dim=2)

        norm = norm1 * norm2 + 1E-20
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)

        en = f1 - f2*a + f3*torch.pow(a, 2) - f4*torch.pow(a, 3) + f5*torch.pow(a, 4)

        en = torch.sum(en, dim=1)
        return en

    def lj(self, atoms, indices):
        """
        Compute LJ energy for given atoms and LJ indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of LJ indices.

        Returns:
        Tensor of LJ energy values.
        """
        ndx1 = indices[:, :, 1]  # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        exp_n = param[:, :, 2]
        exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])  # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        c_n = torch.pow(sigma / dis, exp_n)
        c_m = torch.pow(sigma / dis, exp_m)

        en = 4 * epsilon * (c_n - c_m)

        # cutoff
        # c6_cut = sigma
        # c6_cut = torch.pow(c6_cut, 6)
        # c12_cut = torch.pow(c6_cut, 2)
        # en_cut = 4 * epsilon * (c12_cut - c6_cut)
        # en = en - en_cut
        # en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        return en

    def quadratic_repl(self, atoms, indices):
        """
        Compute quadratic repulsion (soft non-bonded interaction) energy for given atoms and LJ indices.

        Parameters:
        atoms (tensor): Tensor of atoms.
        indices (tensor): Tensor of LJ indices.

        Returns:
        Tensor of LJ energy values.
        """
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        # exp_n = param[:, :, 2]
        # exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])  # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        en = torch.where(dis < sigma, epsilon * (1 - dis / sigma)**2, torch.zeros([], dtype=torch.float32, device=self.device))

        en = torch.sum(en, dim=1)

        return en


