import torch
import numpy as np
import torch.nn as nn

class Histogram():
    """
    Class to compute the histogram of molecular structures
    """
    def __init__(self, cfg, ff, device, n_bins=32):
        """Initialize a new Histogram instance.

        Args:
        - cfg: ConfigParser object containing the configuration parameters
        - ff: force field used to obtain the interactions for which histograms are computed
        - device: device to use for the computations
        - n_bins: number of bins in the histogram (default: 32)

        Properties:
        - n_bins: number of bins in the histogram
        - device: device used for the computations
        - bond: GaussianHistogram_Dis object representing the distance histogram
        - angle: GaussianHistogram_Angle object representing the angle histogram
        - dih: GaussianHistogram_Dih object representing the dihedral angle histogram
        - nb: GaussianHistogram_LJ object representing the Lennard-Jones interaction histogram
        """

        self.n_bins = n_bins
        self.device = device
        self.bond = GaussianHistogram_Dis(bins=n_bins, min=0.0, max=cfg.getfloat('grid', 'length')/2.0, sigma=0.01, ff=ff, device=device)
        self.angle = GaussianHistogram_Angle(bins=n_bins, min=0, max=180, sigma=2.0, ff=ff, device=device)
        self.dih = GaussianHistogram_Dih(bins=n_bins, min=0, max=180, sigma=4.0, ff=ff, device=device)
        self.nb = GaussianHistogram_LJ(bins=n_bins, min=0.0, max=cfg.getfloat('grid', 'length')*1.2, sigma=0.01, ff=ff, device=device)

    def all(self, coords, energy_ndx):
        """Compute the histogram of all energy terms for a given set of coordinates and energy indices.

        Args:
        - coords: tensor of shape (n_atoms, 3) containing the atomic coordinates
        - energy_ndx: tuple of four tensors containing the indices of the atoms involved in each energy term
          (bond_ndx, angle_ndx, dih_ndx, lj_ndx)

        Returns:
        - b_dstr: tensor of shape (n_bins, 1) containing the distance histogram
        - a_dstr: tensor of shape (n_bins, 1) containing the angle histogram
        - d_dstr: tensor of shape (n_bins, 1) containing the dihedral angle histogram
        - nb_dstr: tensor of shape (n_bins, 1) containing the Lennard-Jones interaction histogram
        """
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        if bond_ndx.size()[1] and bond_ndx.size()[1]:
            b_dstr = self.bond(coords, bond_ndx)
        else:
            b_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1] and angle_ndx.size()[1]:
            a_dstr = self.angle(coords, angle_ndx)
        else:
            a_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1] and dih_ndx.size()[1]:
            d_dstr = self.dih(coords, dih_ndx)
        else:
            d_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1] and lj_ndx.size()[1]:
            nb_dstr = self.nb(coords, lj_ndx)
        else:
            nb_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        return b_dstr, a_dstr, d_dstr, nb_dstr

    def loss(self, real_coords, fake_coords, energy_ndx):
        """
        Calculates the dstr loss using the JS divergence of different distributions defined by the interactions included in the forcefield.

        Args:
        - real_coords (torch.Tensor): The real coordinates of the system.
        - fake_coords (torch.Tensor): The fake coordinates of the system.
        - energy_ndx (tuple): A tuple of tensors containing the indices of the different energy terms.

        Returns:
        A tuple of four tensors representing the bond, angle, dihedral, and non-bonded dstr losses, respectively.
        """
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        # Compute Bond distributions for fake and real structures
        # Use Jenson-Shannon divergence between both distributions as loss
        if bond_ndx.size()[1]:

            b_dstr_real = self.bond(real_coords, bond_ndx)
            b_dstr_fake = self.bond(fake_coords, bond_ndx)
            b_dstr_avg = 0.5 * (b_dstr_real + b_dstr_fake)
            b_dstr_loss = 0.5 * ((b_dstr_real * (b_dstr_real / b_dstr_avg).log()).sum(0) + (
                        b_dstr_fake * (b_dstr_fake / b_dstr_avg).log()).sum(0))
        else:
            b_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute Angle distributions for fake and real structures
        # Use Jenson-Shannon divergence between both distributions as loss
        if angle_ndx.size()[1]:
            a_dstr_real = self.angle(real_coords, angle_ndx)
            a_dstr_fake = self.angle(fake_coords, angle_ndx)
            a_dstr_avg = 0.5 * (a_dstr_real + a_dstr_fake)
            a_dstr_loss = 0.5 * ((a_dstr_real * (a_dstr_real / a_dstr_avg).log()).sum(0) + (
                        a_dstr_fake * (a_dstr_fake / a_dstr_avg).log()).sum(0))
        else:
            a_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute Dih distributions for fake and real structures
        # Use Jenson-Shannon divergence between both distributions as loss
        if dih_ndx.size()[1]:
            d_dstr_real = self.dih(real_coords, dih_ndx)
            d_dstr_fake = self.dih(fake_coords, dih_ndx)
            d_dstr_avg = 0.5 * (d_dstr_real + d_dstr_fake)
            d_dstr_loss = 0.5 * ((d_dstr_real * (d_dstr_real / d_dstr_avg).log()).sum(0) + (
                        d_dstr_fake * (d_dstr_fake / d_dstr_avg).log()).sum(0))
        else:
            d_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute LJ distributions for fake and real structures
        # Use Jenson-Shannon divergence between both distributions as loss
        if lj_ndx.size()[1]:
            nb_dstr_real = self.nb(real_coords, lj_ndx)
            nb_dstr_fake = self.nb(fake_coords, lj_ndx)
            nb_dstr_avg = 0.5 * (nb_dstr_real + nb_dstr_fake)
            nb_dstr_loss = 0.5 * ((nb_dstr_real * (nb_dstr_real / nb_dstr_avg).log()).sum(0) + (
                        nb_dstr_fake * (nb_dstr_fake / nb_dstr_avg).log()).sum(0))
        else:
            nb_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        return torch.sum(b_dstr_loss), torch.sum(a_dstr_loss), torch.sum(d_dstr_loss), torch.sum(nb_dstr_loss)

    def abs_loss(self, real_coords, fake_coords, energy_ndx):
        """
        Calculates the dstr abs loss using the abs value of the difference for each bin in the different distributions
        defined by the interactions included in the forcefield.

        Args:
        - real_coords (torch.Tensor): The real coordinates of the system.
        - fake_coords (torch.Tensor): The fake coordinates of the system.
        - energy_ndx (tuple): A tuple of tensors containing the indices of the different energy terms.

        Returns:
        A tuple of four tensors representing the bond, angle, dihedral, and non-bonded dstr losses, respectively.
        """

        # Compute Bond distributions for fake and real structures
        # Use bin-wise abs difference of both distributions as loss
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:

            b_dstr_real = self.bond(real_coords, bond_ndx)
            b_dstr_fake = self.bond(fake_coords, bond_ndx)
            b_loss = torch.mean(torch.abs(b_dstr_real - b_dstr_fake))

        else:
            b_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute Angle distributions for fake and real structures
        # Use bin-wise abs difference of both distributions as loss
        if angle_ndx.size()[1]:
            a_dstr_real = self.angle(real_coords, angle_ndx)
            a_dstr_fake = self.angle(fake_coords, angle_ndx)
            a_loss = torch.mean(torch.abs(a_dstr_real - a_dstr_fake))

        else:
            a_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute Dih distributions for fake and real structures
        # Use bin-wise abs difference of both distributions as loss
        if dih_ndx.size()[1]:
            d_dstr_real = self.dih(real_coords, dih_ndx)
            d_dstr_fake = self.dih(fake_coords, dih_ndx)
            d_loss = torch.mean(torch.abs(d_dstr_real - d_dstr_fake))

        else:
            d_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Compute LJ distributions for fake and real structures
        # Use bin-wise abs difference of both distributions as loss
        if lj_ndx.size()[1]:
            nb_dstr_real = self.nb(real_coords, lj_ndx)
            nb_dstr_fake = self.nb(fake_coords, lj_ndx)
            nb_loss = torch.mean(torch.abs(nb_dstr_real - nb_dstr_fake))

        else:
            nb_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        return b_loss, a_loss, d_loss, nb_loss

class GaussianHistogram(nn.Module):
    """
    A class representing a Gaussian histogram module that can be used as a layer in a neural network.

    Args:
    bins (int): The number of bins in the histogram.
    min (float): The minimum value of the histogram.
    max (float): The maximum value of the histogram.
    sigma (float): The standard deviation of the Gaussian distribution used to smooth the histogram.

    Attributes:
    bins (int): The number of bins in the histogram.
    min (float): The minimum value of the histogram.
    max (float): The maximum value of the histogram.
    sigma (float): The standard deviation of the Gaussian distribution used to smooth the histogram.
    delta (float): The width of each bin.
    centers (torch.Tensor): The centers of each bin.
    """
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers[None, None, :]

    def forward(self, x):
        """
        Forward pass of the Gaussian histogram module.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        x (torch.Tensor): The output tensor representing the smoothed histogram.
        """
        x = x[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=(0, 1))
        x = x / x.sum()
        return x

class GaussianHistogram_Dis(nn.Module):
    """
    A PyTorch module that computes a Gaussian histogram for a set of distances between atoms.

    Args:
        bins (int): The number of bins in the histogram.
        min (float): The minimum value of the histogram range.
        max (float): The maximum value of the histogram range.
        sigma (float): The standard deviation of the Gaussian.
        ff (ForceField): The forcefield object.
        device (str): The device on which the computations will be performed.

    Attributes:
        device (str): The device on which the computations will be performed.
        bins (int): The number of bins in the histogram.
        min (float): The minimum value of the histogram range.
        max (float): The maximum value of the histogram range.
        x (list): The values of the bins in the histogram.
        sigma (float): The standard deviation of the Gaussian.
        delta (float): The distance between adjacent bins.
        centers (Tensor): The centers of the bins.
        n_classes (int): The number of bond classes.
        names (list): The names of the bond classes.
    """
    def __init__(self, bins, min, max, sigma, ff, device):
        super(GaussianHistogram_Dis, self).__init__()
        self.device = device

        self.bins = bins
        self.min = min
        self.max = max
        self.x = [self.min + h * (self.max - self.min) / self.bins for h in range(0, self.bins)]

        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers[None, None, :].to(device=device)

        self.n_classes = len(ff.bond_params())
        self.names = [str(t.name) for t in ff.bond_types.values()]

        #self.types_one_hot = types_one_hot[:,:, None, :] #(BS,N_bond, 1, N_types)

    def forward(self, atoms, indices):
        """
         Computes the Gaussian histogram for a set of distances between atoms.

         Args:
             atoms (list): The list of atom positions.
             indices (Tensor): A tensor of shape (batch_size, num_bonds, 3) that contains the indices of the atoms involved in the bonds.

         Returns:
             Tensor: A tensor of shape (num_bins, num_types) containing the histogram.
         """

        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_classes)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        x = dis[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        #print(x.type())
        #print(param_one_hot.type())
        x = x[:, :, :, None] * param_one_hot[:, :, None, :] #(BS, N_bonds, N_bins, N_types)
        x = x.sum(dim=(0, 1)) + 1E-40
        x = x / (x.sum(dim=0, keepdim=True) + 1E-20)
        return x

class GaussianHistogram_LJ(nn.Module):
    def __init__(self, bins, min, max, sigma, ff, device):
        super(GaussianHistogram_LJ, self).__init__()
        self.device = device

        self.bins = bins
        self.min = min
        self.max = max
        self.x = [self.min + h * (self.max - self.min) / self.bins for h in range(0, self.bins)]

        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers[None, None, :].to(device=device)

        self.n_classes = len(ff.lj_params())
        self.names = [str(t.name) for t in ff.lj_types.values()]

        #self.types_one_hot = types_one_hot[:,:, None, :] #(BS,N_bond, 1, N_types)

    def forward(self, atoms, indices):

        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_classes)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        """
        nan3 = torch.sum(torch.isnan(dis))
        if nan3 > 0:
            print("NANs in dis!!!!")
        else:
            print("keine NANE in dis")
        """


        x = dis[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        #print(x.type())
        #print(param_one_hot.type())
        x = x[:, :, :, None] * param_one_hot[:, :, None, :] #(BS, N_bonds, N_bins, N_types)
        x = x.sum(dim=(0, 1)) + 1E-40
        x = x / (x.sum(dim=0, keepdim=True) + 1E-20)
        return x


class GaussianHistogram_Angle(nn.Module):
    def __init__(self, bins, min, max, sigma, ff, device):
        super(GaussianHistogram_Angle, self).__init__()
        self.device = device
        self.bins = bins
        self.min = min
        self.max = max
        self.x = [self.min + h * (self.max - self.min) / self.bins for h in range(0, self.bins)]

        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers[None, None, :].to(device=device)

        self.n_classes = len(ff.angle_params())
        self.names = [str(t.name) for t in ff.angle_types.values()]

        #self.types_one_hot = types_one_hot[:,:, None, :] #(BS,N_bond, 1, N_types)

    def forward(self, atoms, indices):

        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_classes)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

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

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)*180.0/np.pi

        x = a[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        #print(x.type())
        #print(param_one_hot.type())
        x = x[:, :, :, None] * param_one_hot[:, :, None, :] #(BS, N_bonds, N_bins, N_types)
        x = x.sum(dim=(0, 1)) + 1E-40
        x = x / (x.sum(dim=0, keepdim=True) + 1E-20)
        return x

class GaussianHistogram_Dih(nn.Module):
    def __init__(self, bins, min, max, sigma, ff, device):
        super(GaussianHistogram_Dih, self).__init__()
        self.device = device

        self.bins = bins
        self.min = min
        self.max = max
        self.x = [self.min + h * (self.max - self.min) / self.bins for h in range(0, self.bins)]
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers[None, None, :].to(device=device)

        self.n_classes = len(ff.dih_params())
        self.names = [str(t.name) for t in ff.dih_types.values()]

        #self.types_one_hot = types_one_hot[:,:, None, :] #(BS,N_bond, 1, N_types)

    def forward(self, atoms, indices):

        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_classes)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

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

        norm = norm1 * norm2 #+ 1E-20
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)

        a = torch.acos(a)*180.0/np.pi

        x = a[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        #print(x.type())
        #print(param_one_hot.type())
        x = x[:, :, :, None] * param_one_hot[:, :, None, :] #(BS, N_bonds, N_bins, N_types)
        x = x.sum(dim=(0, 1)) + 1E-40
        x = x / (x.sum(dim=0, keepdim=True) + 1E-20)
        return x