import torch
from dbm.ff import *
from scipy import constants
import sys
import numpy as np
import torch.nn as nn
import math

class GaussianHistogram(nn.Module):
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
        x = x[:, :, None] - self.centers
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=(0, 1))
        x = x / x.sum()
        return x

class GaussianHistogram_Dis(nn.Module):
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


class Energy_torch():

    def __init__(self, ff, device):
        self.ff = ff
        self.device=device
        self.bond_params = torch.tensor(self.ff.bond_params(), dtype=torch.float32, device=device)
        self.bond_threshold_params = torch.tensor(self.ff.bond_threshold_params(), dtype=torch.float32, device=device)
        self.angle_params = torch.tensor(self.ff.angle_params(), dtype=torch.float32, device=device)
        self.angle_threshold_params = torch.tensor(self.ff.angle_threshold_params(), dtype=torch.float32, device=device)
        self.dih_params = torch.tensor(self.ff.dih_params(), dtype=torch.float32, device=device)
        self.dih_rb_params = torch.tensor(self.ff.dih_rb_params(), dtype=torch.float32, device=device)
        self.lj_params = torch.tensor(self.ff.lj_params(), dtype=torch.float32, device=device)
        self.atom_mass = torch.tensor([[[atype.mass for atype in self.ff.atom_types.values()]]], dtype=torch.float32, device=device) #(1, 1, n_atomtypes)

        self.one = torch.tensor(1, dtype=torch.int32, device=device)
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
        #converts from kJ/mol to J
        return energy * 1000.0 / self.avogadro_const

    def bond(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - a_0
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)

        return en

    def bond_test(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - 0.46
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)

        return en

    def angle_test(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
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

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)

        en1 = f_c/2.0*(a - 70.0 * math.pi / 180.)**2
        en2 = f_c/6.0*(a - 100.0 * math.pi / 180.)**2


        #en = en**2
        #en = en * f_c
        #en = en / 2.0
        en = torch.sum(en1+en2, dim=1)
        return en

    def bond_threshold_old(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
        en = torch.where(dis > a_0, dis - a_0, torch.zeros([], dtype=torch.float32, device=self.device))

        #en = dis - a_0
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)

        return en

    def bond_threshold(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_threshold_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        min = param[:, :, 0]
        fc_min = param[:, :, 1]
        max = param[:, :, 2]
        fc_max = param[:, :, 3]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
        en1 = torch.where(dis > max, dis - max, torch.zeros([], dtype=torch.float32, device=self.device))
        en1 = en1**2
        en1 = en1 * fc_min

        en2 = torch.where(dis < min, dis - min, torch.zeros([], dtype=torch.float32, device=self.device))
        en2 = en2**2
        en2 = en2 * fc_max

        en = torch.sum(en1+en2, 1)

        return en

    def angle_threshold_old(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
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

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)

        en = torch.where(a < a_0, a - a_0, torch.zeros([], dtype=torch.float32, device=self.device))


        en = f_c/2.0*en**2
        #en = en**2
        #en = en * f_c
        #en = en / 2.0
        en = torch.sum(en, dim=1)
        return en

    def angle_threshold(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

        param = self.angle_threshold_params[param_ndx]
        min = param[:, :, 0]
        fc_min = param[:, :, 1]
        max = param[:, :, 2]
        fc_max = param[:, :, 3]

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

        a = torch.acos(a)

        en1 = torch.where(a < min, a - min, torch.zeros([], dtype=torch.float32, device=self.device))
        en1 = fc_min*en1**2

        en2 = torch.where(a > max, a - max, torch.zeros([], dtype=torch.float32, device=self.device))

        en2 = fc_max*en2**2


        en = torch.sum(en1+en2, dim=1)

        return en

    def variance(self, quantity, param_one_hot):
        avg = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(quantity, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device))
        mean = torch.mean(avg, 0)[None, None, :] #(1, 1, n_classes)
        var = torch.where(param_one_hot > 0,
                                (mean - quantity)**2 / torch.sum(param_one_hot, 1, keepdims=True),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_bonds, n_classes)
        var = torch.sum(var, 1)
        var = torch.mean(var, 0)
        return var

    def batch_moments(self, measure, param_one_hot):
        measure = measure[:,:,None] * param_one_hot #(BS, n_atoms, 1) * (BS, N_bonds, N_classes)
        measure_sq = measure**2

        mean = torch.where(torch.sum(param_one_hot, (0,1)) > 0,
                                torch.sum(measure, (0,1)) / torch.sum(param_one_hot, (0,1)),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)
        #mean = torch.mean(mean, 0) # (N_classes)

        mean_sq = torch.where(torch.sum(param_one_hot, (0,1)) > 0,
                                torch.sum(measure_sq, (0,1)) / torch.sum(param_one_hot, (0,1)),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)
        #mean_sq = torch.mean(mean_sq, 0) # (N_classes)

        return mean, mean_sq

    def get_quadratic_repl(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        #exp_n = param[:, :, 2]
        #exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        en = torch.where(dis < sigma, epsilon* (1 - dis / sigma)**2, torch.zeros([], dtype=torch.float32, device=self.device))

        return en

    def dis_mse(self, coords1, coords2, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        dis1 = self.get_dis(coords1, ndx1, ndx2)
        dis2 = self.get_dis(coords2, ndx1, ndx2)

        en = dis1 - dis2
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)
        return en

    def angle_mse(self, coords1, coords2, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        angle1 = self.get_angle(coords1, ndx1, ndx2, ndx3)
        angle2 = self.get_angle(coords2, ndx1, ndx2, ndx3)

        en = angle1 - angle2
        en = en ** 2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)
        return en

    def dis_nb(self, coords, ndx1, ndx2):
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, coords)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, coords)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))
        return dis

    def nb_mse(self, coords1, coords2, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]

        dis1 = self.dis_nb(coords1, ndx1, ndx2)
        dis2 = self.dis_nb(coords2, ndx1, ndx2)

        dis1 = torch.where(dis1 < sigma, dis1, sigma)
        dis2 = torch.where(dis2 < sigma, dis2, sigma)

        en = dis1 - dis2
        en = en**2
        en = en * epsilon / 2.0
        en = torch.sum(en, 1)

        return en

    def repl_mse(self, coords1, coords2, indices):
        repl1 = self.get_quadratic_repl(coords1, indices)
        repl2 = self.get_quadratic_repl(coords2, indices)

        en = repl1 - repl2
        en = en ** 2
        en = torch.sum(en, 1)

        return en

    def get_dis(self, coords, ndx1, ndx2):
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, coords)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, coords)])
        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis) #(BS, n_bonds)
        return dis

    def get_angle(self, coords, ndx1, ndx2, ndx3):

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, coords)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, coords)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, coords)])
        vec1 = pos1 - pos2
        vec2 = pos3 - pos2
        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2 + 1E-20
        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)
        angle = dot / norm
        angle = torch.clamp(angle, -0.9999, 0.9999)
        angle = torch.acos(angle) * 180.0 / self.pi

        return angle

    def get_dih(self, coords, ndx1, ndx2, ndx3, ndx4):
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, coords)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, coords)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, coords)])
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, coords)])
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
        dih = dot / norm
        dih = torch.clamp(dih, -0.9999, 0.9999)
        dih = torch.acos(dih) * 180.0 / self.pi
        return dih

    def bond_moment_loss(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_one_hot = indices[:, :, 0] + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_bond_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        dis_a = self.get_dis(atoms_a, ndx1, ndx2) #(BS, n_atoms)
        dis_b = self.get_dis(atoms_b, ndx1, ndx2)

        mean_a, mean_sq_a = self.batch_moments(dis_a, param_one_hot)
        mean_b, mean_sq_b = self.batch_moments(dis_b, param_one_hot)


        #print("bond")
        #print(mean_a)
        #mean_loss = (mean_a - mean_b)**2 * self.bond_params[:-1, 1]
        mean_loss = torch.where(mean_a > 0,
                                torch.abs(mean_a - mean_b)/mean_a * self.bond_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)
        mean_sq_loss = torch.where(mean_sq_a > 0,
                                torch.abs(mean_sq_a - mean_sq_b)/mean_sq_a * self.bond_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)
        #mean_sq_loss = torch.abs(mean_sq_a - mean_sq_b) * self.bond_params[:-1, 1]

        return mean_loss, mean_sq_loss

    def angle_moment_loss(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]

        param_one_hot = indices[:, :, 0] + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_angle_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        angle_a = self.get_angle(atoms_a, ndx1, ndx2, ndx3) #(BS, n_atoms)
        angle_b = self.get_angle(atoms_b, ndx1, ndx2, ndx3)

        mean_a, mean_sq_a = self.batch_moments(angle_a, param_one_hot)
        mean_b, mean_sq_b = self.batch_moments(angle_b, param_one_hot)

        #mean_loss = (mean_a - mean_b)**2 * self.angle_params[:-1, 1]
        mean_loss = torch.where(mean_a > 0,
                                torch.abs(mean_a - mean_b)/mean_a * self.angle_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)

        mean_sq_loss = torch.where(mean_sq_a > 0,
                                torch.abs(mean_sq_a - mean_sq_b)/mean_sq_a * self.angle_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)
        #mean_sq_loss = torch.abs(mean_sq_a - mean_sq_b) * self.angle_params[:-1, 1]

        return mean_loss, mean_sq_loss

    def dih_moment_loss(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]

        param_one_hot = indices[:, :, 0] + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_dih_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        dih_a = self.get_dih(atoms_a, ndx1, ndx2, ndx3, ndx4) #(BS, n_atoms)
        dih_b = self.get_dih(atoms_b, ndx1, ndx2, ndx3, ndx4)

        mean_a, mean_sq_a = self.batch_moments(dih_a, param_one_hot)
        mean_b, mean_sq_b = self.batch_moments(dih_b, param_one_hot)

        #mean_loss = (mean_a - mean_b)**2 * self.dih_params[:-1, 1]
        #mean_sq_loss = torch.abs(mean_sq_a - mean_sq_b) * self.dih_params[:-1, 1]
        mean_loss = torch.where(mean_a > 0,
                                torch.abs(mean_a - mean_b)/mean_a * self.dih_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)

        mean_sq_loss = torch.where(mean_sq_a > 0,
                                torch.abs(mean_sq_a - mean_sq_b)/mean_sq_a * self.dih_params[:-1, 1],
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS,N_classes)

        return mean_loss, mean_sq_loss

    def quadratic_repl(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        #exp_n = param[:, :, 2]
        #exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        en = torch.where(dis < sigma, epsilon* (1 - dis / sigma)**2, torch.zeros([], dtype=torch.float32, device=self.device))

        en = torch.sum(en, dim=1)

        return en

    def min_dist(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        #exp_n = param[:, :, 2]
        #exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))
        dis = torch.where(epsilon > 0.0, dis, torch.ones([], dtype=torch.float32, device=self.device)*1E20)

        min_dist, _ = torch.min(dis, 1)

        return min_dist

    def avg_bond(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_one_hot = indices[:, :, 0] + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_bond_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        dis_a = self.dis(atoms_a, ndx1, ndx2) #(BS, n_atoms)
        dis_b = self.dis(atoms_b, ndx1, ndx2)

        mean_a, mean_sq_a = self.batch_moments(dis_a, param_one_hot)
        mean_b, mean_sq_b = self.batch_moments(dis_b, param_one_hot)

        pos1_a = torch.stack([a[n] for n, a in zip(ndx1, atoms_a)])
        pos2_a = torch.stack([a[n] for n, a in zip(ndx2, atoms_a)])
        dis_a = pos1_a - pos2_a
        dis_a = dis_a**2
        dis_a = torch.sum(dis_a, 2)
        dis_a = torch.sqrt(dis_a) #(BS, n_bonds)
        dis_a = dis_a[:,:,None] * param_one_hot
        #dis_a = torch.sum(dis_a, 1) #(BS, n_classes)
        avg_dis_a = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(dis_a, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        pos1_b = torch.stack([a[n] for n, a in zip(ndx1, atoms_b)])
        pos2_b = torch.stack([a[n] for n, a in zip(ndx2, atoms_b)])
        dis_b = pos1_b - pos2_b
        dis_b = dis_b**2
        dis_b = torch.sum(dis_b, 2)
        dis_b = torch.sqrt(dis_b) #(BS, n_bonds)
        dis_b = dis_b[:,:,None] * param_one_hot
        #dis_b = torch.sum(dis_b, 1) #(BS, n_classes)
        #avg_dis_b = torch.sum(dis_b, 1) / torch.sum(param_one_hot, 1) #(BS, n_classes)
        avg_dis_b = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(dis_b, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        var_a = self.variance(dis_a, param_one_hot)
        var_b = self.variance(dis_b, param_one_hot)
        en_var = torch.sum(torch.abs(var_a - var_b) * self.bond_params[:-1, 1])

        en = (avg_dis_a - avg_dis_b)**2
        en = en * self.bond_params[None, :-1, 1]
        en = torch.sum(en, 1)
        en = torch.mean(en)
        return en, en_var

    def avg_angle(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

        param_one_hot = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_angle_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        pos1_a = torch.stack([a[n] for n, a in zip(ndx1, atoms_a)])
        pos2_a = torch.stack([a[n] for n, a in zip(ndx2, atoms_a)])
        pos3_a = torch.stack([a[n] for n, a in zip(ndx3, atoms_a)])
        vec1_a = pos1_a - pos2_a
        vec2_a = pos3_a - pos2_a
        norm1_a = vec1_a**2
        norm1_a = torch.sum(norm1_a, dim=2)
        norm1_a = torch.sqrt(norm1_a)
        norm2_a = vec2_a**2
        norm2_a = torch.sum(norm2_a, dim=2)
        norm2_a = torch.sqrt(norm2_a)
        norm_a = norm1_a * norm2_a
        dot_a = vec1_a * vec2_a
        dot_a = torch.sum(dot_a, dim=2)
        angle_a = dot_a / norm_a
        angle_a = torch.clamp(angle_a, -0.9999, 0.9999)
        angle_a = torch.acos(angle_a) * 180.0 / self.pi
        angle_a = angle_a[:,:,None] * param_one_hot
        avg_angle_a = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(angle_a, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        pos1_b = torch.stack([a[n] for n, a in zip(ndx1, atoms_b)])
        pos2_b = torch.stack([a[n] for n, a in zip(ndx2, atoms_b)])
        pos3_b = torch.stack([a[n] for n, a in zip(ndx3, atoms_b)])
        vec1_b = pos1_b - pos2_b
        vec2_b = pos3_b - pos2_b
        norm1_b = vec1_b**2
        norm1_b = torch.sum(norm1_b, dim=2)
        norm1_b = torch.sqrt(norm1_b)
        norm2_b = vec2_b**2
        norm2_b = torch.sum(norm2_b, dim=2)
        norm2_b = torch.sqrt(norm2_b)
        norm_b = norm1_b * norm2_b
        dot_b = vec1_b * vec2_b
        dot_b = torch.sum(dot_b, dim=2)
        angle_b = dot_b / (norm_b + 1E-20)
        angle_b = torch.clamp(angle_b, -0.9999, 0.9999)
        angle_b = torch.acos(angle_b) * 180.0 / self.pi
        angle_b = angle_b[:,:,None] * param_one_hot
        avg_angle_b = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(angle_b, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        var_a = self.variance(angle_a, param_one_hot)
        var_b = self.variance(angle_b, param_one_hot)
        en_var = torch.sum(torch.abs(var_a - var_b) * self.angle_params[:-1, 1])

        en = (avg_angle_a - avg_angle_b)**2
        en = en * self.angle_params[None, :-1, 1]
        en = torch.sum(en, 1)
        en = torch.mean(en)
        return en, en_var

    def avg_dih(self, atoms_a, atoms_b, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param_one_hot = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_one_hot, num_classes=self.n_dih_class)[:,:,1:] #(BS, N_bonds, N_classes)
        param_one_hot = param_one_hot.type(torch.FloatTensor).to(device=self.device)

        pos1_a = torch.stack([a[n] for n, a in zip(ndx1, atoms_a)])
        pos2_a = torch.stack([a[n] for n, a in zip(ndx2, atoms_a)])
        pos3_a = torch.stack([a[n] for n, a in zip(ndx3, atoms_a)])
        pos4_a = torch.stack([a[n] for n, a in zip(ndx4, atoms_a)])
        vec1_a = pos2_a - pos1_a
        vec2_a = pos2_a - pos3_a
        vec3_a = pos4_a - pos3_a
        plane1_a = torch.cross(vec1_a, vec2_a)
        plane2_a = torch.cross(vec2_a, vec3_a)
        norm1_a = plane1_a**2
        norm1_a = torch.sum(norm1_a, dim=2)
        norm1_a = torch.sqrt(norm1_a)
        norm2_a = plane2_a**2
        norm2_a = torch.sum(norm2_a, dim=2)
        norm2_a = torch.sqrt(norm2_a)
        dot_a = plane1_a * plane2_a
        dot_a = torch.sum(dot_a, dim=2)
        norm_a = norm1_a * norm2_a + 1E-20
        dih_a = dot_a / norm_a
        dih_a = torch.clamp(dih_a, -0.9999, 0.9999)
        dih_a = torch.acos(dih_a) * 180.0 / self.pi
        dih_a = dih_a[:,:,None] * param_one_hot
        avg_dih_a = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(dih_a, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        pos1_b = torch.stack([a[n] for n, a in zip(ndx1, atoms_b)])
        pos2_b = torch.stack([a[n] for n, a in zip(ndx2, atoms_b)])
        pos3_b = torch.stack([a[n] for n, a in zip(ndx3, atoms_b)])
        pos4_b = torch.stack([a[n] for n, a in zip(ndx4, atoms_b)])
        vec1_b = pos2_b - pos1_b
        vec2_b = pos2_b - pos3_b
        vec3_b = pos4_b - pos3_b
        plane1_b = torch.cross(vec1_b, vec2_b)
        plane2_b = torch.cross(vec2_b, vec3_b)
        norm1_b = plane1_b**2
        norm1_b = torch.sum(norm1_b, dim=2)
        norm1_b = torch.sqrt(norm1_b)
        norm2_b = plane2_b**2
        norm2_b = torch.sum(norm2_b, dim=2)
        norm2_b = torch.sqrt(norm2_b)
        dot_b = plane1_b * plane2_b
        dot_b = torch.sum(dot_b, dim=2)
        norm_b = norm1_b * norm2_b + 1E-20
        dih_b = dot_b / norm_b
        dih_b = torch.clamp(dih_b, -0.9999, 0.9999)
        dih_b = torch.acos(dih_b) * 180.0 / self.pi
        dih_b = dih_b[:,:,None] * param_one_hot
        avg_dih_b = torch.where(torch.sum(param_one_hot, 1) > 0,
                                torch.sum(dih_b, 1) / torch.sum(param_one_hot, 1),
                                torch.zeros([], dtype=torch.float32, device=self.device)) #(BS, n_classes)

        var_a = self.variance(dih_a, param_one_hot)
        var_b = self.variance(dih_b, param_one_hot)
        en_var = torch.sum(torch.abs(var_a - var_b) * self.dih_params[:-1, 1])

        en = (avg_dih_a - avg_dih_b)**2
        en = en * self.dih_params[None, :-1, 1]
        en = torch.sum(en, 1)
        en = torch.mean(en)
        return en, en_var


    def bond_dstr(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = dis / torch.sum(dis, 1, keepdim=True)

        #dis.type(torch.FloatTensor)
        return dis

    def angle_dstr(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
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

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)*180.0/np.pi
        #a = a / torch.sum(a, 1, keepdim=True)
        #a.type(torch.FloatTensor)

        return a

    def angle(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
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

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2
        #en = en**2
        #en = en * f_c
        #en = en / 2.0
        en = torch.sum(en, dim=1)
        return en


    def dih_dstr(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]
        func_type = param[:, :, 2].type(torch.int32)
        mult = param[:, :, 3]



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
        #a = torch.clamp(a, -1.0, 1.0)

        a = torch.acos(a)*180.0/np.pi

        #a = torch.where(func_type == 1, 3*a, a)

        #a = a / torch.sum(a, 1, keepdim=True)
        #a.type(torch.FloatTensor)

        #param_ndx = indices[:, :, 0]
        #param_ndx = torch.where(param_ndx >= 0, param_ndx, torch.max(param_ndx)+1)
        #param_ndx = param_ndx - torch.min(param_ndx)
        #param_ndx = param_ndx + 1
        #param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_bond_class)[:,:,1:] #(BS, N_bonds, N_classes)

        return a

    def dih(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]
        func_type = param[:, :, 2].type(torch.int32)
        mult = param[:, :, 3]



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
        #a = torch.clamp(a, -1.0, 1.0)

        a = torch.acos(a)

        a = torch.where(func_type == 1, 3*a, a)

        en = a - a_0

        en = torch.where(func_type == 1, (torch.cos(en)+ 1.0 ) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)
        return en



    def dih_rb(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
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
        #a = torch.clamp(a, -1.0, 1.0)

        #a = torch.acos(a)
        #a = a - self.pi

        #a = torch.where(func_type == 1, 3*a, a)


        en = f1 - f2*a + f3*torch.pow(a,2) - f4*torch.pow(a,3) + f5*torch.pow(a,4)


        #en = torch.where(func_type == 1, (torch.cos(en)+ 1.0 ) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)
        return en

    def exp_repl(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        exp_n = param[:, :, 2]
        exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        exp = torch.exp(-dis / sigma)
        #c_n = torch.pow(sigma / dis, exp_n)
        #c_m = torch.pow(sigma / dis, exp_m)

        #c6 = torch.pow(sigma / dis, 6)
        #c12 = torch.pow(c6, 2)

        #en = 4 * epsilon * (c12 - c6)
        en = epsilon * exp

        #cutoff
        #c6_cut = sigma
        #c6_cut = torch.pow(c6_cut, 6)
        #c12_cut = torch.pow(c6_cut, 2)
        #en_cut = 4 * epsilon * (c12_cut - c6_cut)
        #en = en - en_cut
        #en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        return en

    def lj(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]
        exp_n = param[:, :, 2]
        exp_m = param[:, :, 3]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        c_n = torch.pow(sigma / dis, exp_n)
        c_m = torch.pow(sigma / dis, exp_m)

        #c6 = torch.pow(sigma / dis, 6)
        #c12 = torch.pow(c6, 2)

        #en = 4 * epsilon * (c12 - c6)
        en = 4 * epsilon * (c_n - c_m)

        #cutoff
        #c6_cut = sigma
        #c6_cut = torch.pow(c6_cut, 6)
        #c12_cut = torch.pow(c6_cut, 2)
        #en_cut = 4 * epsilon * (c12_cut - c6_cut)
        #en = en - en_cut
        #en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        return en

    def bond_grid(self, pos_grid, atoms, indices):
        #ndx1 = indices[:, :, 1] # (BS, n_ljs)

        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        #print(param_ndx)

        param = self.bond_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]

        dis = torch.sqrt(torch.sum((pos_grid - pos2)**2, dim=2))

        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - a_0
        en = en**2

        en = en * f_c / 2.0

        en = torch.sum(en, dim=1)
        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum



        return en


    def angle_grid1(self, pos_grid, atoms, indices):
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]

        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])[:,:,:, None, None, None]

        #print(pos2)
        vec1 = pos_grid - pos2
        vec2 = pos3 - pos2

        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2 + 1E-12

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2

        en = torch.sum(en, dim=1)

        #en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        #en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        #en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        #en = en / en_sum



        return en

    def angle_grid2(self, pos_grid, atoms, indices):
        ndx1 = indices[:, :, 1]
        ndx3 = indices[:, :, 3]

        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])[:,:,:, None, None, None]
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])[:,:,:, None, None, None]

        #print(pos2)
        vec1 = pos1 - pos_grid
        vec2 = pos3 - pos_grid

        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        a = dot / norm + 1E-12
        a = torch.clamp(a, -0.9999, 0.9999)

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2

        en = torch.sum(en, dim=1)

        #en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        #en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        #en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        #en = en / en_sum


        return en

    def energy_to_prop(self, en):
        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum

        return en

    def dih_grid(self, pos_grid, atoms, indices):
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]

        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]
        func_type = param[:, :, 2].type(torch.int32)[:,:, None, None, None]
        mult = param[:, :, 3]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])[:,:,:, None, None, None]
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])[:,:,:, None, None, None]

        vec1 = pos2 - pos_grid
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3

        plane1 = torch.cross(vec1, vec2.repeat(1,1,1,16,16,16))
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

        a = torch.acos(a)

        a = torch.where(func_type == 1, 3*a, a)

        en = a - a_0

        en = torch.where(func_type == 1, (torch.cos(en)+ 1.0 ) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)

        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        #print(en)
        en = en / en_sum
        #print(en)

        return en


    def lj_grid(self, pos_grid, atoms, indices):
        #ndx1 = indices[:, :, 1] # (BS, n_ljs)

        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0, None, None, None]
        epsilon = param[:, :, 1, None, None, None]

        # pos_grid (1, 1, 3, N_x, N_y, N_z)

        #pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]


        dis = torch.sqrt(torch.sum((pos_grid - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        c6 = torch.pow(sigma / dis, 6)
        c12 = torch.pow(c6, 2)

        en = 4 * epsilon * (c12 - c6)

        #cutoff
        #c6_cut = sigma
        #c6_cut = torch.pow(c6_cut, 6)
        #c12_cut = torch.pow(c6_cut, 2)
        #en_cut = 4 * epsilon * (c12_cut - c6_cut)
        #en = en - en_cut
        #en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum


        return en

    """
    def bond_dstr(self, atoms, indices, n_bins=20, bin_width=0.01, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        #param_ndx = torch.where(param_ndx >= 0, param_ndx, torch.max(param_ndx)+1)
        #param_ndx = param_ndx - torch.min(param_ndx)
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_bond_class)[:,:,1:] #(BS, N_bonds, N_classes)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)[:,:, None] #(BS, N_bonds, 1)
        histo = (gauss_centers - dis)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo
    """
    """
    def lj_dstr(self, atoms, indices, n_bins=20, bin_width=0.025, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        #param_ndx = torch.where(param_ndx >= 0, param_ndx, torch.max(param_ndx)+1)
        #param_ndx = param_ndx - torch.min(param_ndx)
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_lj_class)[:,:,1:] #(BS, N_bonds, N_classes)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)[:,:, None] #(BS, N_bonds, 1)
        histo = (gauss_centers - dis)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)
        return histo
    """
    """
    def angle_dstr(self, atoms, indices, n_bins=20, bin_width=9.0, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
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
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        a = torch.acos(a)[:,:,None]
        a = a * 180. / np.pi
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_angle_class)[:,:,1:] #(BS, N_bonds, N_classes)

        histo = (gauss_centers - a)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo
    """
    """
    def dih_dstr(self, atoms, indices, n_bins=20, bin_width=9.0, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
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
        a = torch.acos(a)[:,:,None]
        a = a * 180. / np.pi


        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_dih_class)[:,:,1:] #(BS, N_bonds, N_classes)

        histo = (gauss_centers - a)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo
    """