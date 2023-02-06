import torch
from dbm.ff_tab import *
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


class _TabBondEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dict):
        #np_input = round(float(input.detach().cpu().numpy()), 3)
        np_input = round(input.item(), 3)
        energy, force = dict[np_input]
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        (neg_force,) = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None

_evaluate_bond_energy = _TabBondEnergyWrapper.apply

class _TabAngleEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dict):
        #np_input = round(float(input.detach().cpu().numpy()), 0)
        np_input = round(input.item(), 0)
        energy, force = dict[np_input]
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        (neg_force,) = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None

_evaluate_angle_energy = _TabAngleEnergyWrapper.apply

class _TabLJEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dict):
        #np_input = round(float(input.detach().cpu().numpy()), 2)
        np_input = round(input.item(), 2)
        energy, force = dict[np_input]
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        (neg_force,) = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None

_evaluate_lj_energy = _TabLJEnergyWrapper.apply

class Energy_torch():

    def __init__(self, ff, device):
        self.ff = ff
        self.device=device

        self.bond_dict = self.make_list_of_torch_dicts(self.ff.bond_params())
        self.n_bond_dicts = len(self.bond_dict)-1
        self.angle_dict = self.make_list_of_torch_dicts(self.ff.angle_params())
        self.n_angle_dicts = len(self.angle_dict)-1
        self.dih_dict = self.make_list_of_torch_dicts(self.ff.dih_params())
        self.n_dih_dicts = len(self.dih_dict)-1
        self.lj_dict = self.make_list_of_torch_dicts(self.ff.lj_params())
        self.n_lj_dicts = len(self.lj_dict)-1

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

    def make_list_of_torch_dicts(self, list_of_numpy_dicts):
        dict_list = []
        for dict in list_of_numpy_dicts:
            d = self.make_torch_dict(dict)
            dict_list.append(d)
        return np.array(dict_list)

    def make_torch_dict(self, numpy_dict):
        d = {}
        for key, value in numpy_dict.items():
            v = torch.tensor(value, dtype=torch.float32, device=self.device)
            d[key] = v
        return d

    def convert_to_joule(self, energy):
        #converts from kJ/mol to J
        return energy * 1000.0 / self.avogadro_const

    def bond(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        e = torch.zeros((1,), dtype=torch.float32, device=self.device)
        for dis_batch, param_ndx_batch in zip(dis, param_ndx):
            for d, ndx in zip(dis_batch, param_ndx_batch):
                if ndx > 0:
                    e += _evaluate_bond_energy(d, self.bond_dict[ndx])

        return e

    def angle(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

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
        norm = norm1 * norm2 + 1E-20

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)* 180.0 / self.pi

        e = torch.zeros((1,), dtype=torch.float32, device=self.device)
        for angle_batch, param_ndx_batch in zip(a, param_ndx):
            for angle, ndx in zip(angle_batch, param_ndx_batch):
                if ndx > 0:
                    e += _evaluate_angle_energy(angle, self.angle_dict[ndx])

        return e


    def dih(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

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

        a = torch.acos(a)* 180.0 / self.pi

        e = torch.zeros((1,), dtype=torch.float32, device=self.device)
        for angle_batch, param_ndx_batch in zip(a, param_ndx):
            for angle, ndx in zip(angle_batch, param_ndx_batch):
                if ndx > 0:
                    e += _evaluate_angle_energy(angle, self.dih_dict[ndx])

        return e




    def lj(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        energies = []
        for dis_batch, param_ndx_batch in zip(dis, param_ndx):
            e = torch.zeros((1,), dtype=torch.float32, device=self.device)
            for d, ndx in zip(dis_batch, param_ndx_batch):
                if ndx > 0:
                    e += _evaluate_lj_energy(d, self.lj_dict[ndx])
            energies.append(e)
        energies = torch.stack(energies)
        print(energies.size())
        return energies
