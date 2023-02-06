import torch
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from dbm.util import make_grid_np, rand_rot_mtx, rot_mtx_batch, voxelize_gauss, make_dir, avg_blob, voxelize_gauss_batch
from dbm.torch_energy import *
from dbm.output import *
from dbm.recurrent_generator import Recurrent_Generator
from tqdm import tqdm
import numpy as np
# from tqdm import tqdm
from timeit import default_timer as timer
import os
import math
# from configparser import ConfigParser
# import mdtraj as md
# from universe import *
import dbm.model as model
from dbm.data import *
from dbm.stats import *
from scipy import constants
# import dbm.tf_utils as tf_utils
# import dbm.tf_energy as tf_energy
from copy import deepcopy
from shutil import copyfile
from contextlib import redirect_stdout
from operator import add
from itertools import cycle
import gc
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution()

torch.set_default_dtype(torch.float32)


class DS_seq(Dataset):
    def __init__(self, data, cfg, train=True):

        self.data = data

        generators = []
        #if cfg.getboolean('training', 'init_intra'):
        #    generators.append(
        #        Recurrent_Generator(data, hydrogens=False, gibbs=False, train=train, rand_rot=False, inter=False))
        generators.append(Recurrent_Generator(data, hydrogens=False, gibbs=False, train=train, rand_rot=False))
        generators.append(Recurrent_Generator(data, hydrogens=False, gibbs=True, train=train, rand_rot=False))

        if cfg.getboolean('training', 'hydrogens'):
            #if cfg.getboolean('training', 'init_intra'):
            #    generators.append(
            #        Recurrent_Generator(data, hydrogens=True, gibbs=False, train=train, rand_rot=False, inter=False))
            generators.append(Recurrent_Generator(data, hydrogens=True, gibbs=False, train=train, rand_rot=False))
            generators.append(Recurrent_Generator(data, hydrogens=True, gibbs=True, train=train, rand_rot=False))

        self.elems = []
        for g in generators:
            self.elems += g.all_elems()

        self.resolution = cfg.getint('grid', 'resolution')
        self.delta_s = cfg.getfloat('grid', 'length') / cfg.getint('grid', 'resolution')
        self.sigma = cfg.getfloat('grid', 'sigma')

        if cfg.getboolean('training', 'rand_rot'):
            self.rand_rot = True
            print("using random rotations during training...")
        else:
            self.rand_rot = False
        self.align = int(cfg.getboolean('universe', 'align'))

        self.grid = make_grid_np(self.delta_s, self.resolution)

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, ndx):
        if self.rand_rot:
            R = rand_rot_mtx(self.data.align)
        else:
            R = np.eye(3)

        d = self.elems[ndx]

        target_atom = voxelize_gauss(np.dot(d['target_pos'], R.T), self.sigma, self.grid)
        aa_coords = np.dot(d['aa_pos'], R.T).astype('float32')
        atom_grid = voxelize_gauss(aa_coords, self.sigma, self.grid)
        bead_grid = voxelize_gauss(np.dot(d['cg_pos'], R.T), self.sigma, self.grid)

        cg_features = d['cg_feat'][:, :, None, None, None] * bead_grid[:, None, :, :, :]
        # (N_beads, N_chn, 1, 1, 1) * (N_beads, 1, N_x, N_y, N_z)
        cg_features = np.sum(cg_features, 0)

        elems = (target_atom, d['target_type'], d['aa_feat'], d['repl'], d['mask'])
        initial = (atom_grid, cg_features, aa_coords)
        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])

        return elems, initial, energy_ndx

    def array(self, elems, dtype):
        return tuple(np.array(t, dtype=dtype) for t in elems)


class GAN_seq():

    def __init__(self, device, cfg):

        self.device = device
        self.cfg = cfg

        self.bs = self.cfg.getint('training', 'batchsize')

        # Data pipeline
        self.data = Data(cfg, save=False)
        ds_train = DS_seq(self.data, cfg)
        if len(ds_train) != 0:
            self.loader_train = DataLoader(
                ds_train,
                batch_size=self.bs,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
        else:
            self.loader_train = []
        self.steps_per_epoch = int(len(self.loader_train) / (self.cfg.getint('training', 'n_critic') + 1))
        print(len(self.loader_train), self.steps_per_epoch)
        self.ff = self.data.ff

        self.loader_val = []
        if len(ds_train) != 0:
            ds_val = DS_seq(self.data, cfg, train=False)
            if len(ds_val) != 0:
                self.loader_val = DataLoader(
                    ds_val,
                    batch_size=self.bs,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=0,
                )

            # self.loader_val = cycle(loader_val)

        self.n_gibbs = int(cfg.getint('validate', 'n_gibbs'))

        # model
        self.name = cfg.get('model', 'name')
        self.z_dim = int(cfg.getint('model', 'noise_dim'))
        self.n_atom_chns = self.ff.n_atom_chns
        self.z_and_label_dim = self.z_dim + self.n_atom_chns

        self.step = 0
        self.epoch = 0

        # Make Dirs for saving
        self.out = OutputHandler(
            self.name,
            self.cfg.getint('training', 'n_checkpoints'),
            self.cfg.get('model', 'output_dir'),
        )
        self.energy = Energy_torch(self.ff, self.device)
        self.n_bins = 32
        self.gauss_hist_bond = GaussianHistogram_Dis(bins=self.n_bins, min=0.0, max=self.cfg.getfloat('grid', 'length')/2.0, sigma=0.01, ff=self.ff, device=device)
        self.gauss_hist_angle = GaussianHistogram_Angle(bins=self.n_bins, min=0, max=180, sigma=2.0, ff=self.ff, device=device)
        self.gauss_hist_dih = GaussianHistogram_Dih(bins=self.n_bins, min=0, max=180, sigma=4.0, ff=self.ff, device=device)
        self.gauss_hist_nb = GaussianHistogram_LJ(bins=self.n_bins, min=0.0, max=self.cfg.getfloat('grid', 'length')*1.2, sigma=0.01, ff=self.ff, device=device)

        prior_weights = self.cfg.get('prior', 'weights')
        self.prior_weights = [float(v) for v in prior_weights.split(",")]
        prior_schedule = self.cfg.get('prior', 'schedule')
        self.prior_schedule = np.array([0] + [int(v) for v in prior_schedule.split(",")])

        # self.prior_weights = self.get_prior_weights()
        self.ratio_bonded_nonbonded = cfg.getfloat('prior', 'ratio_bonded_nonbonded')
        # self.lj_weight = cfg.getfloat('training', 'lj_weight')
        # self.covalent_weight = cfg.getfloat('training', 'covalent_weight')
        self.prior_mode = cfg.get('prior', 'mode')
        self.val_mode = self.cfg.get('validate', 'mode')
        self.val_bs = self.cfg.getint('validate', 'batchsize')

        try:
            self.nb_pot = self.cfg.get('prior', 'nb_pot')
        except:
            self.nb_pot = "lj"
        try:
            self.bond_pot = self.cfg.get('prior', 'bond_pot')
        except:
            self.bond_pot = "harmonic"
        try:
            self.angle_pot = self.cfg.get('prior', 'angle_pot')
        except:
            self.angle_pot = "harmonic"

        print(self.prior_mode)
        # self.w_prior = torch.tensor(self.prior_weights[self.step], dtype=torch.float32, device=device)

        # Model selection
        if cfg.get('model', 'model_type') == "tiny":
            print("Using tiny model")
            if cfg.getint('grid', 'resolution') == 8:
                self.critic = model.AtomCrit_tiny(in_channels=self.ff.n_channels + 1,
                                                  start_channels=self.cfg.getint('model', 'n_chns'),
                                                  fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                                  device=device)
                self.generator = model.AtomGen_tiny(z_dim=self.z_and_label_dim,
                                                    in_channels=self.ff.n_channels,
                                                    start_channels=self.cfg.getint('model', 'n_chns'),
                                                    fac=1,
                                                    sn=self.cfg.getint('model', 'sn_gen'),
                                                    device=device)
            else:
                self.critic = model.AtomCrit_tiny16(in_channels=self.ff.n_channels + 1,
                                                    start_channels=self.cfg.getint('model', 'n_chns'),
                                                    fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                                    device=device)
                self.generator = model.AtomGen_tiny16(z_dim=self.z_and_label_dim,
                                                      in_channels=self.ff.n_channels,
                                                      start_channels=self.cfg.getint('model', 'n_chns'),
                                                      fac=1,
                                                      sn=self.cfg.getint('model', 'sn_gen'),
                                                      device=device)
        elif cfg.get('model', 'model_type') == "big":
            print("Using big model")
            if cfg.getint('grid', 'resolution') == 8:
                self.critic = model.AtomCrit_tiny(in_channels=self.ff.n_channels + 1,
                                                  start_channels=self.cfg.getint('model', 'n_chns'),
                                                  fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                                  device=device)
                self.generator = model.AtomGen_tiny(z_dim=self.z_and_label_dim,
                                                    in_channels=self.ff.n_channels,
                                                    start_channels=self.cfg.getint('model', 'n_chns'),
                                                    fac=1,
                                                    sn=self.cfg.getint('model', 'sn_gen'),
                                                    device=device)
            else:
                raise Exception('big model not implemented for resolution of 16')
        else:
            print("Using regular model")
            if cfg.getint('grid', 'resolution') == 8:
                self.critic = model.AtomCrit(in_channels=self.ff.n_channels + 1,
                                             start_channels=self.cfg.getint('model', 'n_chns'),
                                             fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                             device=device)
                self.generator = model.AtomGen(z_dim=self.z_and_label_dim,
                                               in_channels=self.ff.n_channels,
                                               start_channels=self.cfg.getint('model', 'n_chns'),
                                               fac=1,
                                               sn=self.cfg.getint('model', 'sn_gen'),
                                               device=device)
            else:
                self.critic = model.AtomCrit16(in_channels=self.ff.n_channels + 1,
                                               start_channels=self.cfg.getint('model', 'n_chns'),
                                               fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                               device=device)
                self.generator = model.AtomGen16(z_dim=self.z_and_label_dim,
                                                 in_channels=self.ff.n_channels,
                                                 start_channels=self.cfg.getint('model', 'n_chns'),
                                                 fac=1,
                                                 sn=self.cfg.getint('model', 'sn_gen'),
                                                 device=device)

        self.use_gp = cfg.getboolean('model', 'gp')

        # self.mse = torch.nn.MSELoss()
        # self.kld = torch.nn.KLDivLoss(reduction="batchmean")

        self.critic.to(device=device)
        self.generator.to(device=device)
        # self.mse.to(device=device)

        self.opt_generator_pretrain = Adam(self.generator.parameters(), lr=0.00005, betas=(0, 0.9))
        self.opt_generator = Adam(self.generator.parameters(), lr=0.00005, betas=(0, 0.9))
        self.opt_critic = Adam(self.critic.parameters(), lr=0.0001, betas=(0, 0.9))

        self.restored_model = False
        self.restore_latest_checkpoint()

    def prior_weight(self):
        try:
            ndx = next(x[0] for x in enumerate(self.prior_schedule) if x[1] > self.epoch) - 1
        except:
            ndx = len(self.prior_schedule) - 1
        if ndx > 0 and self.prior_schedule[ndx] == self.epoch:
            weight = self.prior_weights[ndx - 1] + self.prior_weights[ndx] * (
                        self.step - self.epoch * self.steps_per_epoch) / self.steps_per_epoch
        else:
            weight = self.prior_weights[ndx]

        return weight

    def get_prior_weights(self):
        steps_per_epoch = len(self.loader_train)
        tot_steps = steps_per_epoch * self.cfg.getint('training', 'n_epoch')

        prior_weights = self.cfg.get('training', 'energy_prior_weights')
        prior_weights = [float(v) for v in prior_weights.split(",")]
        prior_steps = self.cfg.get('training', 'n_start_prior')
        prior_steps = [int(v) for v in prior_steps.split(",")]
        n_trans = self.cfg.getint('training', 'n_prior_transition')
        weights = []
        for s in range(self.step, self.step + tot_steps):
            if s > prior_steps[-1]:
                ndx = len(prior_weights) - 1
                # weights.append(self.energy_prior_values[-1])
            else:
                for n in range(0, len(prior_steps)):
                    if s < prior_steps[n]:
                        # weights.append(self.energy_prior_values[n])
                        ndx = n
                        break
            # print(ndx)
            if ndx > 0 and s < prior_steps[ndx - 1] + self.cfg.getint('training', 'n_prior_transition'):
                weights.append(prior_weights[ndx - 1] + (prior_weights[ndx] - prior_weights[ndx - 1]) * (
                            s - prior_steps[ndx - 1]) / n_trans)
            else:
                weights.append(prior_weights[ndx])

        return weights

    def make_checkpoint(self):
        return self.out.make_checkpoint(
            self.step,
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_generator": self.opt_generator.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            },
        )

    def restore_latest_checkpoint(self):
        latest_ckpt = self.out.latest_checkpoint()
        if latest_ckpt is not None:
            checkpoint = torch.load(latest_ckpt)
            self.generator.load_state_dict(checkpoint["generator"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.opt_generator.load_state_dict(checkpoint["opt_generator"])
            self.opt_critic.load_state_dict(checkpoint["opt_critic"])
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.restored_model = True
            print("restored model!!!")

        self.out.prune_checkpoints()

    def map_to_device(self, tup):
        return tuple(
            tuple(y.to(device=self.device) for y in x) if type(x) is list else x.to(device=self.device) for x in tup)

    def transpose_and_zip(self, args):
        args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def featurize(self, grid, features):
        grid = grid[:, :, None, :, :, :] * features[:, :, :, None, None, None]
        # grid (BS, N_atoms, 1, N_x, N_y, N_z) * features (BS, N_atoms, N_features, 1, 1, 1)
        return torch.sum(grid, 1)

    def prepare_condition(self, fake_atom_grid, real_atom_grid, aa_featvec, bead_features):
        fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
        real_aa_features = self.featurize(real_atom_grid, aa_featvec)
        c_fake = fake_aa_features + bead_features
        c_real = real_aa_features + bead_features
        return c_fake, c_real

    def generator_loss(self, critic_fake):
        return (-1.0 * critic_fake).mean()

    def critic_loss(self, critic_real, critic_fake):
        loss_on_generated = critic_fake.mean()
        loss_on_real = critic_real.mean()

        loss = loss_on_generated - loss_on_real
        return loss

    def epsilon_penalty(self, epsilon, critic_real_outputs):
        if epsilon > 0:
            penalties = torch.pow(critic_real_outputs, 2)
            penalty = epsilon * penalties.mean()
            return penalty
        return 0.0

    def gradient_penalty(self, real_data, fake_data, mask):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradients_norm = ((gradients_norm - 1) ** 2)
        gradients_norm = gradients_norm * mask

        # Return gradient penalty
        return gradients_norm.mean()

    def get_energies_from_grid(self, atom_grid, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            if self.bond_pot == "threshold":
                b_energy = self.energy.bond_threshold2(coords, bond_ndx)
            else:
                b_energy = self.energy.bond(coords, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            if self.angle_pot == "threshold":
                a_energy = self.energy.angle_threshold2(coords, angle_ndx)
            else:
                a_energy = self.energy.angle(coords, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy.dih(coords, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            if self.nb_pot == "lj":
                l_energy = self.energy.lj(coords, lj_ndx)
            else:
                l_energy = self.energy.quadratic_repl(coords, lj_ndx)

            #l_energy = self.energy.lj(coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)

        return b_energy, a_energy, d_energy, l_energy

    def get_energies_from_coords(self, coords, energy_ndx):

        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        if bond_ndx.size()[1]:
            if self.bond_pot == "threshold":
                b_energy = self.energy.bond_threshold2(coords, bond_ndx)
            else:
                b_energy = self.energy.bond(coords, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            if self.angle_pot == "threshold":
                a_energy = self.energy.angle_threshold2(coords, angle_ndx)
            else:
                a_energy = self.energy.angle(coords, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy.dih(coords, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            if self.nb_pot == "lj":
                l_energy = self.energy.lj(coords, lj_ndx)
            else:
                l_energy = self.energy.quadratic_repl(coords, lj_ndx)
            #l_energy = self.energy.lj(coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)

        return b_energy, a_energy, d_energy, l_energy

    def get_forces(self, x, energy_ndx):
        x = x.requires_grad_(True)
        b_energy, angle_energy, dih_energy, lj_energy = self.get_energies_from_coords(x, energy_ndx)
        energy = b_energy + angle_energy + dih_energy + lj_energy
        # for f in torch.autograd.grad(energy, x, torch.ones_like(energy), create_graph=True, retain_graph=True):
        #    print(f.size())
        return -torch.autograd.grad(energy, x, torch.ones_like(energy), create_graph=True, retain_graph=True)[0]

    def dstr(self, atom_grid, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        if bond_ndx.size()[1] and bond_ndx.size()[1]:
            b_dstr = self.gauss_hist_bond(coords, bond_ndx)
        else:
            b_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1] and angle_ndx.size()[1]:
            a_dstr = self.gauss_hist_angle(coords, angle_ndx)
        else:
            a_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1] and dih_ndx.size()[1]:
            d_dstr = self.gauss_hist_dih(coords, dih_ndx)
        else:
            d_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1] and lj_ndx.size()[1]:
            nb_dstr = self.gauss_hist_nb(coords, lj_ndx)
        else:
            nb_dstr = torch.zeros([self.n_bins,1], dtype=torch.float32, device=self.device)
        return b_dstr, a_dstr, d_dstr, nb_dstr

    def dstr_loss(self, real_atom_grid, fake_atom_grid, energy_ndx):
        real_coords = avg_blob(
            real_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        fake_coords = avg_blob(
            fake_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            # b_dstr_real = self.gauss_hist_bond(self.energy.bond_dstr(real_coords, bond_ndx))
            # b_dstr_fake = self.gauss_hist_bond(self.energy.bond_dstr(fake_coords, bond_ndx))
            b_dstr_real = self.gauss_hist_bond(real_coords, bond_ndx)
            b_dstr_fake = self.gauss_hist_bond(fake_coords, bond_ndx)
            b_dstr_avg = 0.5 * (b_dstr_real + b_dstr_fake)
            # x = [h * 0.4/64 for h in range(0,64)]
            # plt.plot(x, b_dstr_real.detach().cpu().numpy()[:,0])
            # plt.plot(x, b_dstr_fake.detach().cpu().numpy()[:,0])
            # plt.show()

            # b_dstr_loss = (b_dstr_real * (b_dstr_real / b_dstr_fake).log()).sum(0)
            b_dstr_loss = 0.5 * ((b_dstr_real * (b_dstr_real / b_dstr_avg).log()).sum(0) + (
                        b_dstr_fake * (b_dstr_fake / b_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 0.4 / 64 for h in range(0, 64)]
                ax.plot(x, b_dstr_real.detach().cpu().numpy()[:, 0], label='ref')
                ax.plot(x, b_dstr_fake.detach().cpu().numpy()[:, 0], label='fake')
                # ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                # ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("bond", fig, global_step=self.step)
                plt.close(fig)

            # print(b_dstr_loss)
            # print(b_dstr_loss.size())
        else:
            b_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_dstr_real = self.gauss_hist_angle(real_coords, angle_ndx)
            a_dstr_fake = self.gauss_hist_angle(fake_coords, angle_ndx)
            a_dstr_avg = 0.5 * (a_dstr_real + a_dstr_fake)

            # self.out.add_(key, value, global_step=self.step)

            # plt.show()

            # a_dstr_loss = (a_dstr_real * (a_dstr_real / a_dstr_fake).log()).sum(0)
            a_dstr_loss = 0.5 * ((a_dstr_real * (a_dstr_real / a_dstr_avg).log()).sum(0) + (
                        a_dstr_fake * (a_dstr_fake / a_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 180.0 / 64 for h in range(0, 64)]
                ax.plot(x, a_dstr_real.detach().cpu().numpy()[:, 0], label='ref')
                ax.plot(x, a_dstr_fake.detach().cpu().numpy()[:, 0], label='fake')
                # ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                # ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("angle", fig, global_step=self.step)
                plt.close(fig)

            # print(a_dstr_loss)
            # print(a_dstr_loss.size())

        else:
            a_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_dstr_real = self.gauss_hist_dih(real_coords, dih_ndx)
            d_dstr_fake = self.gauss_hist_dih(fake_coords, dih_ndx)
            d_dstr_avg = 0.5 * (d_dstr_real + d_dstr_fake)

            # x = [h * 180.0/64 for h in range(0,64)]
            # plt.plot(x, d_dstr_real.detach().cpu().numpy()[:,0])
            # plt.plot(x, d_dstr_fake.detach().cpu().numpy()[:,0])
            # plt.show()
            # d_dstr_loss = (d_dstr_real * (d_dstr_real / d_dstr_fake).log()).sum(0)
            d_dstr_loss = 0.5 * ((d_dstr_real * (d_dstr_real / d_dstr_avg).log()).sum(0) + (
                        d_dstr_fake * (d_dstr_fake / d_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 180.0 / 64 for h in range(0, 64)]
                ax.plot(x, d_dstr_real.detach().cpu().numpy()[:, 0], label='ref')
                ax.plot(x, d_dstr_fake.detach().cpu().numpy()[:, 0], label='fake')
                # ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                # ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("dih", fig, global_step=self.step)
                plt.close(fig)

            # print(d_dstr_loss)
            # print(d_dstr_loss.size())
        else:
            d_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        if lj_ndx.size()[1]:
            nb_dstr_real = self.gauss_hist_nb(real_coords, lj_ndx)
            nb_dstr_fake = self.gauss_hist_nb(fake_coords, lj_ndx)
            nb_dstr_avg = 0.5 * (nb_dstr_real + nb_dstr_fake)

            nb_dstr_loss = 0.5 * ((nb_dstr_real * (nb_dstr_real / nb_dstr_avg).log()).sum(0) + (
                        nb_dstr_fake * (nb_dstr_fake / nb_dstr_avg).log()).sum(0))

            if self.step % 1 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 2.0 / 64 for h in range(0, 64)]
                ax.plot(x, nb_dstr_real.detach().cpu().numpy()[:, 0], label='ref')
                ax.plot(x, nb_dstr_fake.detach().cpu().numpy()[:, 0], label='fake')
                # ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                # ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("nonbonded", fig, global_step=self.step)
                plt.close(fig)

            # print(b_dstr_loss)
            # print(b_dstr_loss.size())
        else:
            nb_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        # print(torch.sum(b_dstr_loss))
        # print(torch.sum(a_dstr_loss))
        # print(torch.sum(d_dstr_loss))
        # print(torch.sum(nb_dstr_loss))

        return torch.sum(b_dstr_loss), torch.sum(a_dstr_loss), torch.sum(d_dstr_loss), torch.sum(nb_dstr_loss)

    def energy_min_loss(self, aa_coords, energy_ndx):

        fb, fa, fd, fl = self.get_energies_from_coords(aa_coords, energy_ndx)

        b_loss = torch.mean(fb)
        a_loss = torch.mean(fa)
        d_loss = torch.mean(fd)
        l_loss = torch.mean(fl)

        return b_loss, a_loss, d_loss, l_loss

    def energy_match_loss(self, real_atom_grid, fake_atom_grid, energy_ndx):

        rb, ra, rd, rl = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        b_loss = torch.mean(torch.abs(rb - fb))
        a_loss = torch.mean(torch.abs(ra - fa))
        d_loss = torch.mean(torch.abs(rd - fd))
        l_loss = torch.mean(torch.abs(rl - fl))

        return b_loss, a_loss, d_loss, l_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def moment_loss(self, real_atom_grid, fake_atom_grid, energy_ndx):

        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        coords_ref = avg_blob(
            real_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        coords_bm = avg_blob(
            fake_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        if bond_ndx.size()[1]:
            b1_loss, b2_loss = self.energy.bond_moment_loss(coords_ref, coords_bm, bond_ndx)
        else:
            #b_loss, b_var = torch.zeros([], dtype=torch.float32, device=self.device), torch.zeros([], dtype=torch.float32, device=self.device)
            b1_loss, b2_loss = torch.zeros([], dtype=torch.float32, device=self.device), torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a1_loss, a2_loss = self.energy.angle_moment_loss(coords_ref, coords_bm, angle_ndx)
        else:
            a1_loss, a2_loss = torch.zeros([], dtype=torch.float32, device=self.device), torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d1_loss, d2_loss = self.energy.dih_moment_loss(coords_ref, coords_bm, dih_ndx)
        else:
            d1_loss, d2_loss = torch.zeros([], dtype=torch.float32, device=self.device), torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            #nb_energy_ref = self.energy.exp_repl(coords_ref, lj_ndx)
            nb_loss = self.energy.quadratic_repl(coords_bm, lj_ndx)
            #nb_loss = torch.mean((nb_energy_bm - nb_energy_ref)**2)
        else:
            nb_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        #print("FFFFFFF")
        #print(b1_loss, b2_loss, a1_loss, a2_loss, d1_loss, d2_loss, nb_loss)

        return torch.sum(b1_loss + b2_loss), torch.sum(a1_loss + a2_loss), torch.sum(d1_loss + d2_loss), torch.sum(nb_loss)

    def detach(self, t):
        t = tuple([c.detach().cpu().numpy() for c in t])
        return t

    def train(self):
        steps_per_epoch = len(self.loader_train)
        n_critic = self.cfg.getint('training', 'n_critic')
        n_save = int(self.cfg.getint('training', 'n_save'))

        epochs = tqdm(range(self.epoch, self.cfg.getint('training', 'n_epoch')))
        epochs.set_description('epoch: ')
        for epoch in epochs:
            n = 0
            loss_epoch = [[], [], [], [], [], [], []]
            val_iterator = iter(self.loader_val)
            tqdm_train_iterator = tqdm(self.loader_train, total=steps_per_epoch, leave=False)
            for train_batch in tqdm_train_iterator:

                train_batch = self.map_to_device(train_batch)
                elems, initial, energy_ndx = train_batch
                elems = self.transpose_and_zip(elems)

                if n == n_critic:
                    g_loss_dict = self.train_step_gen(elems, initial, energy_ndx)
                    for key, value in g_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    tqdm_train_iterator.set_description('D: {}, G: {}, {}, {}, {}, {}, {}, {}'.format(c_loss,
                                                                                                      g_loss_dict[
                                                                                                          'Generator/wasserstein'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/energy'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/energy_bond'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/energy_angle'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/energy_dih'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/energy_lj'],
                                                                                                      g_loss_dict[
                                                                                                          'Generator/prior_weight']))

                    for value, l in zip([c_loss] + list(g_loss_dict.values()), loss_epoch):
                        l.append(value)
                    if self.loader_val:
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(self.loader_val)
                            val_batch = next(val_iterator)
                        val_batch = self.map_to_device(val_batch)
                        elems, initial, energy_ndx = val_batch
                        elems = self.transpose_and_zip(elems)
                        g_loss_dict = self.train_step_gen(elems, initial, energy_ndx, backprop=False)
                        for key, value in g_loss_dict.items():
                            self.out.add_scalar(key, value, global_step=self.step, mode='val')
                    self.step += 1
                    n = 0

                else:
                    c_loss = self.train_step_critic(elems, initial)
                    n += 1

            tqdm.write('epoch {} steps {} : D: {} G: {}, {}, {}, {}, {}, {}'.format(
                self.epoch,
                self.step,
                sum(loss_epoch[0]) / len(loss_epoch[0]),
                sum(loss_epoch[1]) / len(loss_epoch[1]),
                sum(loss_epoch[2]) / len(loss_epoch[2]),
                sum(loss_epoch[3]) / len(loss_epoch[3]),
                sum(loss_epoch[4]) / len(loss_epoch[4]),
                sum(loss_epoch[5]) / len(loss_epoch[5]),
                sum(loss_epoch[6]) / len(loss_epoch[6]),
            ))

            self.epoch += 1

            if self.epoch % n_save == 0:
                self.make_checkpoint()
                self.out.prune_checkpoints()
                self.validate()

    def train_step_critic(self, elems, initial):
        c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        aa_grid, cg_features, _ = initial

        fake_atom_grid = aa_grid.clone()
        real_atom_grid = aa_grid.clone()

        for target_atom, target_type, aa_featvec, repl, mask in elems:
            # prepare input for generator
            c_fake, c_real = self.prepare_condition(fake_atom_grid, real_atom_grid, aa_featvec, cg_features)
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            fake_data = torch.cat([fake_atom, c_fake], dim=1)
            real_data = torch.cat([target_atom[:, None, :, :, :], c_real], dim=1)

            # critic
            critic_fake = self.critic(fake_data)
            critic_real = self.critic(real_data)

            # mask
            critic_fake = torch.squeeze(critic_fake) * mask
            critic_real = torch.squeeze(critic_real) * mask

            # loss
            c_wass = self.critic_loss(critic_real, critic_fake)
            c_eps = self.epsilon_penalty(1e-3, critic_real)
            c_loss += c_wass + c_eps
            if self.use_gp:
                c_gp = self.gradient_penalty(real_data, fake_data, mask)
                c_loss += c_gp

            # update aa grids
            fake_atom_grid = torch.where(repl[:, :, None, None, None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:, :, None, None, None], real_atom_grid, target_atom[:, None, :, :, :])

        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        return c_loss.detach().cpu().numpy()

    def train_step_gen(self, elems, initial, energy_ndx, backprop=True):

        aa_grid, cg_features, aa_coords = initial

        g_wass = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_aa_coords = aa_coords.clone()
        #real_aa_coords = aa_coords.clone()
        fake_atom_grid = aa_grid.clone()
        real_atom_grid = aa_grid.clone()

        for target_atom, target_type, aa_featvec, repl, mask in elems:
            # prepare input for generator
            fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
            c_fake = fake_aa_features + cg_features
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            # critic
            critic_fake = self.critic(torch.cat([fake_atom, c_fake], dim=1))

            # mask
            critic_fake = torch.squeeze(critic_fake) * mask

            # loss
            g_wass += self.generator_loss(critic_fake)
            # g_loss += g_wass

            new_coord = avg_blob(
            fake_atom,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
            )
            fake_aa_coords = torch.where(repl[:,:,None], fake_aa_coords, new_coord)

            # update aa grids
            fake_atom_grid = torch.where(repl[:, :, None, None, None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:, :, None, None, None], real_atom_grid, target_atom[:, None, :, :, :])

        g_loss_dict = {}
        b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_aa_coords, energy_ndx)
        energy_loss = (self.ratio_bonded_nonbonded * (b_energy + a_energy + d_energy) + l_energy) * self.prior_weight()

        b_dstr_real, a_dstr_real, d_dstr_real, nb_dstr_real = self.dstr(real_atom_grid, energy_ndx)
        b_dstr_fake, a_dstr_fake, d_dstr_fake, nb_dstr_fake = self.dstr(fake_atom_grid, energy_ndx)
        if self.step % 200 == 0:
            self.out.mk_and_add_fig(b_dstr_real, b_dstr_fake, self.gauss_hist_bond, tag="bond", global_step=self.step)
            self.out.mk_and_add_fig(a_dstr_real, a_dstr_fake, self.gauss_hist_angle, tag="angle", global_step=self.step)
            self.out.mk_and_add_fig(d_dstr_real, d_dstr_fake, self.gauss_hist_dih, tag="dih", global_step=self.step)
            self.out.mk_and_add_fig(nb_dstr_real, nb_dstr_fake, self.gauss_hist_nb, tag="nonbonded", global_step=self.step)

        if self.prior_mode == 'match' and self.prior_weight() > 0.0:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_match_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded*(b_loss + a_loss + d_loss) + l_loss) * self.prior_weight()
            g_loss = g_wass + energy_loss
        elif self.prior_mode == 'moment' and self.prior_weight() > 0.0:
            b_loss, a_loss, d_loss, nb_loss = self.moment_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded*(b_loss + a_loss + d_loss) + nb_loss) * self.prior_weight()
            g_loss =  g_wass + energy_loss

            g_loss_dict["Generator/b_dstr"] = b_loss.detach().cpu().numpy()
            g_loss_dict["Generator/a_dstr"] = a_loss.detach().cpu().numpy()
            g_loss_dict["Generator/d_dstr"] = d_loss.detach().cpu().numpy()
            g_loss_dict["Generator/nb_dstr"] = nb_loss.detach().cpu().numpy()
        elif self.prior_mode == 'min' and self.prior_weight() > 0.0:
            g_loss = g_wass + energy_loss
        elif self.prior_mode == 'dstr_lj' and self.prior_weight() > 0.0:
            b_loss, a_loss, d_loss = self.dstr_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            dstr_loss = (self.ratio_bonded_nonbonded*(b_loss + a_loss + d_loss) + l_energy) * self.prior_weight()
            g_loss = g_wass + dstr_loss
        elif self.prior_mode == 'dstr' and self.prior_weight() > 0.0:
            b_dstr_avg = 0.5 * (b_dstr_real + b_dstr_fake)
            b_dstr_loss = 0.5 * ((b_dstr_real * (b_dstr_real / b_dstr_avg).log()).sum(0) + (
                        b_dstr_fake * (b_dstr_fake / b_dstr_avg).log()).sum(0))

            a_dstr_avg = 0.5 * (a_dstr_real + a_dstr_fake)
            a_dstr_loss = 0.5 * ((a_dstr_real * (a_dstr_real / a_dstr_avg).log()).sum(0) + (
                        a_dstr_fake * (a_dstr_fake / a_dstr_avg).log()).sum(0))

            d_dstr_avg = 0.5 * (d_dstr_real + d_dstr_fake)
            d_dstr_loss = 0.5 * ((d_dstr_real * (d_dstr_real / d_dstr_avg).log()).sum(0) + (
                        d_dstr_fake * (d_dstr_fake / d_dstr_avg).log()).sum(0))

            nb_dstr_avg = 0.5 * (nb_dstr_real + nb_dstr_fake)
            nb_dstr_loss = 0.5 * ((nb_dstr_real * (nb_dstr_real / nb_dstr_avg).log()).sum(0) + (
                        nb_dstr_fake * (nb_dstr_fake / nb_dstr_avg).log()).sum(0))


            dstr_loss = self.prior_weight() * (
                        (b_dstr_loss + a_dstr_loss + d_dstr_loss) * self.ratio_bonded_nonbonded + nb_dstr_loss)
            #dstr_loss.register_hook(self.hook_fn_backward)
            g_loss = g_wass + dstr_loss

            g_loss_dict["Generator/b_dstr"] = b_dstr_loss.detach().cpu().numpy()
            g_loss_dict["Generator/a_dstr"] = a_dstr_loss.detach().cpu().numpy()
            g_loss_dict["Generator/d_dstr"] = d_dstr_loss.detach().cpu().numpy()
            g_loss_dict["Generator/nb_dstr"] = nb_dstr_loss.detach().cpu().numpy()

        elif self.prior_mode == 'dstr_abs' and self.prior_weight() > 0.0:
            b_loss = torch.mean(torch.abs(b_dstr_real - b_dstr_fake))
            a_loss = torch.mean(torch.abs(a_dstr_real - a_dstr_fake))
            d_loss = torch.mean(torch.abs(d_dstr_real - d_dstr_fake))
            l_loss = torch.mean(torch.abs(nb_dstr_real - nb_dstr_fake))
            dstr_loss = self.prior_weight() * ((b_loss + a_loss + d_loss) * self.ratio_bonded_nonbonded + l_loss)
            # dstr_loss.register_hook(self.hook_fn_backward)
            g_loss = g_wass + dstr_loss

            g_loss_dict["Generator/b_dstr"] = b_loss.detach().cpu().numpy()
            g_loss_dict["Generator/a_dstr"] = a_loss.detach().cpu().numpy()
            g_loss_dict["Generator/d_dstr"] = d_loss.detach().cpu().numpy()
            g_loss_dict["Generator/nb_dstr"] = l_loss.detach().cpu().numpy()
        else:
            g_loss = g_wass

        #g_loss = g_wass + self.prior_weight() * energy_loss
        #g_loss = g_wass

        if backprop:
            self.opt_generator.zero_grad()
            g_loss.backward()
            self.opt_generator.step()


        g_loss_dict["Generator/wasserstein"] = g_wass.detach().cpu().numpy()
        g_loss_dict["Generator/energy"] = energy_loss.detach().cpu().numpy()
        g_loss_dict["Generator/energy_bond"] = b_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_angle"] = a_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_dih"] = d_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_lj"] = l_energy.detach().cpu().numpy()
        g_loss_dict["Generator/prior_weight"] = self.prior_weight()

        return g_loss_dict

    """
        g_loss_dict = {}

        if self.prior_mode == 'match':
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_match_loss(
                real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded * (b_loss + a_loss + d_loss) + l_loss) * self.prior_weight()
            g_loss = g_wass + energy_loss
        elif self.prior_mode == 'min':
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded * (
                        b_energy + a_energy + d_energy) + l_energy) * self.prior_weight()
            g_loss = g_wass + energy_loss
        elif self.prior_mode == 'dstr_lj':
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)
            b_loss, a_loss, d_loss = self.dstr_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded * (b_loss + a_loss + d_loss) + l_energy) * self.prior_weight()
            g_loss = g_wass + energy_loss
        elif self.prior_mode == 'dstr':
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)
            b_loss, a_loss, d_loss, l_loss = self.dstr_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = (b_loss + a_loss + d_loss + l_loss) * self.prior_weight()
            # energy_loss = (d_loss ) * self.prior_weight()

            g_loss = g_wass + energy_loss
            g_loss_dict["Generator/b_dstr"] = b_loss.detach().cpu().numpy()
            g_loss_dict["Generator/a_dstr"] = a_loss.detach().cpu().numpy()
            g_loss_dict["Generator/d_dstr"] = d_loss.detach().cpu().numpy()

        else:
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded * (
                        b_energy + a_energy + d_energy) + l_energy) * self.prior_weight()
            g_loss = g_wass

        # g_loss = g_wass + self.prior_weight() * energy_loss
        # g_loss = g_wass

        if backprop:
            self.opt_generator.zero_grad()
            g_loss.backward()
            self.opt_generator.step()

        g_loss_dict["Generator/wasserstein"] = g_wass.detach().cpu().numpy()
        g_loss_dict["Generator/energy"] = energy_loss.detach().cpu().numpy()
        g_loss_dict["Generator/energy_bond"] = b_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_angle"] = a_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_dih"] = d_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_lj"] = l_energy.detach().cpu().numpy()
        g_loss_dict["Generator/prior_weight"] = self.prior_weight()

        return g_loss_dict
    """

    def to_tensor_and_zip(self, *args):
        args = tuple(
            torch.from_numpy(x).float().to(self.device) if x.dtype == np.dtype(np.float64) else torch.from_numpy(x).to(
                self.device) for x in args)
        # args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def to_tensor(self, t):
        return tuple(torch.from_numpy(x).to(self.device) for x in t)

    def transpose(self, t):
        return tuple(torch.transpose(x, 0, 1) for x in t)

    def insert_dim(self, t):
        return tuple(x[None, :] for x in t)

    def repeat(self, t, bs):
        return tuple(torch.stack(bs * [x]) for x in t)

    def to_voxel(self, coords, grid, sigma):
        coords = coords[..., None, None, None]
        return torch.exp(-1.0 * torch.sum((grid - coords) * (grid - coords), axis=2) / sigma).float()

    def predict(self, elems, initial, energy_ndx, bs):

        aa_grid, cg_features, aa_coords = initial

        if self.prior_mode == 'critic':
            c_loss = torch.zeros([aa_grid.shape[0]], dtype=torch.float32, device=self.device)

        generated_atoms = []
        for target_type, aa_featvec, repl in zip(*elems):
            fake_aa_features = self.featurize(aa_grid, aa_featvec)
            c_fake = fake_aa_features + cg_features
            target_type = target_type.repeat(bs, 1)
            z = torch.empty(
                [target_type.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)
            generated_atoms.append(fake_atom)

            if self.prior_mode == 'critic':
                fake_data = torch.cat([fake_atom, c_fake], dim=1)
                c_loss += torch.squeeze(-self.critic(fake_data))

            # update aa grids
            aa_grid = torch.where(repl[:, :, None, None, None], aa_grid, fake_atom)

            new_coord = avg_blob(
            fake_atom,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
            )
            aa_coords = torch.where(repl[:,:,None], aa_coords, new_coord)

        # generated_atoms = torch.stack(generated_atoms, dim=1)
        generated_atoms = torch.cat(generated_atoms, dim=1)

        generated_atoms_coords = avg_blob(
            generated_atoms,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        if self.prior_mode == 'critic':
            energy = c_loss
        else:
            b_energy, a_energy, d_energy, l_energy = self.get_energies_from_coords(aa_coords, energy_ndx)
            energy = b_energy + a_energy + d_energy + l_energy

        return generated_atoms_coords, energy

    def validate(self, samples_dir=None):

        if samples_dir:
            samples_dir = self.out.output_dir / samples_dir
            make_dir(samples_dir)
        else:
            samples_dir = self.out.samples_dir
        stats = Stats(self.data, dir=samples_dir / "stats")

        print("Saving samples in {}".format(samples_dir), "...", end='')

        resolution = self.cfg.getint('grid', 'resolution')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma = self.cfg.getfloat('grid', 'sigma')
        #val_bs = self.cfg.getint('validate', 'batchsize')
        # grid = make_grid_np(delta_s, resolution)

        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)
        rot_mtxs = torch.from_numpy(rot_mtx_batch(self.val_bs)).to(self.device).float()
        rot_mtxs_transposed = torch.from_numpy(rot_mtx_batch(self.val_bs, transpose=True)).to(self.device).float()

        data_generators = []
        #if self.cfg.getboolean('training', 'init_intra'):
        #    data_generators.append(iter(
        #        Recurrent_Generator(self.data, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False,
        #                            ref_pos=False, inter=False)))
        #    if self.cfg.getboolean('training', 'hydrogens'):
        #        data_generators.append(iter(
        #            Recurrent_Generator(self.data, hydrogens=True, gibbs=False, train=False, rand_rot=False,
        #                                pad_seq=False, ref_pos=False, inter=False)))

        try:
            initialize_from_reference = self.cfg.getboolean('validate', 'initialize_from_reference')
        except:
            initialize_from_reference = False

        if initialize_from_reference:
            for sample in self.data.samples_val:
                for a in sample.atoms:
                    a.pos = a.ref_pos
            print("Reference structure is used as initial positions, i.e. this is not real backampping. Only used for morphing.")

        data_generators.append(iter(
            Recurrent_Generator(self.data, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False,
                                ref_pos=False)))
        if self.cfg.getboolean('training', 'hydrogens'):
            data_generators.append(iter(
                Recurrent_Generator(self.data, hydrogens=True, gibbs=False, train=False, rand_rot=False, pad_seq=False,
                                    ref_pos=False)))

        for m in range(self.n_gibbs):
            data_generators.append(iter(
                Recurrent_Generator(self.data, hydrogens=False, gibbs=True, train=False, rand_rot=False, pad_seq=False,
                                    ref_pos=False)))
            if self.cfg.getboolean('training', 'hydrogens'):
                data_generators.append(iter(
                    Recurrent_Generator(self.data, hydrogens=True, gibbs=True, train=False, rand_rot=False,
                                        pad_seq=False, ref_pos=False)))
        m = 0
        try:
            self.generator.eval()
            self.critic.eval()

            times = []

            for data_gen in data_generators:
                start = timer()

                for d in data_gen:
                    print(m)
                    m = m + 1

                    """
                    for t_pos, aa_feat, repl, a in zip(d['target_pos'], d['aa_feat'], d['repl'], d['atom_seq']):
                        print(a.type.name)
                        print(a.mol_index)
                        coords = np.concatenate((d['aa_pos'], d['cg_pos']))
                        featvec = np.concatenate((aa_feat, d['cg_feat']))
                        _, n_channels = featvec.shape
                        fig = plt.figure(figsize=(20, 20))
                        for c in range(0, n_channels):
                            ax = fig.add_subplot(int(np.ceil(np.sqrt(n_channels))), int(np.ceil(np.sqrt(n_channels))),
                                                 c + 1, projection='3d')
                            ax.set_title("Chn. Nr:" + str(c))
                            for n in range(0, len(coords)):
                                if featvec[n, c] == 1:
                                    ax.scatter(coords[n, 0], coords[n, 1], coords[n, 2], s=5, marker='o', color='black',
                                               alpha=0.5)
                            ax.scatter(t_pos[0, 0], t_pos[0, 1], t_pos[0, 2], s=5, marker='o', color='red')
                            ax.set_xlim3d(-1.0, 1.0)
                            ax.set_ylim3d(-1.0, 1.0)
                            ax.set_zlim3d(-1.0, 1.0)
                            ax.set_xticks(np.arange(-1, 1, step=0.5))
                            ax.set_yticks(np.arange(-1, 1, step=0.5))
                            ax.set_zticks(np.arange(-1, 1, step=0.5))
                            ax.tick_params(labelsize=6)
                            plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
                        plt.show()
                        # print(repl.shape)
                        # print(d['aa_pos'].shape)
                        # print(t_pos.shape)
                        d['aa_pos'] = np.where(repl[:, np.newaxis], d['aa_pos'], t_pos)
                    """

                    with torch.no_grad():

                        aa_coords = torch.matmul(torch.from_numpy(d['aa_pos']).to(self.device).float(), rot_mtxs)
                        cg_coords = torch.matmul(torch.from_numpy(d['cg_pos']).to(self.device).float(), rot_mtxs)

                        # aa_coords = torch.from_numpy(d['aa_pos']).to(self.device).float()
                        # cg_coords = torch.from_numpy(d['cg_pos']).to(self.device).float()

                        aa_grid = self.to_voxel(aa_coords, grid, sigma)
                        cg_grid = self.to_voxel(cg_coords, grid, sigma)

                        cg_features = torch.from_numpy(d['cg_feat'][None, :, :, None, None, None]).to(
                            self.device) * cg_grid[:, :, None, :, :, :]
                        cg_features = torch.sum(cg_features, 1)

                        initial = (aa_grid, cg_features, aa_coords)

                        elems = (d['target_type'], d['aa_feat'], d['repl'])
                        elems = self.transpose(self.insert_dim(self.to_tensor(elems)))

                        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])
                        energy_ndx1 = self.repeat(self.to_tensor(energy_ndx), 1)
                        energy_ndx = self.repeat(self.to_tensor(energy_ndx), self.val_bs)

                        new_coords, energies = self.predict(elems, initial, energy_ndx, self.val_bs)

                    if self.val_mode in ['min', 'match', 'critic']:
                        ndx = energies.argmin()
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                    elif self.val_mode == "rm_outlier":
                        energies = (-energies).detach().cpu().numpy()
                        sorted_arg = energies.argsort()[int(self.val_bs/2):]
                        ndx = np.random.choice(sorted_arg)
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                    elif self.val_mode == "avg":
                        new_coords = torch.matmul(new_coords, rot_mtxs_transposed)
                        new_coords = torch.sum(new_coords, 0) / self.val_bs
                    elif self.val_mode == "EM":
                        # new_coords = torch.matmul(new_coords, rot_mtxs_transposed)
                        # new_coords = torch.sum(new_coords, 0) / self.val_bs
                        ndx = energies.argmin()
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                        new_coords = self.minimize_energy(new_coords,
                                                          torch.from_numpy(d['aa_pos']).to(self.device).float(),
                                                          torch.from_numpy(d['repl']).to(self.device), energy_ndx1,
                                                          1E-8, 20)

                    else:
                        ndx = np.random.randint(self.val_bs)
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])

                    new_coords = new_coords.detach().cpu().numpy()

                    for c, a in zip(new_coords, d['atom_seq']):
                        a.pos = d['loc_env'].rot_back(c)
                        # a.ref_pos = d['loc_env'].rot_back(c)
                times.append(timer() - start)

                print(timer() - start)

            subdir = "ep" + str(self.epoch) + "_valbs" + str(self.val_bs) + "_gibbs" + str(self.n_gibbs) + self.prior_mode
            stats.save_samples(train=False, subdir=subdir)
            # stats.save_samples(train=False, subdir="ep" + str(self.epoch) + "_valbs" + str(self.val_bs)+ "_gibbs" + str(self.n_gibbs), vs=True)
            if self.cfg.getboolean('validate', 'evaluate'):
                stats.evaluate(train=False, subdir=subdir)
            # reset atom positions
            for sample in self.data.samples_val:
                # sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
                sample.kick_atoms()

            with open("timings_" + self.name, 'a') as f:
                f.write(str(times))

        finally:
            self.generator.train()
            self.critic.train()

    def model_energy(self, coords, mus):
        dis_sq = (coords - mus) ** 2
        e = torch.sum(dis_sq)
        return e

    def minimize_energy(self, var_coords, fixed_coords, repls, energy_ndx, delta, iterations):

        var_coords = Variable(var_coords, requires_grad=True)

        opt_em = SGD([var_coords], lr=delta)

        start_pos = var_coords.clone()
        for i in range(0, iterations):
            for repl, var_coord in zip(repls, var_coords):
                fixed_coords = torch.where(repl[:, None], fixed_coords, var_coord[None, :])
            # print(fixed_coords.shape)
            b_energy, a_energy, d_energy, l_energy = self.get_energies_from_coords(fixed_coords[None, :, :], energy_ndx)
            energy = b_energy + a_energy + d_energy + l_energy + 1000000 * self.model_energy(var_coords, start_pos)
            if i == 0:
                start_energy = energy.clone()

            if i == 0 or i == iterations - 1:
                print("BOND: ", b_energy)
                print("ANGLE: ", a_energy)
                print("DIH: ", d_energy)
                print("LJ: ", l_energy)
                print("MODEL: ", self.model_energy(var_coords, start_pos))

            opt_em.zero_grad()
            energy.backward(retain_graph=True)
            opt_em.step()

        if energy > start_energy:
            return start_pos
        else:
            return var_coords
