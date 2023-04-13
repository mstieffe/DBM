import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from timeit import default_timer as timer
from dbm.util import rot_mtx_batch, make_dir, avg_blob, make_grid_np, transpose_and_zip, transpose, insert_dim, repeat, to_voxel
from dbm.output import OutputHandler
from tqdm.autonotebook import tqdm
import dbm.model as model
from dbm.data import Data
from dbm.histogram import Histogram
from dbm.energy import Energy
from dbm.stats import Stats
from dbm.dataset import DS
from dbm.recurrent_generator import Recurrent_Generator

torch.set_default_dtype(torch.float32)

class GAN_seq():

    """
    A class representing a Generative Adversarial Network (GANs) for generating molecular structures sequentially.

    Args:
    - device: The device to use for computation (e.g. "cpu", "cuda").
    - cfg (ConfigParser): A ConfigParser object containing the configuration settings for the GAN model.

    Class properties:
    - device: The device to use for computation.
    - cfg (ConfigParser): A ConfigParser object containing the configuration settings for the GAN model.
    - bs (int): An integer representing the batch size used for training.
    - data (Data): A Data object containing the data pipeline for generating molecular structures.
    - loader_train (DataLoader): A DataLoader object containing the data pipeline for training the GAN model.
    - loader_val (DataLoader): A DataLoader object containing the data pipeline for validation of the GAN model.
    - steps_per_epoch (int): An integer representing the number of steps per epoch used for training.
    - name (str): A string representing the name of the GAN model.
    - z_dim (int): An integer representing the dimension of the noise vector used for generating molecular structures.
    - n_atom_chns (int): An integer representing the number of atom  feature channels in the input tensor.
    - z_and_label_dim (int): An integer representing the dimension of the noise vector and the number of atom channels.
    - use_gp (bool): A boolean value indicating whether or not to use gradient penalty during training.
    - out (OutputHandler): An OutputHandler object for handling saving and tracking of the GAN model.
    - energy (Energy): An Energy object for energy calculation of molecular structures.
    - histogram (Histogram): A Histogram object for computing histograms of molecular properties.
    - prior_weights (list of float): A list of weights for balancing the prior loss term (energy term).
    - prior_schedule (numpy array of int): A numpy array representing the schedule for updating the prior weights during training.
    - ratio_bonded_nonbonded (float): A float representing the ratio between bonded and nonbonded interactions.
    - prior_mode (str): A string indicating the mode for the prior loss term.
    - val_mode (str): A string indicating the mode for validation during training.
    - val_bs (int): An integer representing the batch size used for validation during training.
    - n_gibbs (int): An integer representing the number of Gibbs iterations used for sampling during validation.
    - critic (AtomCrit_tiny): An AtomCrit_tiny object representing the critic network in the GAN model.
    - generator (AtomGen_tiny): An AtomGen_tiny object representing the generator network in the GAN model.
    - opt_generator (Adam): An Adam optimizer object for optimizing the generator network.
    - opt_critic (Adam): An Adam optimizer object for optimizing the critic network.
    - restored_model (bool): A boolean value indicating whether or not the GAN model has been restored from a saved checkpoint.
    - step (int): An integer representing the current step during training.
    - epoch (int): An integer representing the current epoch during training.
    """

    def __init__(self, device, cfg):

        # Initialize the class with given device and configuration
        self.device = device
        self.cfg = cfg

        # Set batch size for training
        self.bs = self.cfg.getint('training', 'batchsize')

        # Create data object
        self.data = Data(cfg, save=False)

        # Set up training data loader
        ds_train = DS(self.data, cfg, train=True, verbose=True)
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

        # Set up validation data loader
        self.loader_val = []
        if len(ds_train) != 0:
            ds_val = DS(self.data, cfg, train=False)
            # If the dataset is not empty, create a data loader object for validation data
            if len(ds_val) != 0:
                self.loader_val = DataLoader(
                    ds_val,
                    batch_size=self.bs,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=0,
                )

        # Calculate steps per epoch based on training data loader size and number of critic training steps + 1 step of the generator
        self.steps_per_epoch = int(len(self.loader_train) / (self.cfg.getint('training', 'n_critic') + 1))


        # Set model properties
        self.name = cfg.get('model', 'name')
        self.z_dim = int(cfg.getint('model', 'noise_dim'))
        self.n_atom_chns = self.data.ff.n_atom_chns
        self.z_and_label_dim = self.z_dim + self.n_atom_chns
        self.use_gp = cfg.getboolean('model', 'gp')


        # Create output handler object for saving results
        self.out = OutputHandler(
            self.name,
            self.cfg.getint('training', 'n_checkpoints'),
            self.cfg.get('model', 'output_dir'),
        )

        # Create energy object
        self.energy = Energy(self.data.ff, self.device)

        # Create histogram object
        self.histogram = Histogram(self.cfg, self.data.ff, self.device, n_bins=32)

        # Load prior configuration from cfg
        try:
            prior_weights = self.cfg.get('prior', 'weights')
            self.prior_weights = [float(v) for v in prior_weights.split(",")]
        except:
            self.prior_weights = [0.0]
        try:
            prior_schedule = self.cfg.get('prior', 'schedule')
            self.prior_schedule = np.array([0] + [int(v) for v in prior_schedule.split(",")])
        except:
            self.prior_schedule = np.array([0])
        self.ratio_bonded_nonbonded = self.cfg.getfloat('prior', 'ratio_bonded_nonbonded')
        self.prior_mode = cfg.get('prior', 'mode')

        # Load validation configuration from cfg
        try:
            self.val_mode = self.cfg.get('validate', 'mode')
        except:
            self.val_mode = "min"
        try:
            self.val_bs = self.cfg.getint('validate', 'batchsize')
        except:
            self.val_bs = 64
        self.n_gibbs = int(cfg.getint('validate', 'n_gibbs'))

        # Model set up
        # If the resolution is 8, use the AtomCrit_tiny and AtomGen_tiny models defined in the model module
        if cfg.getint('grid', 'resolution') == 8:
            self.critic = model.AtomCrit_tiny(in_channels=self.data.ff.n_channels + 1,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)
            self.generator = model.AtomGen_tiny(z_dim=self.z_and_label_dim,
                                                in_channels=self.data.ff.n_channels,
                                                start_channels=self.cfg.getint('model', 'n_chns'),
                                                fac=1,
                                                sn=self.cfg.getint('model', 'sn_gen'),
                                                device=device)
        else:
            raise Exception("Model for other resolution than 8 is not implemented currently. Go ahead and add it to model.py")

        # Move the critic and generator models to the specified device
        self.critic.to(device=device)
        self.generator.to(device=device)

        # Define the optimizer for the generator and the critic models
        self.opt_generator = Adam(self.generator.parameters(), lr=0.00005, betas=(0, 0.9))
        self.opt_critic = Adam(self.critic.parameters(), lr=0.0001, betas=(0, 0.9))

        # Set a flag to indicate that the model has not been restored from a checkpoint yet
        self.restored_model = False

        # Attempt to restore the latest checkpoint
        self.restore_latest_checkpoint()

        # Set the initial step and epoch to 0
        self.step = 0
        self.epoch = 0


    def make_checkpoint(self):
        # This function creates a checkpoint of the current state of the model
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
        # This function restores the latest checkpoint of the model if one exists
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

        # Delete any old checkpoints that are no longer needed
        self.out.prune_checkpoints()

    def to_tensor(self, t):
        # Method for converting a tuple of NumPy arrays to PyTorch tensors on the device specified in self.device
        return tuple(torch.from_numpy(x).to(self.device) for x in t)

    def map_to_device(self, tup):
        # Method for mapping a tuple of PyTorch tensors to the device specified in self.device
        return tuple(
            tuple(y.to(device=self.device) for y in x) if type(x) is list else x.to(device=self.device) for x in tup)

    def prior_weight(self):
        # Method for calculating the weight to use for the prior loss term based on the current epoch and step

        # Find the index of the current epoch in the prior schedule
        try:
            ndx = next(x[0] for x in enumerate(self.prior_schedule) if x[1] > self.epoch) - 1
        except:
            ndx = len(self.prior_schedule) - 1

        # Calculate the weight based on the index and the current step
        if ndx > 0 and self.prior_schedule[ndx] == self.epoch:
            # If the current epoch is in the middle of a weight transition, interpolate between the two weights
            weight = self.prior_weights[ndx - 1] + self.prior_weights[ndx] * (
                        self.step - self.epoch * self.steps_per_epoch) / self.steps_per_epoch
        else:
            # Otherwise, use the weight corresponding to the current epoch
            weight = self.prior_weights[ndx]

        return weight



    def featurize(self, grid, features):
        # This method featurizes the input grid based on the given features.
        grid = grid[:, :, None, :, :, :] * features[:, :, :, None, None, None]
        # grid (BS, N_atoms, 1, N_x, N_y, N_z) * features (BS, N_atoms, N_features, 1, 1, 1)
        return torch.sum(grid, 1)

    def prepare_condition(self, fake_atom_grid, real_atom_grid, aa_featvec, bead_features):
        # This method prepares the condition for training the generator and the critic.
        fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
        real_aa_features = self.featurize(real_atom_grid, aa_featvec)
        c_fake = fake_aa_features + bead_features
        c_real = real_aa_features + bead_features
        return c_fake, c_real

    def generator_loss(self, critic_fake):
        # This method calculates the wasserstein loss for the generator.
        return (-1.0 * critic_fake).mean()

    def critic_loss(self, critic_real, critic_fake):
        # This method calculates the wasserstein loss for the critic.
        loss_on_generated = critic_fake.mean()
        loss_on_real = critic_real.mean()

        loss = loss_on_generated - loss_on_real
        return loss

    def epsilon_penalty(self, epsilon, critic_real_outputs):
        # This method calculates the epsilon penalty for the Wasserstein GAN, i.e. makes the putput of the critic
        # for real samples close to 0
        if epsilon > 0:
            penalties = torch.pow(critic_real_outputs, 2)
            penalty = epsilon * penalties.mean()
            return penalty
        return 0.0

    def gradient_penalty(self, real_data, fake_data, mask):
        # This method calculates the gradient penalty for the Wasserstein GAN.
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

    def train(self):
        # Method to train the model

        # Get the number of steps per epoch, critic iteration count, and save count from the configuration file
        steps_per_epoch = len(self.loader_train)
        n_critic = self.cfg.getint('training', 'n_critic')
        n_save = int(self.cfg.getint('training', 'n_save'))

        # Use tqdm to display a progress bar for the number of epochs
        epochs = tqdm(range(self.epoch, self.cfg.getint('training', 'n_epoch')))

        for epoch in epochs:
            n = 0
            # Initialize the loss for the epoch
            loss_epoch = [[], [], [], [], [], [], []]

            # Initialize the validation data loader
            val_iterator = iter(self.loader_val)

            # Use tqdm to display a progress bar for the training data loader
            tqdm_train_iterator = tqdm(self.loader_train, total=steps_per_epoch, leave=False)

            for train_batch in tqdm_train_iterator:

                # Move the training batch to the device
                train_batch = self.map_to_device(train_batch)
                elems, initial, energy_ndx = train_batch
                elems = transpose_and_zip(elems)

                # If the critic iteration count is reached, train the generator and update the loss
                if n == n_critic:
                    g_loss_dict = self.train_step_gen(elems, initial, energy_ndx)

                    # Log the generator loss to the output
                    for key, value in g_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)

                    # Update the epoch loss
                    for value, l in zip([c_loss] + list(g_loss_dict.values()), loss_epoch):
                        l.append(value)

                    # If a validation data loader exists, validate the generator on it
                    if self.loader_val:
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(self.loader_val)
                            val_batch = next(val_iterator)
                        val_batch = self.map_to_device(val_batch)
                        elems, initial, energy_ndx = val_batch
                        elems = transpose_and_zip(elems)
                        g_loss_dict = self.train_step_gen(elems, initial, energy_ndx, backprop=False)

                        # Log the validation generator loss to the output
                        for key, value in g_loss_dict.items():
                            self.out.add_scalar(key, value, global_step=self.step, mode='val')

                    self.step += 1
                    n = 0

                # If the critic iteration count is not reached, train the critic and update the critic iteration count
                else:
                    c_loss = self.train_step_critic(elems, initial)
                    n += 1

            # Write the average epoch loss to the output
            d_losses = [sum(loss_epoch[i]) / len(loss_epoch[i]) for i in range(1)]
            g_losses = [sum(loss_epoch[i + 1]) / len(loss_epoch[i + 1]) for i in range(6)]
            msg = f"epoch {self.epoch} steps {self.step} : D: {d_losses} G: {g_losses}"
            tqdm.write(msg)

            # Increment the epoch count
            self.epoch += 1

            # If the epoch count is divisible by the save count, save a checkpoint and
            if self.epoch % n_save == 0:
                self.make_checkpoint()
                self.out.prune_checkpoints()
                self.validate()

    def train_step_critic(self, elems, initial):
        # This function is used to perform a single training step on the critic model.

        # Initialize the critic loss to zero.
        c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        # Unpack the initial inputs.
        aa_grid, cg_features, _ = initial

        # Create copies of the atom grids for fake and real atoms.
        fake_atom_grid = aa_grid.clone()
        real_atom_grid = aa_grid.clone()

        # Iterate through each target atom and its features in the input elements.
        for target_atom, target_type, aa_featvec, repl, mask in elems:

            # Prepare input for generator.
            c_fake, c_real = self.prepare_condition(fake_atom_grid, real_atom_grid, aa_featvec, cg_features)

            # Create a tensor of random values with normal distribution.
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # Generate fake atom using the generator.
            fake_atom = self.generator(z, target_type, c_fake)

            # Concatenate the fake atom and fake condition and real atom and real condition.
            fake_data = torch.cat([fake_atom, c_fake], dim=1)
            real_data = torch.cat([target_atom[:, None, :, :, :], c_real], dim=1)

            # Feed the fake and real data through the critic model.
            critic_fake = self.critic(fake_data)
            critic_real = self.critic(real_data)

            # Multiply the critic outputs by the mask.
            critic_fake = torch.squeeze(critic_fake) * mask
            critic_real = torch.squeeze(critic_real) * mask

            # Compute the critic loss.
            c_wass = self.critic_loss(critic_real, critic_fake)
            c_eps = self.epsilon_penalty(1e-3, critic_real)
            c_loss += c_wass + c_eps

            # If gradient penalty is used, compute the gradient penalty and add it to the loss.
            if self.use_gp:
                c_gp = self.gradient_penalty(real_data, fake_data, mask)
                c_loss += c_gp

            # Update the atom grids.
            fake_atom_grid = torch.where(repl[:, :, None, None, None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:, :, None, None, None], real_atom_grid, target_atom[:, None, :, :, :])

        # Reset gradients, perform backward pass, and update critic parameters.
        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        return c_loss.detach().cpu().numpy()

    def train_step_gen(self, elems, initial, energy_ndx, backprop=True):
        # This function is used to perform a single training step on the generator model.


        # Unpack the initial inputs
        aa_grid, cg_features, aa_coords = initial

        # Initialize the generator loss
        g_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        g_loss_dict = {}


        # Clone the initial coordinates and grids for both fake and real cases
        fake_aa_coords = aa_coords.clone()
        real_aa_coords = aa_coords.clone()
        fake_atom_grid = aa_grid.clone()
        real_atom_grid = aa_grid.clone()

        # Loop over the input elements
        for target_atom, target_type, aa_featvec, repl, mask in elems:
            # Prepare input for generator
            fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
            c_fake = fake_aa_features + cg_features

            # Sample random noise vector and generate fake atom
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # Generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            # Compute critic output for the fake atom and apply mask
            critic_fake = self.critic(torch.cat([fake_atom, c_fake], dim=1))

            # mask
            critic_fake = torch.squeeze(critic_fake) * mask

            # Compute generator loss (wasserstein loss)
            g_loss += self.generator_loss(critic_fake)

            # Update coordinates
            new_coord = avg_blob(
            fake_atom,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
            )
            fake_aa_coords = torch.where(repl[:, :, None], fake_aa_coords, new_coord)

            # Update aa grids
            fake_atom_grid = torch.where(repl[:, :, None, None, None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:, :, None, None, None], real_atom_grid, target_atom[:, None, :, :, :])

        # Compute real coordinates from grid instead of using the raw real coordinates for a proper comparison
        # that takes the resolution limit of the grid into account
        real_aa_coords = avg_blob(
            real_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        # Compute the energy terms
        b_energy, a_energy, d_energy, l_energy = self.energy.batch_mean(fake_aa_coords, energy_ndx)

        if self.prior_mode == 'match' and self.prior_weight() > 0.0:
            # Compute the match loss
            b_loss, a_loss, d_loss, l_loss = self.energy.match_loss(real_aa_coords, fake_aa_coords, energy_ndx)
            energy_loss = (self.ratio_bonded_nonbonded*(b_loss + a_loss + d_loss) + l_loss) * self.prior_weight()
            g_loss += energy_loss
            g_loss_dict["Generator/energy_loss"] = energy_loss.detach().cpu().numpy()

        elif self.prior_mode == 'min' and self.prior_weight() > 0.0:
            # Compute the min loss
            energy_loss = (self.ratio_bonded_nonbonded * (b_energy + a_energy + d_energy) + l_energy) * self.prior_weight()
            g_loss += energy_loss
            g_loss_dict["Generator/energy_loss"] = energy_loss.detach().cpu().numpy()

        elif self.prior_mode == 'dstr' and self.prior_weight() > 0.0:
            # Compute the dstr loss
            b_loss, a_loss, d_loss, nb_loss = self.histogram.loss(real_aa_coords, fake_aa_coords, energy_ndx)
            dstr_loss = self.prior_weight() * ((b_loss + a_loss + d_loss) * self.ratio_bonded_nonbonded + nb_loss)
            g_loss += dstr_loss
            g_loss_dict["Generator/histogram_loss"] = dstr_loss.detach().cpu().numpy()

        elif self.prior_mode == 'dstr_abs' and self.prior_weight() > 0.0:
            # Compute the dstr loss with absolut values loss
            b_loss, a_loss, d_loss, nb_loss = self.histogram.abs_loss(real_aa_coords, fake_aa_coords, energy_ndx)
            dstr_loss = self.prior_weight() * ((b_loss + a_loss + d_loss) * self.ratio_bonded_nonbonded + nb_loss)
            g_loss += dstr_loss
            g_loss_dict["Generator/histogram_loss"] = dstr_loss.detach().cpu().numpy()

        if backprop:
            # Reset gradients, perform backward pass, and update critic parameters.
            self.opt_generator.zero_grad()
            g_loss.backward()
            self.opt_generator.step()

        # compute histograms for structural properties every 200 steps and log them
        if self.step % 200 == 0:
            b_dstr_real, a_dstr_real, d_dstr_real, nb_dstr_real = self.histogram.all(real_aa_coords, energy_ndx)
            b_dstr_fake, a_dstr_fake, d_dstr_fake, nb_dstr_fake = self.histogram.all(fake_aa_coords, energy_ndx)
            self.out.add_fig(b_dstr_real, b_dstr_fake, self.histogram.bond, tag="bond", global_step=self.step)
            self.out.add_fig(a_dstr_real, a_dstr_fake, self.histogram.angle, tag="angle", global_step=self.step)
            self.out.add_fig(d_dstr_real, d_dstr_fake, self.histogram.dih, tag="dih", global_step=self.step)
            self.out.add_fig(nb_dstr_real, nb_dstr_fake, self.histogram.nb, tag="nonbonded", global_step=self.step)

        # insert current loss into the generator loss dictionary
        g_loss_dict["Generator/wasserstein"] = g_loss.detach().cpu().numpy()
        g_loss_dict["Generator/energy"] = (b_energy + a_energy + d_energy + l_energy).detach().cpu().numpy()
        g_loss_dict["Generator/energy_bond"] = b_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_angle"] = a_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_dih"] = d_energy.detach().cpu().numpy()
        g_loss_dict["Generator/energy_lj"] = l_energy.detach().cpu().numpy()
        g_loss_dict["Generator/prior_weight"] = self.prior_weight()

        return g_loss_dict


    def predict(self, elems, initial, energy_ndx, bs):

        # Unpack the initial inputs
        aa_grid, cg_features, aa_coords = initial

        # Initialize an empty list to store the generated atoms
        generated_atoms = []

        # Iterate over each element in elems
        for target_type, aa_featvec, repl in zip(*elems):

            # Generate aa features
            fake_aa_features = self.featurize(aa_grid, aa_featvec)

            # Combine aa features with CG features
            c_fake = fake_aa_features + cg_features

            # Repeat the target type bs times to match the batch size
            target_type = target_type.repeat(bs, 1)

            # Generate random noise z to use as input to the generator
            z = torch.empty(
                [target_type.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            # Generate a fake atom using the generator
            fake_atom = self.generator(z, target_type, c_fake)
            generated_atoms.append(fake_atom)

            # Update aa grids
            aa_grid = torch.where(repl[:, :, None, None, None], aa_grid, fake_atom)

            # Compute coordinate of the genrated atom
            new_coord = avg_blob(
            fake_atom,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
            )
            # Insert new coordinate into environment coordinates
            aa_coords = torch.where(repl[:,:,None], aa_coords, new_coord)

        # stack generated atoms
        generated_atoms = torch.cat(generated_atoms, dim=1)

        # Compute coordinates of generated atoms
        generated_atoms_coords = avg_blob(
            generated_atoms,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        # Compute energy of the genrated structure
        b_energy, a_energy, d_energy, l_energy = self.energy.all(aa_coords, energy_ndx)
        energy = b_energy + a_energy + d_energy + l_energy

        return generated_atoms_coords, energy

    def validate(self, samples_dir=None):
        # This method is used to backmap CG structures with the trained model

        if samples_dir:
            # If a samples directory is specified, create a new directory with the given name
            samples_dir = self.out.output_dir / samples_dir
            make_dir(samples_dir)
        else:
            # Otherwise, use the default samples directory
            samples_dir = self.out.samples_dir
        print("Saving samples in {}".format(samples_dir), "...", end='')

        # Create a Stats object to compute statistics of the generated structures
        stats = Stats(self.data, dir=samples_dir / "stats")

        # Get the resolution, delta_s, and sigma values for the grid
        resolution = self.cfg.getint('grid', 'resolution')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma = self.cfg.getfloat('grid', 'sigma')

        # Create the grid and rotation matrices
        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)
        rot_mtxs = torch.from_numpy(rot_mtx_batch(self.val_bs)).to(self.device).float()
        rot_mtxs_transposed = torch.from_numpy(rot_mtx_batch(self.val_bs, transpose=True)).to(self.device).float()

        # Create a list of data generators for the validation set
        data_generators = []

        # Data generators for initial structure
        # Heavy atoms
        data_generators.append(iter(
            Recurrent_Generator(self.data, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False,
                                ref_pos=False)))
        # Hydrogens
        if self.cfg.getboolean('training', 'hydrogens'):
            data_generators.append(iter(
                Recurrent_Generator(self.data, hydrogens=True, gibbs=False, train=False, rand_rot=False, pad_seq=False,
                                    ref_pos=False)))
        # Data generators for gibbs sampling
        for m in range(self.n_gibbs):
            # Heavy atoms
            data_generators.append(iter(
                Recurrent_Generator(self.data, hydrogens=False, gibbs=True, train=False, rand_rot=False, pad_seq=False,
                                    ref_pos=False)))
            # Hydrogens
            if self.cfg.getboolean('training', 'hydrogens'):
                data_generators.append(iter(
                    Recurrent_Generator(self.data, hydrogens=True, gibbs=True, train=False, rand_rot=False,
                                        pad_seq=False, ref_pos=False)))
        m = 0
        try:
            # Set generator and critic to evaluation mode
            self.generator.eval()
            self.critic.eval()


            for data_gen in data_generators:
                for d in data_gen:

                    m += 1

                    with torch.no_grad():

                        # Create batch of AA and CG coordinates, each element in the batch is rotated differently
                        aa_coords = torch.matmul(torch.from_numpy(d['aa_pos']).to(self.device).float(), rot_mtxs)
                        cg_coords = torch.matmul(torch.from_numpy(d['cg_pos']).to(self.device).float(), rot_mtxs)

                        # Map coords to grid
                        aa_grid = to_voxel(aa_coords, grid, sigma)
                        cg_grid = to_voxel(cg_coords, grid, sigma)

                        # Compute Cg features
                        cg_features = torch.from_numpy(d['cg_feat'][None, :, :, None, None, None]).to(
                            self.device) * cg_grid[:, :, None, :, :, :]
                        cg_features = torch.sum(cg_features, 1)

                        # Set initial values
                        initial = (aa_grid, cg_features, aa_coords)

                        # Set elems to iterate over
                        elems = (d['target_type'], d['aa_feat'], d['repl'])
                        elems = transpose(insert_dim(self.to_tensor(elems)))

                        # Set energy ndx to compute energies
                        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])
                        energy_ndx1 = repeat(self.to_tensor(energy_ndx), 1)
                        energy_ndx = repeat(self.to_tensor(energy_ndx), self.val_bs)

                        # Predict new coordinates
                        new_coords, energies = self.predict(elems, initial, energy_ndx, self.val_bs)

                    # based on the validation mode, select predicted coordinates from the batch and insert into the sample
                    if self.val_mode in ['min', 'match']:
                        # Use structure with lowest energy
                        ndx = energies.argmin()
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                    elif self.val_mode == "rm_outlier":
                        # Remove half of the batch with the highest energies, then chose a random structure
                        energies = (-energies).detach().cpu().numpy()
                        sorted_arg = energies.argsort()[int(self.val_bs/2):]
                        ndx = np.random.choice(sorted_arg)
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                    elif self.val_mode == "avg":
                        # Compute the average over all structures (not recommended)
                        new_coords = torch.matmul(new_coords, rot_mtxs_transposed)
                        new_coords = torch.sum(new_coords, 0) / self.val_bs
                    elif self.val_mode == "EM":
                        # Take the structrue with lowest energy and run EM on it before inserting it into the sample
                        ndx = energies.argmin()
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                        new_coords = self.minimize_energy(new_coords,
                                                          torch.from_numpy(d['aa_pos']).to(self.device).float(),
                                                          torch.from_numpy(d['repl']).to(self.device), energy_ndx1,
                                                          1E-8, 20)
                    else:
                        # Just take a random structure from the batch
                        ndx = np.random.randint(self.val_bs)
                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])

                    new_coords = new_coords.detach().cpu().numpy()

                    # insert seected structure into the sample
                    for c, a in zip(new_coords, d['atom_seq']):
                        a.pos = d['loc_env'].rot_back(c)

            # Save sample
            subdir = "ep" + str(self.epoch) + "_valbs" + str(self.val_bs) + "_gibbs" + str(self.n_gibbs) + self.prior_mode + self.val_mode
            stats.save_samples(train=False, subdir=subdir)

            # if validate is set to true in cfg file, compute statistics of the generated sample
            if self.cfg.getboolean('validate', 'evaluate'):
                stats.evaluate(train=False, subdir=subdir)

            # Reset atom positions to start from scratch for next evaluation
            for sample in self.data.samples_val:
                sample.kick_atoms()

        finally:
            # Set models back to training mode
            self.generator.train()
            self.critic.train()

    def shift_loss(self, coords, mus):
        dis_sq = (coords - mus) ** 2
        e = torch.sum(dis_sq)
        return e

    def minimize_energy(self, var_coords, loc_env_coords, repls, energy_ndx, delta, iterations):
        # Method to perform EM on generated fragment

        # create torch variable for coordinates which we want to use for EM
        var_coords = Variable(var_coords, requires_grad=True)

        # save starting positions
        start_pos = var_coords.clone()

        # use gradient descent
        opt_em = SGD([var_coords], lr=delta)

        for i in range(0, iterations):

            # update coordinates of atoms in local environment to compute energy
            for repl, var_coord in zip(repls, var_coords):
                loc_env_coords = torch.where(repl[:, None], loc_env_coords, var_coord[None, :])
            # compute energy of local environment
            b_energy, a_energy, d_energy, l_energy = self.energy.all(loc_env_coords[None, :, :], energy_ndx)
            # compute toal energy and add shift loss
            energy = b_energy + a_energy + d_energy + l_energy + 1000000 * self.shift_loss(var_coords, start_pos)

            if i == 0:
                start_energy = energy.clone()
            if i == 0 or i == iterations - 1:
                print("BOND: ", b_energy)
                print("ANGLE: ", a_energy)
                print("DIH: ", d_energy)
                print("LJ: ", l_energy)
                print("MODEL: ", self.shift_loss(var_coords, start_pos))

            # gradient descent
            opt_em.zero_grad()
            energy.backward(retain_graph=True)
            opt_em.step()

        # only use new positions if end energy is lower than starting energy
        if energy > start_energy:
            return start_pos
        else:
            return var_coords
