import pickle
from timeit import default_timer as timer
from dbm.ff import FF
from dbm.universe import Universe
from pathlib import Path
import itertools


class Data():
    """
    The class contains all necessary data for the training and validation of the ML model.
    """
    def __init__(self, cfg, save=True):
        """
        Initializes a Data object.

        Parameters:
            - cfg (ConfigParser): configuration object with the parameters to configure the object
            - save (bool): whether to save the created samples

        Returns:
            None
        """

        # Start timing the execution of the data creation
        start = timer()

        # Extract and store the configuration data for the universe.
        self.cfg = cfg
        self.aug = int(cfg.getboolean('universe', 'aug'))
        self.align = int(cfg.getboolean('universe', 'align'))
        self.order = cfg.get('universe', 'order')
        self.cutoff = cfg.getfloat('universe', 'cutoff')
        self.kick = cfg.getfloat('universe', 'kick')

        # Load the forcefield file specified in the configuration and store the forcefield object.
        self.ff_name = cfg.get('forcefield', 'ff_file')
        self.ff_path = Path("./data/forcefield") / self.ff_name
        self.ff = FF(self.ff_path)

        # Create a string to append to the file name when saving samples.
        self.desc = '_aug={}_align={}_order={}_cutoff={}_kick={}_ff={}.pkl'.format(self.aug,
                                                                              self.align,
                                                                              self.order,
                                                                              self.cutoff,
                                                                              self.kick,
                                                                              self.ff_name)

        # Set up directories for loading data and storing processed data.
        self.data_dir = Path("./data/")
        self.dirs_train = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'train_data').split(",")]
        self.dirs_val = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'val_data').split(",")]
        self.dir_processed = Path("./data/processed")
        self.dir_processed.mkdir(exist_ok=True)

        # Load the training and validation samples into dictionaries.
        # Each key in the dictionary corresponds to a directory containing samples.
        # Each value is a list of tuples containing the sample data and the sample name.
        print("preparing training data...")
        self.dict_train, self.dict_val = {}, {}
        for path in self.dirs_train:
            self.dict_train[path.stem] = self.get_samples(path, save=save)
        self.samples_train = list(itertools.chain.from_iterable(self.dict_train.values()))
        # Print information about the processed samples
        self.print_sample_info(train=True)

        print("preparing validation data...")
        for path in self.dirs_val:
            self.dict_val[path.stem] = self.get_samples(path, save=save)
        self.samples_val = list(itertools.chain.from_iterable(self.dict_val.values()))
        # Print information about the processed samples
        self.print_sample_info(train=False)

        # Find the maximum values for each feature in the samples for padding.
        self.max = self.get_max_dict()

        # Print the time needed to create the data object
        print("Successfully processed data! This took ", timer()-start, "secs")



    def get_samples(self, path, save=False):
        """
        This method generates samples for training and validation data.

        Parameters:
            - path (str): directory path containing CG and AA files.
            - save (bool): flag indicating whether to save the created samples or not.

        Returns:
            - samples (list): list of Universe objects.
        """

        # Define the processed file path
        name = path.stem + self.desc
        processed_path = self.dir_processed / name

        # Check if the processed file already exists
        if processed_path.exists():
            # If the processed file exists, load the Universe objects from the file
            with open(processed_path, 'rb') as input:
                samples = pickle.load(input)
            print("Loaded train universe from " + str(processed_path))
        else:
            # If the processed file does not exist, create the Universe objects
            samples = []
            cg_dir = path / "cg"
            aa_dir = path / "aa"
            for cg_path in cg_dir.glob('*.gro'):
                aa_path = aa_dir / cg_path.name
                path_dict = {'data_dir': self.data_dir, 'cg_path': cg_path, 'file_name': cg_path.stem}
                if aa_path.exists():
                    path_dict['aa_path'] = aa_path
                else:
                    path_dict['aa_path'] = None
                u = Universe(self.cfg, path_dict, self.ff)
                samples.append(u)
            # Save the samples to the processed file
            if save and not processed_path.exists():
                with open(processed_path, 'wb') as output:
                    pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)
        return samples

    def print_sample_info(self, train=True):
        if train:
            samples = self.samples_train
        else:
            samples = self.samples_val
        for sample in samples:
            unique_mols, mol_names = [], []
            for mol in sample.mols:
                if mol.name not in mol_names:
                    unique_mols.append(mol)
                    mol_names.append(mol.name)
            msg = "Found {} molecules in sample {}".format(len(mol_names), sample.name)
            print(msg)
            for mol in unique_mols:
                msg = "Molecule {} contains {} atoms and {} beads.".format(mol.name, len(mol.atoms), len(mol.beads))
                print(msg)
                msg = "The AA topology has {} bonds, {} angles, {} dihedrals".format(len(mol.bonds),
                                                                                     len(mol.angles),
                                                                                     len(mol.dihs)+len(mol.dihs_rb))
                print(msg)
                msg = "The CG topology has {} bonds".format(len(mol.cg_edges))
                print(msg)
                bead_types = set([b.type.name for b in mol.beads])
                atom_types = set([a.type.name for a in mol.atoms])
                msg = "The molecule contains {} atom types: {}".format(len(atom_types), atom_types)
                print(msg)
                msg = "The molecule contains {} beads types: {}".format(len(bead_types), bead_types)
                print(msg)
                order = [o[0] for o in mol.cg_seq(order=sample.order, train=False)]
                order = [b.index for b in order]
                msg = "The CG order of reconstruction is (bead indices): {}".format(order)
                print(msg)



    def get_max_dict(self):
        """
        Get the maximum values of various features across all training and validation samples.
        Parameters: None.
        Returns: A dictionary containing the maximum values of various features across all training and validation samples.
        """

        # Define a list of keys that will be used in the dictionary
        keys = ['seq_len',
                'beads_loc_env',
                'atoms_loc_env',
                'bonds_per_atom',
                'angles_per_atom',
                'dihs_per_atom',
                'dihs_rb_per_atom',
                'ljs_per_atom',
                'bonds_per_bead',
                'angles_per_bead',
                'dihs_per_bead',
                'dihs_rb_per_bead',
                'ljs_per_bead']
        # Create a dictionary with initial values of 0 for each key
        max_dict = dict([(key, 0) for key in keys])

        # Combine training and validation samples
        samples = self.samples_train + self.samples_val

        # Loop over each sample in the combined list of samples
        for sample in samples:
            # Loop over each bead in the sample
            for bead in sample.beads:
                # Update the max_dict values for seq_len, beads_loc_env, and atoms_loc_env
                max_dict['seq_len'] = max(len(sample.aa_seq_heavy[bead]), len(sample.aa_seq_hydrogens[bead]), max_dict['seq_len'])
                max_dict['beads_loc_env'] = max(len(sample.loc_envs[bead].beads), max_dict['beads_loc_env'])
                max_dict['atoms_loc_env'] = max(len(sample.loc_envs[bead].atoms), max_dict['atoms_loc_env'])

                # Loop over heavy and hydrogen aa sequences for each bead
                for aa_seq in [sample.aa_seq_heavy[bead], sample.aa_seq_hydrogens[bead]]:
                    bonds_ndx, angles_ndx, dihs_ndx, dihs_rb_ndx, ljs_ndx = [], [], [], [], []
                    # Loop over each atom in the aa sequence
                    for atom in aa_seq:
                        # Extract the features of the current atom
                        f = sample.aa_features[atom]
                        # Update the max_dict values
                        max_dict['bonds_per_atom'] = max(len(f.energy_ndx_gibbs['bonds']), max_dict['bonds_per_atom'])
                        max_dict['angles_per_atom'] = max(len(f.energy_ndx_gibbs['angles']), max_dict['angles_per_atom'])
                        max_dict['dihs_per_atom'] = max(len(f.energy_ndx_gibbs['dihs']), max_dict['dihs_per_atom'])
                        max_dict['dihs_rb_per_atom'] = max(len(f.energy_ndx_gibbs['dihs_rb']), max_dict['dihs_rb_per_atom'])
                        max_dict['ljs_per_atom'] = max(len(f.energy_ndx_gibbs['ljs']), max_dict['ljs_per_atom'])
                        # Append the energy indices of the current atom to the relevant lists
                        bonds_ndx += f.energy_ndx_gibbs['bonds']
                        angles_ndx += f.energy_ndx_gibbs['angles']
                        dihs_ndx += f.energy_ndx_gibbs['dihs']
                        dihs_rb_ndx += f.energy_ndx_gibbs['dihs_rb']
                        ljs_ndx += f.energy_ndx_gibbs['ljs']
                    # Update the max_dict values for per bead quantities
                    max_dict['bonds_per_bead'] = max(len(set(bonds_ndx)), max_dict['bonds_per_bead'])
                    max_dict['angles_per_bead'] = max(len(set(angles_ndx)), max_dict['angles_per_bead'])
                    max_dict['dihs_per_bead'] = max(len(set(dihs_ndx)), max_dict['dihs_per_bead'])
                    max_dict['dihs_rb_per_bead'] = max(len(set(dihs_rb_ndx)), max_dict['dihs_rb_per_bead'])
                    max_dict['ljs_per_bead'] = max(len(set(ljs_ndx)), max_dict['ljs_per_bead'])

        return max_dict
