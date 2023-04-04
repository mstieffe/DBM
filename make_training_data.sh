#!/bin/bash

data_set_name=Mat1653
res_name=M1653 #max 5 characters plz, don't start with a number
path_to_xml=../MERCK/OPLS/Mat-1653/CG_Mapping/Mat-1653_four_beadtypes.xml
path_to_aa_tpr=../MERCK/OPLS/Mat-1653/MD_Atomistic/NVT_530K/nvt.tpr
path_to_aa_trr=../MERCK/OPLS/Mat-1653/MD_Atomistic/NVT_530K/nvt.trr
path_to_aa_top=../MERCK/OPLS/Mat-1653/MD_Atomistic/NVT_530K/topol.top
path_to_aa_ff=../MERCK/OPLS/Mat-1653/MD_Atomistic/NVT_530K/-
path_to_cg_tpr=../MERCK/OPLS/Mat-1653/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_530K/NVT_530K/nvt.tpr
path_to_cg_top=../MERCK/OPLS/Mat-1653/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_530K/NVT_530K/topol.top
path_to_cg_dir_deposition=../MERCK/OPLS/Mat-1653/Deposition/IBI_onlynonbonded_bonded_melt/Potential_530K/Parallel_Runs_Rate_10_10_K_s_ts_10fs_tau0.5ps/Slurm_Submission/T250K_depositionbydensity/Structures_1600_deposited_molecules

n_frames_train=2
n_frames_val=1

####################################################################################################################################
#nothing to do below
####################################################################################################################################

#total number of frames
n_frames=$((n_frames_train + n_frames_val))

#make directories
mkdir -p data configs
cd data
mkdir -p reference_snapshots aa_top cg_top mapping forcefield
cd reference_snapshots
mkdir -p $data_set_name 
cd $data_set_name
mkdir -p train val deposition
cd train
mkdir -p aa aa_temp cg cg_temp
cd ../val
mkdir -p aa cg
cd ../deposition
mkdir -p aa cg cg_temp
cd ../../../..

#define file paths for DBM
path_to_aa_snapshots=data/reference_snapshots/$data_set_name/train/aa
path_to_aa_snapshots_val=data/reference_snapshots/$data_set_name/val/aa
path_to_aa_snapshots_depo=data/reference_snapshots/$data_set_name/deposition/aa
path_to_aa_temp_snapshots=data/reference_snapshots/$data_set_name/train/aa_temp
path_to_cg_snapshots=data/reference_snapshots/$data_set_name/train/cg
path_to_cg_snapshots_val=data/reference_snapshots/$data_set_name/val/cg
path_to_cg_snapshots_depo=data/reference_snapshots/$data_set_name/deposition/cg
path_to_cg_snapshots_depo_temp=data/reference_snapshots/$data_set_name/deposition/cg_temp
path_to_cg_temp_snapshots=data/reference_snapshots/$data_set_name/train/cg_temp

#adjust formatting of aa itp file
cp $path_to_aa_top data/aa_top/$res_name.itp
gawk -i inplace 'NF' data/aa_top/$res_name.itp
sed -i s/".*\[ "/"\["/ data/aa_top/$res_name.itp
sed -i s/" \]"/"\]"/ data/aa_top/$res_name.itp
gawk -i inplace '/dihedrals/&&c++>0 {next} 1' data/aa_top/$res_name.itp
gawk -i inplace '/exclusions/&&c++>0 {next} 1' data/aa_top/$res_name.itp
gawk -i inplace '/pairs/&&c++>0 {next} 1' data/aa_top/$res_name.itp
gawk -i inplace '/angles/&&c++>0 {next} 1' data/aa_top/$res_name.itp
gawk -i inplace '/bonds/&&c++>0 {next} 1' data/aa_top/$res_name.itp
sed -i -e '$a\\[\]' data/aa_top/$res_name.itp

#adjust formatting of cg itp file
cp $path_to_cg_top data/cg_top/$res_name.itp
gawk -i inplace 'NF' data/cg_top/$res_name.itp
sed -i s/".*\[ "/"\["/ data/cg_top/$res_name.itp
sed -i s/" \]"/"\]"/ data/cg_top/$res_name.itp
gawk -i inplace '/dihedrals/&&c++>0 {next} 1' data/cg_top/$res_name.itp
gawk -i inplace '/exclusions/&&c++>0 {next} 1' data/cg_top/$res_name.itp
gawk -i inplace '/pairs/&&c++>0 {next} 1' data/cg_top/$res_name.itp
gawk -i inplace '/angles/&&c++>0 {next} 1' data/cg_top/$res_name.itp
gawk -i inplace '/bonds/&&c++>0 {next} 1' data/cg_top/$res_name.itp
sed -i -e '$a\\[\]' data/cg_top/$res_name.itp


#map atomistic trajectory to CG trajectory
csg_map --cg $path_to_xml --top $path_to_aa_tpr --trj $path_to_aa_trr --force --out $path_to_cg_temp_snapshots/cg.trr

# Get the total number of frames in the TRR file
n_frames_total=$(gmx check -f $path_to_cg_temp_snapshots/cg.trr 2>&1 > /dev/null | grep -o 'Last.*' | awk '{print $3}')

# Calculate the number of frames that we skip between extracted frames
skip=$(( (n_frames_total / n_frames)+1 ))

#write out frames CG
echo 0 | gmx trjconv -f $path_to_cg_temp_snapshots/cg.trr -s $path_to_cg_tpr -sep -o $path_to_cg_temp_snapshots/.gro -skip $skip 

#delete CG trr (we only keep .gro files)
rm $path_to_cg_temp_snapshots/cg.trr

#rename CG residues
python ./dbm/rename_residues.py $path_to_cg_temp_snapshots $path_to_cg_snapshots $res_name data/cg_top/$res_name.itp
rm -r $path_to_cg_temp_snapshots

#write out frames AA
echo 0 | gmx trjconv -f $path_to_aa_trr -s $path_to_aa_tpr -sep -o $path_to_aa_temp_snapshots/.gro -skip $skip 

#rename and renumber AA residues
python ./dbm/rename_residues.py $path_to_aa_temp_snapshots $path_to_aa_snapshots $res_name data/aa_top/$res_name.itp
rm -r $path_to_aa_temp_snapshots

#move a fraction of the files to the validation directory
#get the list of files sorted by modification time in descending order and only take the latest n files
val_files=$(ls -1t $path_to_aa_snapshots | head -$n_frames_val)
# Move each file to the destination directory
for file in $val_files; do
  mv $path_to_aa_snapshots/$file $path_to_aa_snapshots_val/$file
  mv $path_to_cg_snapshots/$file $path_to_cg_snapshots_val/$file
done

#make mapping file
python ./dbm/create_mapping.py data/aa_top/$res_name.itp data/cg_top/$res_name.itp $path_to_xml data/mapping/$res_name.map

#make ff file
if [[ -f $path_to_aa_ff ]]; then
python ./dbm/create_ff.py $path_to_aa_ff $res_name data/aa_top/$res_name.itp data/cg_top/$res_name.itp data/forcefield/$res_name.txt
else
python ./dbm/create_ff_from_template.py ./data/forcefield/template.txt $res_name data/aa_top/$res_name.itp data/cg_top/$res_name.itp data/forcefield/$res_name.txt
fi

#python ./dbm/create_ff.py $path_to_aa_ff $res_name data/aa_top/$res_name.itp data/cg_top/$res_name.itp data/forcefield/$res_name.txt || python ./dbm/create_ff_from_template.py ./data/forcefield/template.txt $res_name data/aa_top/$res_name.itp data/cg_top/$res_name.itp data/forcefield/$res_name.txt

#make deposition data
for f in $path_to_cg_dir_deposition/*.gro; do
	file_basename=$(basename $f)
	cp $f $path_to_cg_snapshots_depo_temp/$file_basename
done

#rename CG residues in deposition directory
python ./dbm/rename_residues.py $path_to_cg_snapshots_depo_temp $path_to_cg_snapshots_depo $res_name data/cg_top/$res_name.itp
rm -r $path_to_cg_snapshots_depo_temp

cat > ./configs/$res_name.ini << EOL
[model]
name = ${res_name}
output_dir = results 
model_type = tiny
n_chns = 64
noise_dim = 64
sn_gen = 0
sn_crit = 1
gp = True

[forcefield]
ff_file = ${res_name}.txt

[universe]
aug = True
align = True
order = dfs
cutoff = 0.6
kick = 0.05

[data]
train_data = ${data_set_name}/train
val_data = ${data_set_name}/val

[training]
recurrent = True
n_epoch = 50
rand_rot = True
batchsize = 64
n_critic = 4
n_checkpoints = 10
n_save = 5 
hydrogens = True

[prior]
mode = none 
ratio_bonded_nonbonded = 0.1
weights = 0.0
schedule = 

[grid]
resolution = 8
length = 1.2
sigma = 0.02

[validate]
n_gibbs = 1
batchsize = 32
evaluate = False
EOL






