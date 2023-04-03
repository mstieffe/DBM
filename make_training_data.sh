name=BCzPh
res_name=BCZ #only three characters plz
path_to_xml=BCzPh_Mat-917_MERCK/CG_Mapping/BCzPh_three_beadtypes.xml
path_to_aa_tpr=BCzPh_Mat-917_MERCK/MD_Atomistic/NVT_520K/nvt.tpr
path_to_aa_trr=BCzPh_Mat-917_MERCK/MD_Atomistic/NVT_520K/nvt.trr
path_to_aa_top=BCzPh_Mat-917_MERCK/MD_Atomistic/NVT_520K/topol.top
path_to_aa_ff=BCzPh_Mat-917_MERCK/MD_Atomistic/NVT_520K/-
path_to_cg_tpr=BCzPh_Mat-917_MERCK/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_520K/NVT_520K/nvt.tpr
path_to_cg_top=BCzPh_Mat-917_MERCK/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_520K/NVT_520K/topol.top
skip=100 #number of frames to skip in AA trajectory


#nothing to do below
####################################################################################################################################

#make directories
mkdir -p data
cd data
mkdir -p reference_snapshots aa_top cg_top mapping forcefield
cd reference_snapshots
mkdir -p $name
cd $name
mkdir -p aa aa_temp cg cg_temp
cd ../../..

#define file paths for DBM
path_to_aa_snapshots=data/reference_snapshots/$name/aa
path_to_aa_temp_snapshots=data/reference_snapshots/$name/aa_temp
path_to_cg_snapshots=data/reference_snapshots/$name/cg
path_to_cg_temp_snapshots=data/reference_snapshots/$name/cg_temp

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
sed -i -e '$a\\[\]' data/cg_top/$name.itp

#map atomistic trajectory to CG trajectory
csg_map --cg $path_to_xml --top $path_to_aa_tpr --trj $path_to_aa_trr --force --out $path_to_cg_temp_snapshots/cg.trr

#write out frames CG
gmx trjconv -f $path_to_cg_temp_snapshots/cg.trr -s $path_to_cg_tpr -sep -o $path_to_cg_temp_snapshots/.gro -skip $skip

#delete CG trr (we only keep .gro files)
rm $path_to_cg_temp_snapshots/cg.trr

#rename CG residues
python rename_residues.py $path_to_cg_temp_snapshots $path_to_cg_snapshots $res_name data/cg_top/$res_name.itp
rm -r $path_to_cg_temp_snapshots

#write out frames AA
gmx trjconv -f $path_to_aa_trr -s $path_to_aa_tpr -sep -o $path_to_aa_temp_snapshots/.gro -skip $skip

#rename and renumber AA residues
python rename_residues.py $path_to_aa_temp_snapshots $path_to_aa_snapshots $res_name data/aa_top/$res_name.itp
rm -r $path_to_aa_temp_snapshots

#make mapping file
python create_mapping.py data/aa_top/$res_name.itp data/cg_top/$res_name.itp $path_to_xml data/mapping/$res_name.map

#make ff file
python create_ff.py $path_to_aa_ff $res_name data/aa_top/$res_name.itp data/cg_top/$res_name.itp data/forcefield/$res_name.txt






