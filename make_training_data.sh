#Files
name=BCzPh_DDEC
res_name=BCZ #only three characters plz
path_to_xml=BCzPh_DDEC/CG_Mapping/BCzPh_three_beadtypes.xml
path_to_aa_tpr=BCzPh_DDEC/MD_Atomistic/NPT_470K/npt.tpr
path_to_aa_trr=BCzPh_DDEC/MD_Atomistic/NPT_470K/npt.trr
path_to_aa_top=BCzPh_DDEC/MD_Atomistic/NPT_470K/BCzPh-DDEC.top
path_to_aa_ff=forcefield/BCzPh-DDEC.ff/forcefield.itp
path_to_cg_tpr=BCzPh_DDEC/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_470K/NVT_470K/nvt.tpr
path_to_cg_top=BCzPh_DDEC/MD_CG/IBI_onlynonbonded_bonded_melt/Potential_470K/NVT_470K/topol.top
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
mkdir -p aa aa_temp cg
cd ..
cd ..
cd ..

#define file paths for DBM
path_to_aa_snapshots=data/reference_snapshots/$name/aa
path_to_aa_temp_snapshots=data/reference_snapshots/$name/aa_temp
path_to_cg_snapshots=data/reference_snapshots/$name/cg

#adjust formatting of aa itp file
cp $path_to_aa_top data/aa_top/$name.itp
gawk -i inplace 'NF' data/aa_top/$name.itp
sed -i s/"\[ "/"\["/ data/aa_top/$name.itp
sed -i s/" \]"/"\]"/ data/aa_top/$name.itp
gawk -i inplace '/dihedrals/&&c++>0 {next} 1' data/aa_top/$name.itp
gawk -i inplace '/exclusions/&&c++>0 {next} 1' data/aa_top/$name.itp
gawk -i inplace '/pairs/&&c++>0 {next} 1' data/aa_top/$name.itp
gawk -i inplace '/angles/&&c++>0 {next} 1' data/aa_top/$name.itp
gawk -i inplace '/bonds/&&c++>0 {next} 1' data/aa_top/$name.itp
sed -i -e '[]' data/aa_top/$name.itp

#adjust formatting of cg itp file
cp $path_to_cg_top data/cg_top/$name.itp
gawk -i inplace 'NF' data/cg_top/$name.itp
sed -i s/"\[ "/"\["/ data/cg_top/$name.itp
sed -i s/" \]"/"\]"/ data/cg_top/$name.itp
gawk -i inplace '/dihedrals/&&c++>0 {next} 1' data/cg_top/$name.itp
gawk -i inplace '/exclusions/&&c++>0 {next} 1' data/cg_top/$name.itp
gawk -i inplace '/pairs/&&c++>0 {next} 1' data/cg_top/$name.itp
gawk -i inplace '/angles/&&c++>0 {next} 1' data/cg_top/$name.itp
gawk -i inplace '/bonds/&&c++>0 {next} 1' data/cg_top/$name.itp
sed -i -e '[]' data/cg_top/$name.itp

#cp $path_to_aa_ff data/forcefield/$name.txt

#map atomistic trajectory to CG trajectory
csg_map --cg $path_to_xml --top $path_to_aa_tpr --trj $path_to_aa_trr --force --out $path_to_cg_snapshots/cg.trr

#write out frames CG
gmx trjconv -f $path_to_cg_snapshots/cg.trr -s $path_to_cg_tpr -sep -o $path_to_cg_snapshots/.gro -skip $skip

#delete CG trr (we only keep .gro files)
rm $path_to_cg_snapshots/cg.trr

#rename CG residues
sed -i s/'BCzPh'/"${res_name}   "/g $path_to_cg_snapshots/*.gro

#write out frames AA
gmx trjconv -f $path_to_aa_trr -s $path_to_aa_tpr -sep -o $path_to_aa_temp_snapshots/.gro -skip $skip

#rename and renumber AA residues
python rename_residues.py $path_to_aa_temp_snapshots $path_to_aa_snapshots $res_name 62
rm -r $path_to_aa_temp_snapshots

#make mapping file
python create_mapping.py data/aa_top/$name.itp $path_to_xml data/mapping/$name.map

#make ff file
python create_ff.py $path_to_aa_ff $name data/aa_top/$name.itp data/cg_top/$name.itp data/forcefield/$name.txt






