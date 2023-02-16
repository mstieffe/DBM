import os
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("inp")
parser.add_argument("name")
parser.add_argument("aa_itp")
parser.add_argument("cg_itp")
parser.add_argument("out")
args = parser.parse_args()
inp = args.inp
name = args.name
aa_itp = args.aa_itp
cg_itp = args.cg_itp
out = args.out

def read_between(start, end, file):
    # generator to yield line between start and end
    file = open(file)
    rec = False
    for line in file:
        if line.startswith(";") or line.startswith("\n"):
            continue
        if not rec:
            if line.startswith(start):
                rec = True
        elif line.startswith(end):
            rec = False
        else:
            yield line
    file.close()
    


with open(out, 'w') as out:
    
    out.write("[general]\n")
    out.write(";name\tnrexcl\n")
    for line in read_between("[moleculetype]", "[", aa_itp):
        out.write("{}\t{}\n".format(name, line.split()[1]))
    out.write("[\\general]\n")
    out.write("\n")

    
    out.write("[atom_types]\n")
    out.write(";name\tchannel\tmass\tcharge\tsigma\tepsilon\n")
    for line in read_between("[ atomtypes ]", "[", inp):
        out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line.split()[0], -1, line.split()[1], line.split()[2], line.split()[4], line.split()[5]))
    out.write("[\\atom_types]\n")
    out.write("\n")
    
    out.write("[bond_types]\n")
    out.write(";i\tj\tchannel\tfunc\tb0\tkb\n")
    for line in read_between("[ bondtypes ]", "[", inp):
        out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line.split()[0], line.split()[1], 0, line.split()[2], line.split()[3], line.split()[4]))
    out.write("[\\bond_types]\n")
    out.write("\n")

    out.write("[angle_types]\n")
    out.write(";i\tj\tk\tchannel\tfunc\tb0\tkb\n")
    for line in read_between("[ angletypes ]", "[", inp):
        out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line.split()[0], line.split()[1], line.split()[2], 0, line.split()[3], line.split()[4], line.split()[5]))
    out.write("[\\angle_types]\n")
    out.write("\n")

    out.write("[dihedral_types]\n")
    out.write(";i\tj\tk\tl\tchannel\tfunc\tparams\n")
    for line in read_between("[ dihedraltypes ]", "[", inp):
        params = "\t".join(line.split()[4:])
        params = params.split(";")[0]
        out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line.split()[0], line.split()[1], line.split()[2], line.split()[3], 0, params))
    out.write("[\\dihedral_types]\n")
    out.write("\n")
        
    atoms = []
    for line in read_between("[ atomtypes ]", "[", inp):
        atoms.append(line.split()[0])
    atoms2 = deepcopy(atoms)
    out.write("[lj_types]\n")
    out.write(";i\tj\tchannel\n")
    for a in atoms:
        for b in atoms2:
            out.write("{}\t{}\t{}\n".format(a, b, 0))
        atoms2 = atoms2[1:]
    out.write("[\\lj_types]\n")
    out.write("\n")
    
    out.write("[bead_types]\n")
    out.write(";name\tchannel\n")
    for line in read_between("[atomtypes]", "[", cg_itp):
        out.write("{}\t{}\n".format(line.split()[0], 0))
    out.write("[\\bead_types]\n")
    
    out.close()
#print(atoms)
#print(root[1].text)


#for line in read_between("<beads>", "</beads>", xml):
#    print(line)
    


    #if len(line.split()) >= 2:
    #    index1 = int(line.split()[0]) - 1
    #    index2 = int(line.split()[1]) - 1
