import os
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("templ")
parser.add_argument("name")
parser.add_argument("aa_itp")
parser.add_argument("cg_itp")
parser.add_argument("out")
args = parser.parse_args()
templ = args.templ
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
    for line in read_between("[atom_types]", "[", templ):
        out.write(line)
    out.write("[\\atom_types]\n")
    out.write("\n")
    
    out.write("[bond_types]\n")
    out.write(";i\tj\tchannel\tfunc\tb0\tkb\n")
    for line in read_between("[bond_types]", "[", templ):
        out.write(line)
    out.write("[\\bond_types]\n")
    out.write("\n")

    out.write("[angle_types]\n")
    out.write(";i\tj\tk\tchannel\tfunc\tb0\tkb\n")
    for line in read_between("[angle_types]", "[", templ):
        out.write(line)
    out.write("[\\angle_types]\n")
    out.write("\n")

    out.write("[dihedral_types]\n")
    out.write(";i\tj\tk\tl\tchannel\tfunc\tparams\n")
    for line in read_between("[dihedral_types]", "[", templ):
        out.write(line)
    out.write("[\\dihedral_types]\n")
    out.write("\n")

    out.write("[lj_types]\n")
    out.write(";i\tj\tchannel\n")
    for line in read_between("[lj_types]", "[", templ):
        out.write(line)
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
