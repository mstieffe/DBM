import os
import argparse
#from xml.dom import minidom
import xml.etree.ElementTree as ET 


parser = argparse.ArgumentParser()
parser.add_argument("aa_itp")
parser.add_argument("xml")
parser.add_argument("out")
args = parser.parse_args()
aa_itp = args.aa_itp
xml = args.xml
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
    
#xml_parser = minidom.parse(xml)
#beads = xml_parser.getElementsByTagName('cg_bead')
#print(beads[0].childNodes[1].text)
#print(beads[1].attributes['name'].value)

class Atom():
    def __init__(self, atom_ndx, res, name, bead_ndx, bead_type):
        self.atom_ndx = atom_ndx
        self.res = res
        self.name = name
        self.bead_ndx = bead_ndx
        self.bead_type = bead_type

tree = ET.parse(xml) 
root = tree.getroot() 
atoms = []

#go through all atoms in itp file
for line in read_between("[atoms]", "[", aa_itp):
    atom_ndx = int(line.split()[0])
    atom_name2 = str(line.split()[4])
    res_name2 = str(line.split()[3])
    
    n=1
    #for each atom, go through all cg beads in the xml file
    for bead in root[2][0].findall('cg_bead'):
        bead_type = bead.findall('type')[0].text
        bead_atoms = bead.findall('beads')[0].text
        #print(bead_type)

        #go through all atoms associated with each bead
        for name in bead_atoms.split():
            atom_name = name.split(':')[2]
            res_name = name.split(':')[1]
               
            #if atom name and res name match with atom in itp file -> append result to atom list
            if atom_name==atom_name2 and res_name==res_name2:
                atoms.append(Atom(atom_ndx, res_name, atom_name, n, bead_type))
        n += 1

with open(out, 'w') as out:
    
    #write first two lines as they are
    out.write("[map]\n")
    for a in atoms:
        out.write("{}\t{}\t{}\t{}\n".format(a.atom_ndx, a.name, a.bead_ndx, a.bead_type))
    out.write("[\\map]\n\n")
    
    out.write("[align]\n")
    for m in range(1,n+1):
        out.write("{}\t{}\n".format(m, 1))
    out.write("[\\align]\n\n")
    
    out.write("[mult]\n")
    for m in range(1,n+1):
        out.write("{}\t{}\n".format(m, 1))
    out.write("[\\mult]\n")
    
    out.close()
#print(atoms)
#print(root[1].text)


#for line in read_between("<beads>", "</beads>", xml):
#    print(line)
    


    #if len(line.split()) >= 2:
    #    index1 = int(line.split()[0]) - 1
    #    index2 = int(line.split()[1]) - 1
