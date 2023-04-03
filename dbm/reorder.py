import os
import argparse

#python reorder.py conf_1.gro 1 2 5 3 4 7 6 8 10 11 12 9

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("order", nargs="+")
args = parser.parse_args()
file_name = args.file
order = list(args.order)
order = [int(o) for o in order]
n_beads = len(order)

def pad_index(index, max_len=5):
    s = str(index)
    for i in range(max_len-len(s)):
        s = " "+s
    return s

with open(file_name, errors='ignore') as file:
    lines = file.readlines()
  
    with open(os.path.splitext(os.path.basename(file_name))[0]+"_reordered.gro", 'w') as out:
    
        #write first two lines as they are
        out.write(lines[0])
        out.write(lines[1])
        
        n_mols = int(len(lines[2:-1]) / n_beads)
        
        print("number of mols: ", n_mols)
        
        index = 1
        for n in range(n_mols):
            mol_lines = lines[2 + n*n_beads:2 + (n+1)*n_beads]
            mol_lines =  [mol_lines[o-1] for o in order]
        
            for line in mol_lines:
                #print(line[15:20])
                line = line[:15] + pad_index(index) + line[20:]
                out.write(line)
                index += 1
        
        out.write(lines[-1])

file.close()
        