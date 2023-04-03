import os
import argparse

#def make_dir(path):
#    if not os.path.exists(path):
#        os.makedirs(path)

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

#res_name = "BCZ"
#n_atoms = 62

parser = argparse.ArgumentParser()
parser.add_argument("dir_old")
parser.add_argument("dir_new")
parser.add_argument("res_name")
parser.add_argument("aa_itp")
args = parser.parse_args()
directory_old = args.dir_old
directory_new = args.dir_new
res_name = args.res_name
aa_itp = args.aa_itp

n_atoms = len(list(read_between("[atoms]", "[", aa_itp)))

#outdir = os.path.dirname(directory)+"/frames_renamed"
#make_dir(outdir)


for file_name in os.listdir(directory_old):
    #print(directory+"/"+file)
    with open(directory_old+"/"+file_name, errors='ignore') as file:
        lines = file.readlines()
      
        
        n = 0
        res_n = 1
        with open(os.path.join(directory_new,file_name), 'w') as out:
            
            #write first two lines as they are
            out.write(lines[0])
            out.write(lines[1])
                   
            for line in lines[2:-1]:
                if res_n < 10:
                    line = "    "+str(res_n)+res_name+"  "+line[10:]
                elif res_n < 100:
                    line = "   "+str(res_n)+res_name+"  "+line[10:]
                elif res_n < 1000:
                    line = "  "+str(res_n)+res_name+"  "+line[10:]
                elif res_n < 10000:
                    line = " "+str(res_n)+res_name+"  "+line[10:]
                else:
                    line = str(res_n)+res_name+"  "+line[10:]
                out.write(line)
                n += 1
                if n % n_atoms == 0:
                    n = 0
                    res_n += 1
            out.write(lines[-1])
    
        out.close()
    file.close()
        