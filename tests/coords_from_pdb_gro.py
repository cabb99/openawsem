import os
import numpy as np

pdb_info = {}
gro_info = {}

folder_location = '/home/fc36/openawsem/tests/data/fraglib-4tlt-fragment_memory_term'
coords_folder_location = f'{"/".join(folder_location.split("/")[:-1])}/coords_{folder_location.split("/")[-1]}'

for filename in os.listdir(folder_location):
    if filename[-4:] == '.pdb':
        pdb_info.update({filename[:-4]:[]})
        with open(f'{folder_location}/{filename}','r') as f:
            coord_lines = []
            for line in f:
                if line[:4] == "ATOM" or line[:6] == "HETATM":
                    x = str(round(float(line[30:38].strip())/10,3))
                    y = str(round(float(line[38:46].strip())/10,3))
                    z = str(round(float(line[46:54].strip())/10,3))
                    pdb_info[filename[:-4]] += [float(x),float(y),float(z)]
                    coord_lines.append(f'{x.strip().ljust(10)}{y.strip().ljust(10)}{z.strip().ljust(10)}\n')
        if not os.path.exists(coords_folder_location):
            os.mkdir(coords_folder_location)
        #with open(f'tests/data/fraglib-1r69-fragment_memory_term/{filename}.coords','w') as f:
        with open(f'{coords_folder_location}/{filename}.coords','w') as f:
            for line in coord_lines:
                f.write(line)
    elif filename[-4:] == '.gro':
        gro_info.update({filename[:-4]:[]})
        with open(f'{folder_location}/{filename}','r') as f:
            coord_lines = []
            for counter,line in enumerate(f):
                if counter >= 2:
                    try:
                        x, y, z, = [coord for coord in line[20:].split(' ') if coord!='']
                    except ValueError:
                        print(filename)
                        print(line)
                        raise
                    gro_info[filename[:-4]] += [float(x.strip()),float(y.strip()),float(z.strip())]
                    coord_lines.append(f'{x.strip().ljust(10)}{y.strip().ljust(10)}{z.strip().ljust(10)}\n')
        if not os.path.exists(coords_folder_location):
            os.mkdir(coords_folder_location)
        with open(f'{coords_folder_location}/{filename}.coords','w') as f:
        #with open(f'tests/data/fraglib-1r69-fragment_memory_term/{filename}.coords','w') as f:
            for line in coord_lines:
                f.write(line)

with open(f'{coords_folder_location}/coords_compare.txt','w') as f:
    combined_info = {}
    failed = []
    for pdbid in pdb_info.keys():
        try:
            combined_info[pdbid] = np.array(pdb_info[pdbid]) - np.array(gro_info[pdbid])
        except ValueError:
            failed.append(pdbid)
            continue
        max_num = np.max(np.abs(combined_info[pdbid]))
        if max_num > 0.0010000001:
            f.write(f'{pdbid}\n')
            f.write(f'max: {max_num}\n')
            f.write(f'first differing line in coords files: {np.where(np.abs(combined_info[pdbid])==max_num)[0][0]//3}')
            raise ValueError(f"coordinate files are significantly different! See {coords_folder_location}/coords_compare.txt")
    f.write('failed\n')
    f.write(str(failed))
    if failed:
        raise AssertionError(f"strange issue in coordinate comparison. see {coords_folder_location}/coords_compare.txt")
