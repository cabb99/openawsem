import os
import subprocess

folder_location = '/home/fc36/openawsem/tests/data/fraglib-4tlt-fragment_memory_term'
coords_folder_location = f'{"/".join(folder_location.split("/")[:-1])}/coords_{folder_location.split("/")[-1]}'

for filename in os.listdir(folder_location):
    bad_flag = False
    if filename[-4:] == '.pdb':
        base = filename[:-4]
        coords_filename = f'{base}.gro.coords'
    else: # gro file
        continue
    coords = []
    names = []
    with open(f'{coords_folder_location}/{coords_filename}','r') as f:
        for line in f:
            string_coords = [element for element in line.split(' ') if element not in ['','\n']]
            float_coords_in_angstrom = [round(float(coord)*10,3) for coord in string_coords]
            num_digits_after_decimal = [len(str(coord).split('.')[1]) for coord in float_coords_in_angstrom]
            # make sure we have 3 decimal digits and line up our number with the correct column
            string_coords_in_angstrom = []
            for counter in range(3):
                to_append = str(float_coords_in_angstrom[counter])
                for _ in range(3-num_digits_after_decimal[counter]):
                    to_append += "0"
                string_coords_in_angstrom.append(to_append.rjust(8))
            #string_coords_in_angstrom = [str(coord).rjust(8) for coord in float_coords_in_angstrom]
            coords.append(string_coords_in_angstrom)
    lines = []
    counter = 0
    with open(f'{folder_location}/{filename}','r') as f:
        for line in f:
            if line[:4] == "ATOM" or line[:6] == "HETATM":
                try:
                    lines.append(f'{line[:30]}{coords[counter][0]}{coords[counter][1]}{coords[counter][2]}{line[54:]}')
                    #print(filename)
                    #print(f'{line[:30]}{coords[counter][0]}{coords[counter][1]}{coords[counter][2]}{line[54:]}')
                    #exit()
                except IndexError:
                    print(filename)
                    print(line)
                    raise
                    #bad_flag = True
                    #break
                counter += 1
            else:
                lines.append(line)
    #if bad_flag:
    #    continue
    #else:
    #    with open(f'tests/data/fraglib-1r69-fragment_memory_term/foo_{filename}','w') as f:
    #        for line in lines:
    #            f.write(line)


    #if filename == "4m83A.pdb":
    #    with open('foo.txt','w') as f:
    #        for line in lines:
    #            f.write(line)
    with open(f'{folder_location}/{filename}','w') as f:
        for line in lines:
            f.write(line)
