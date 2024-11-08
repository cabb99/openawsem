import argparse
from pathlib import Path

def process_trajectory(movieFile, fastaFile, outputFile, startFrame=0, endFrame=None, stride=1):
    seq_dic = get_seq_dic(fasta=fastaFile)
    convert_openMM_to_standard_pdb(fileName=movieFile, seq_dic=seq_dic, back=True)

    with open(movieFile, "r") as f:
        a = f.readlines()
    n = len(a)
    model_indices = [i for i, line in enumerate(a) if len(line) >= 5 and line[:5] == "MODEL"]

    if endFrame is None:
        endFrame = len(model_indices)

    selected_frames = []
    for i in range(startFrame, endFrame, stride):
        if i < len(model_indices):
            start_idx = model_indices[i]
            end_idx = model_indices[i + 1] if i + 1 < len(model_indices) else n
            selected_frames.extend(a[start_idx:end_idx])

    with open(outputFile, "w") as out:
        out.write("".join(selected_frames))

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Convert openMM output to standard Pdbs."
    )
    parser.add_argument("PDBFile", help="The name of the PDB file, such as native.pdb or movie.pdb")
    parser.add_argument("-f", "--fasta", default="./crystal_structure.fasta", help="Default is ./crystal_structure.fasta")
    parser.add_argument("-o", "--output", default="lastFrame.pdb", help="Output file name. Default is lastFrame.pdb")
    parser.add_argument("--start", type=int, default=0, help="Start frame. Default is 0")
    parser.add_argument("--end", type=int, help="End frame. Default is the last frame")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frames. Default is 1")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    process_trajectory(args.PDBFile, args.fasta, args.output, args.start, args.end, args.stride)

def get_seq_dic(fasta="../crystal_structure.fasta"):
    seq_dic = {}
    chain = None
    with open(fasta) as f:
        for line in f:
            if line[0] == ">":
                assert line[:19] == ">CRYSTAL_STRUCTURE:"
                if chain is not None:
                    seq_dic[chain] = seq
                chain = line[19]
                seq = ""
            else:
                seq += line.replace("\n", "")
        seq_dic[chain] = seq
    return seq_dic


def convert_openMM_to_standard_pdb(fileName="last_frame.pdb", seq_dic=None, back=True):
    code = {"GLY": "G", "ALA": "A", "LEU": "L", "ILE": "I",
            "ARG": "R", "LYS": "K", "MET": "M", "CYS": "C",
            "TYR": "Y", "THR": "T", "PRO": "P", "SER": "S",
            "TRP": "W", "ASP": "D", "GLU": "E", "ASN": "N",
            "GLN": "Q", "PHE": "F", "HIS": "H", "VAL": "V"}
    inv_code_map = {v: k for k, v in code.items()}
    if seq_dic is None:
        seq_dic = get_seq_dic()
    
    file_path = Path(fileName)
    backup_path = file_path.with_suffix(file_path.suffix + '.bak') if back and file_path.with_suffix('.bak').exists() else None

    if backup_path:
        file_path.rename(backup_path)
        file_path = backup_path

    lines = file_path.read_text().splitlines()
    new_lines = []

    for line in lines:
        if len(line) >= 4 and line[:4] == "END":
            continue
        if len(line) > 25:
            if line[:6] == "REMARK":
                continue
            i = int(line[22:26])
            chain = line[21]

            tmp = list(line)
            if "".join(tmp[17:20]) in ["IGL", "NGP", "IPR"]:
                res = seq_dic[chain][i - 1]
                tmp[17:20] = inv_code_map[res]
                if line[:6] == "HETATM":
                    tmp[:6] = "ATOM  "
            new_lines.append("".join(tmp))
        else:
            new_lines.append(line)

    file_path.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    main()
