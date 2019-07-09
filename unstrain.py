from rdkit import Chem
from rdkit.Chem import AllChem
import re
import os
from subprocess import Popen


path_to_orca = "/usr/local/orca_4_1_1_linux_x86-64/orca"

level = "Default"

maxcore = 4000

method_dict = {"Default": ("! PBE0 def2-SVP RIJCOSX def2/J PAL4 TIGHTSCF TightOpt Freq D3BJ",
                           "! wB97X-D3 def2-TZVPP RIJCOSX def2/J PAL4 TIGHTSCF D3BJ GridX6"),
               "High-level": ("! RI-MP2 def2-TZVP RIJCOSX def2/J def2-TZVP/C PAL8 TIGHTSCF TightOpt NumFreq",
                              "! DLPNO-CCSD(T) def2-QZVPP RIJCOSX def2/J PAL8 TIGHTSCF GridX6"),
               "Cheap": ("! PBE def2-SVP RIJCOSX def2/J PAL4 Opt Freq D3BJ",
                         "None")}


def make_mol_obj(smiles_string):
    obj = Chem.MolFromSmiles(smiles_string)
    obj = Chem.AddHs(obj)
    AllChem.EmbedMultipleConfs(obj, numConfs=1, params=AllChem.ETKDG())
    return obj


def gen_conformer_xyzs(mol_obj, conf_ids):
    """
    Generate xyz lists for all the conformers in mol.conf_ids
    :param mol_obj: rdkit object
    :param conf_ids: (list) list of conformer ids to convert to xyz
    :return: (list) of xyz lists
    """
    xyzs = []

    for i in range(len(conf_ids)):
        mol_block_lines = Chem.MolToMolBlock(mol_obj, confId=conf_ids[i]).split('\n')
        mol_file_xyzs = []

        for line in mol_block_lines:
            split_line = line.split()
            if len(split_line) == 16:
                atom_label, x, y, z = split_line[3], split_line[0], split_line[1], split_line[2]
                mol_file_xyzs.append([atom_label, float(x), float(y), float(z)])

        xyzs.append(mol_file_xyzs)

    if len(xyzs) == 0:
        exit()

    return xyzs


def modify_adduct_smiles(smiles_string):
    return smiles_string.replace("[*]","%99")


def xyzs2xyzfile(xyzs, filename=None, basename=None, title_line=''):
    """
    For a list of xyzs in the form e.g [[C, 0.0, 0.0, 0.0], ...] convert create a standard .xyz file

    :param xyzs: List of xyzs
    :param filename: Name of the generated xyz file
    :param basename: Name of the generated xyz file without the file extension
    :param title_line: String to print on the title line of an xyz file
    :return: The filename
    """

    if basename:
        filename = basename + '.xyz'

    if filename is None:
        return 1

    if filename.endswith('.xyz'):
        with open(filename, 'w') as xyz_file:
            if xyzs:
                print(len(xyzs), '\n', title_line, sep='', file=xyz_file)
            else:
                return 1
            [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(*line), file=xyz_file) for line in xyzs]

    return filename


def add_H_to_adduct(adduct_smiles):

    pattern1 = "\[.\]"
    pattern2 = "\[..\]"
    atom_labels_sq_brackets = re.findall(pattern1, adduct_smiles)
    atom_labels_sq_brackets += re.findall(pattern2, adduct_smiles)

    for atom_label_sq_brackets in atom_labels_sq_brackets:
        if atom_label_sq_brackets != "[H]":
            adduct_smiles = adduct_smiles.replace(atom_label_sq_brackets, atom_label_sq_brackets[1:-1])
    return adduct_smiles


def did_orca_calculation_terminate_normally(out_filename):

    out_lines = [line for line in open(out_filename, 'r', encoding="utf-8")]
    for n_line, line in enumerate(reversed(out_lines)):
        if 'ORCA TERMINATED NORMALLY' or 'The optimization did not converge' in line:
            return True
        if n_line > 50:
            # The above lines are pretty close to the end of the file – there's no point parsing it all
            break

    return False


def gen_orca_inp(mol, name, opt=False, sp=False):
    inp_filename = name + ".inp"
    with open(inp_filename, "w") as inp_file:
        if opt:
            print(method_dict[level][0], file=inp_file)
        if sp:
            print(method_dict[level][1], file=inp_file)
        print("%maxcore", maxcore, file=inp_file)
        print("*xyz", mol.charge, mol.mult, file=inp_file)
        [print('{:<3}{:^12.8f}{:^12.8f}{:^12.8f}'.format(*line), file=inp_file) for line in mol.xyzs]
        print('*', file=inp_file)
    return inp_filename


def run_orca(inp_filename, out_filename):
    """
    Run the ORCA calculation given the .inp file as a subprocess
    :param inp_filename:
    :param out_filename:
    :return:
    """

    if os.path.exists(os.path.join("Library", level)):
        if os.path.exists(os.path.join("Library", os.path.join(level, out_filename))):
            return [line for line in open(out_filename, 'r', encoding="utf-8")]

    orca_terminated_normally = False

    if os.path.exists(out_filename):
        orca_terminated_normally = did_orca_calculation_terminate_normally(out_filename)

    if not orca_terminated_normally:
        with open(out_filename, 'w') as orca_out:
            orca_run = Popen([path_to_orca, inp_filename], stdout=orca_out)
        orca_run.wait()

    return [line for line in open(out_filename, 'r', encoding="utf-8")]


def get_orca_opt_xyzs_energy(out_lines):
    """
    For a lost of ORCA output file lines find the optimised xyzs and energy
    :param out_lines:
    :return:
    """

    opt_converged, geom_section = False, False
    opt_xyzs, energy = [], 0.0

    for line in out_lines:

        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            opt_converged = True
        if 'CARTESIAN COORDINATES' in line and opt_converged:
            geom_section = True

        if geom_section and len(line.split()) == 0:
            geom_section = False

        if geom_section and len(line.split()) == 4:
            atom_label, x, y, z = line.split()
            opt_xyzs.append([atom_label, float(x), float(y), float(z)])

        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])             # e.g. line = 'FINAL SINGLE POINT ENERGY     -4143.815610365798'

    return opt_xyzs, energy


class Molecule(object):

    def optimise(self):
        inp_filename = gen_orca_inp(mol=self, name=self.name + "_opt", opt=True)
        orca_output_lines = run_orca(inp_filename, out_filename=inp_filename.replace(".inp", ".out"))
        self.xyzs, self.energy = get_orca_opt_xyzs_energy(out_lines=orca_output_lines)

    def __init__(self, smiles, charge=0, mult=1, name="strained"):

        self.smiles = smiles
        self.charge = charge
        self.mult = mult
        self.name = name
        self.energy = None
        self.obj = make_mol_obj(smiles)
        self.xyzs = gen_conformer_xyzs(mol_obj=self.obj, conf_ids=[0])[0]


if __name__ == "__main__":
    ethene_smiles = "C=C"
    methyl_smiles = "[H][C]([H])[H]"
    adduct_smiles = "[H][C]([H])C[*]"
    adduct_smiles = modify_adduct_smiles(smiles_string=adduct_smiles)
    methyl_addition_smiles = ".C%99"

    ethene = Molecule(smiles=ethene_smiles)
    methyl = Molecule(smiles=methyl_smiles)
    adduct = Molecule(smiles=adduct_smiles + methyl_addition_smiles)
    product = Molecule(smiles=add_H_to_adduct(adduct_smiles=adduct.smiles))

    ethene.optimise()








