from rdkit import Chem
from rdkit.Chem import AllChem
import re
import os
from subprocess import Popen
import matplotlib.pyplot as plt


# path_to_orca = "/usr/local/orca_4_1_1_linux_x86-64/orca"
path_to_orca = "/usr/local/orca_4_1_1/orca"

level = "Cheap"

maxcore = 4000

conversion_factor = 627.5  # (kcal/mol)/Ha^-1

method_dict = {"Default": ("! PBE0 def2-SVP RIJCOSX def2/J PAL4 TIGHTSCF TightOpt Freq D3BJ",
                           "! wB97X-D3 def2-TZVPP RIJCOSX def2/J PAL4 TIGHTSCF GridX6"),
               "High-level": ("! RI-MP2 def2-TZVP RIJCOSX def2/J def2-TZVP/C PAL8 TIGHTSCF TightOpt NumFreq",
                              "! DLPNO-CCSD(T) def2-QZVPP RIJCOSX def2/J PAL8 TIGHTSCF GridX6"),
               "Cheap": ("! PBE def2-SVP RIJCOSX def2/J PAL4 Opt Freq D3BJ",
                         None)}

probe_dict = {"SeH": ("[Se][H]", "[H][Se][H]", ".[Se]%99[H]"),
              "Br": ("[Br]", "Br", ".[Br]%99"),
              "Cl": ("[Cl]", "Cl", ".[Cl]%99"),
              "F": ("[F]", "F", ".[F]%99"),
              "I": ("[I]", "I", ".[I]%99"),
              "OH": ("[O][H]", "[H]O[H]", ".O%99[H]"),
              "SH": ("[S][H]", "[H]S[H]", ".S%99[H]"),
              "SeH": ("[Se][H]", "[H][Se][H]", ".Se%99[H]"),
              "TeH": ("[Te][H]", "[H][Te][H]", ".Te%99[H]"),
              "NH2": ("[N]([H])[H]", "[H]N([H])[H]", ".[N]%99([H]])[H]"),
              "PH2": ("[P]([H])[H]", "[H]P([H])[H]", ".[P]%99([H])[H]"),
              "AsH2": ("[As]([H])[H]", "[H][As]([H])[H]", ".[As]%99([H])[H]"),
              "CH3": ("[H][C]([H])[H]", "C", ".C%99"),
              "SiH3": ("[H][Si]([H])[H]", "[H][Si]([H])([H])[H]", ".[H][Si]%99([H])[H]"),
              "GaH3": ("[H][Ga]([H])[H]", "[H][Ga]([H])([H])[H]", ".[H][Ga]%99([H])[H]"),
              "SnH3": ("[H][Sn]([H])[H]", "[H][Sn]([H])([H])[H]", ".[H][Sn]%99][H])[H]")}

light_atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl']
heavy_atoms_and_analogues = {'Te': 'S', 'As': 'P', 'Ga': 'Al', 'Sn': 'Si', 'Se': 'S'}


def make_mol_obj(smiles_string):
    obj = Chem.MolFromSmiles(smiles_string)
    obj = Chem.AddHs(obj)
    AllChem.EmbedMultipleConfs(obj, numConfs=1, params=AllChem.ETKDG())
    return obj


def gen_conformer_xyzs(mol_obj, conf_ids, alt_atom_and_new):
    """
    Generate xyz lists for all the conformers in mol.conf_ids
    :param mol_obj: rdkit object
    :param conf_ids: (list) list of conformer ids to convert to xyz
    :param alt_atom_and_new: (tuple)
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

                if alt_atom_and_new is not None:
                    print('here')
                    if atom_label == alt_atom_and_new[1]:
                        atom_label = alt_atom_and_new[0]

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
            keyword_line = method_dict[level][0]
            if len(mol.xyzs) == 1:
                if "TightOpt" in keyword_line:
                    keyword_line = keyword_line.replace("TightOpt", "")
                if "Opt" in keyword_line:
                    keyword_line = keyword_line.replace("Opt", "")
            print(keyword_line, file=inp_file)
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
    opt_xyzs, energy, gibbs_corr = [], 0.0, 0.0

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

        if 'G-E(el)' in line:
            gibbs_corr = float(line.split()[2])

    return opt_xyzs, energy, gibbs_corr


def get_orca_gibbs_corr_energy_single_atom(out_lines):

    S_trans, H_total, energy = 0.0, 0.0, 0.0

    for line in out_lines:

        if 'Translational entropy' in line:
            S_trans = float(line.split()[3])

        if 'Total enthalpy' in line:
            H_total = float(line.split()[3])

        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])

    gibbs_corr = (S_trans + H_total) - energy

    return energy, gibbs_corr


def get_orca_sp_energy(out_lines):

    for line in out_lines[::-1]:

        if 'FINAL SINGLE POINT ENERGY' in line:
            return float(line.split()[4])             # e.g. line = 'FINAL SINGLE POINT ENERGY     -4143.815610365798'


def change_heavy_atom_smiles(smiles):

    changed_atom_and_new = None
    for heavy_atom in heavy_atoms_and_analogues.keys():
        if heavy_atom in smiles:
            analogue = heavy_atoms_and_analogues[heavy_atom]
            smiles = smiles.replace(heavy_atom, analogue)
            changed_atom_and_new = (heavy_atom, analogue)

    return smiles, changed_atom_and_new


def get_atoms_in_smiles_string(smiles):

    atoms = []

    smiles_str_list = list(smiles)
    for i, char in enumerate(smiles_str_list):
        if i < len(smiles_str_list) - 1:
            if char.isupper() and smiles_str_list[i+1].islower():
                atoms.append(''.join(smiles_str_list[i:i+2]))
            if char.isupper() and not smiles_str_list[i+1].islower():
                atoms.append(char)
        else:
            if char.isupper():
                atoms.append(char)

    return atoms


class Molecule(object):

    def calc_gibbs(self):
        self.optimise()
        self.single_point()
        self.set_gibbs()

    def optimise(self):
        inp_filename = gen_orca_inp(mol=self, name=self.name + "_opt", opt=True)
        orca_output_lines = run_orca(inp_filename, out_filename=inp_filename.replace(".inp", ".out"))
        if len(self.xyzs) == 1:
            self.energy, self.gibbs_corr = get_orca_gibbs_corr_energy_single_atom(out_lines=orca_output_lines)
        else:
            self.xyzs, self.energy, self.gibbs_corr = get_orca_opt_xyzs_energy(out_lines=orca_output_lines)

    def single_point(self):
        if method_dict[level][1] is not None:
            inp_filename = gen_orca_inp(mol=self, name=self.name + "_sp", sp=True)
            orca_output_lines = run_orca(inp_filename, out_filename=inp_filename.replace(".inp", ".out"))
            self.energy = get_orca_sp_energy(out_lines=orca_output_lines)

    def set_gibbs(self):
        self.gibbs = self.energy + self.gibbs_corr

    def __init__(self, smiles, charge=0, mult=1, name="strained"):

        self.smiles = smiles
        self.charge = charge
        self.mult = mult
        self.name = name
        self.energy = None
        self.gibbs_corr = None
        self.gibbs = None
        self.switched_atom_and_new = None

        if any([heavy_atom in smiles for heavy_atom in heavy_atoms_and_analogues.keys()]):
            atoms_in_smiles = get_atoms_in_smiles_string(smiles=smiles)
            if any([sec_row_atom in atoms_in_smiles for sec_row_atom in heavy_atoms_and_analogues.values()]):
                exit('RDKit cannot handle heavy atoms, so changing them to second row will fail when the'
                     'xyzs are generated and we need to convert them back.')        # TODO fix RDKit

            self.smiles, self.switched_atom_and_new = change_heavy_atom_smiles(smiles=smiles)

        self.obj = make_mol_obj(smiles)
        self.xyzs = gen_conformer_xyzs(mol_obj=self.obj, conf_ids=[0], alt_atom_and_new=self.switched_atom_and_new)[0]

        self.calc_gibbs()


def calc_dG_addition(strained, probe, adduct):
    return (adduct.gibbs - (strained.gibbs + probe.gibbs))*conversion_factor


def calc_dG_isodesmic(probeH, adduct, probe, adductH):
    return ((probe.gibbs + adductH.gibbs) - (probeH.gibbs + adduct.gibbs))*conversion_factor


def plot_strain_graph(strained_smiles, general_adduct_smiles, charge_on_probe):
    mult = 1
    if charge_on_probe == 0:
        mult = 2
    strained = Molecule(smiles=strained_smiles)
    general_adduct_smiles = modify_adduct_smiles(smiles_string=general_adduct_smiles)

    xs, ys = [], []

    for probe_name in probe_dict.keys():
        probe = Molecule(smiles=probe_dict[probe_name][0], name=probe_name + str(charge_on_probe),
                         charge=charge_on_probe, mult=mult)
        probeH = Molecule(smiles=probe_dict[probe_name][1], name=probe_name + "H")
        adduct = Molecule(smiles=general_adduct_smiles + probe_dict[probe_name][2],
                          name=strained.name + "_" + probe.name, charge=charge_on_probe, mult=mult)
        adductH = Molecule(smiles=add_H_to_adduct(adduct_smiles=adduct.smiles),
                           name=strained.name + "_" + probe.name + "H")
        dG_addition = calc_dG_addition(strained, probe, adduct)
        dG_isodesmic = calc_dG_isodesmic(probeH, adduct, probe, adductH)
        xs.append(dG_addition)
        ys.append(dG_isodesmic)

    plt.scatter(xs, ys)
    plt.xlabel("dG_addition")
    plt.ylabel("dG_isodesmic")
    return plt.savefig("strain_graph.png")


if __name__ == "__main__":
    ethene_smiles = "C=C"
    test_adduct_smiles = "[H][C]([H])C[*]"
    charge_on_probe = 1

    plot_strain_graph(strained_smiles=ethene_smiles, general_adduct_smiles=test_adduct_smiles,
                      charge_on_probe=charge_on_probe)
