from rdkit import Chem
from rdkit.Chem import AllChem
import re
import os
from subprocess import Popen
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import argparse
from multiprocessing import Pool


path_to_orca = "/usr/local/orca_4_1_1_linux_x86-64/orca"
# path_to_orca = "/usr/local/orca_4_1_1/orca"

conversion_factor = 627.5  # (kcal/mol)/Ha^-1

method_dict = {"Default": ("! PBE0 def2-SVP RIJCOSX def2/J TIGHTSCF TightOpt Freq D3BJ",
                           "! wB97X-D3 def2-TZVPP RIJCOSX def2/J TIGHTSCF Grid6 GridX6"),
               "High-level": ("! RI-MP2 def2-TZVP RIJCOSX def2/J def2-TZVP/C TIGHTSCF TightOpt NumFreq",
                              "! DLPNO-CCSD(T) def2-QZVPP RIJCOSX def2/J TIGHTSCF GridX6"),
               "Cheap": ("! PBE def2-SVP RIJCOSX def2/J Opt Freq D3BJ",
                         None)}

probe_dict = {"SiH3": ("[H][Si]([H])[H]", "[H][Si]([H])([H])[H]", ".[Si]%99([H])([H])[H]"),
              "SeH": ("[Se][H]", "[H][Se][H]", ".[Se]%99[H]"),
              "Br": ("[Br]", "Br", ".[Br]%99"),
              "Cl": ("[Cl]", "Cl", ".[Cl]%99"),
              "F": ("[F]", "F", ".[F]%99"),
              "I": ("[I]", "I", ".[I]%99"),
              "OH": ("[O][H]", "[H]O[H]", ".[O]%99[H]"),
              "SH": ("[S][H]", "[H]S[H]", ".[S]%99[H]"),
              "TeH": ("[Te][H]", "[H][Te][H]", ".[Te]%99[H]"),
              "NH2": ("[N]([H])[H]", "[H]N([H])[H]", ".[N]%99([H])[H]"),
              "PH2": ("[P]([H])[H]", "[H]P([H])[H]", ".[P]%99([H])[H]"),
              "AsH2": ("[As]([H])[H]", "[H][As]([H])[H]", ".[As]%99([H])[H]"),
              "CH3": ("[H][C]([H])[H]", "C", ".C%99"),
              "GeH3": ("[H][Ge]([H])[H]", "[H][Ge]([H])([H])[H]", ".[Ge]%99([H])([H])[H]"),
              "SnH3": ("[H][Sn]([H])[H]", "[H][Sn]([H])([H])[H]", ".[Sn]%99([H])([H])[H]")
              }


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("strained", action='store', type=str, help='SMILES string of strained molecule')
    parser.add_argument("adduct", action='store', type=str, help='SMILES string of adduct')
    parser.add_argument("-np", '--number_processors', action='store', type=int, default=1, help='Number of processors')
    parser.add_argument("-l", '--level', action='store', type=str, default="Default",
                        help='Level of theory for calculations', choices=['Default', 'High-level', 'Cheap'])
    parser.add_argument("-chg", '--charge_on_probe', action='store', type=int, default=0,
                        help='Charge on probe for addition to strained molecule')
    parser.add_argument("-m", '--max_core', action='store', type=float, default=4000, help='Maximum memory per core')

    return parser.parse_args()


def make_mol_obj(smiles_string):
    """
    Make an RDKit molecule object from a SMILES string (e.g. generated from ChemDraw)
    :param smiles_string: (str) SMILES of a molecule
    :return: (object) RDKit Mol object
    """
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
        exit('Could not generate xyzs from RDKit object')

    return xyzs


def modify_adduct_smiles(smiles_string):

    # e.g, [*]C([H])([H])[C]([H])[H]
    if smiles_string.startswith('[*]'):

        # If the 3rd and 4th characters are letters..
        if smiles_string[3].isalpha() and smiles_string[4].isalpha():
            smiles_string = smiles_string[3:5] + '[*]' + smiles_string[5:]

        # If the 3rd character is a letter..
        elif smiles_string[3].isalpha():
            smiles_string = smiles_string[3:4] + '[*]' + smiles_string[4:]

        else:
            exit('Failed to modify the adduct SMILES string')

    return smiles_string.replace("[*]", "%99")


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


def add_h_to_adduct(adduct_smiles):
    """
    For a SMILES string of and adduct

    i.e.

    C

    :param adduct_smiles:
    :return:
    """

    pattern1 = "\[.\]"
    pattern2 = "\[..\]"
    atom_labels_sq_brackets = re.findall(pattern1, adduct_smiles)
    atom_labels_sq_brackets += re.findall(pattern2, adduct_smiles)

    len_strained_smiles = len(adduct_smiles.split('.', 1)[0])

    for atom_label_sq_brackets in atom_labels_sq_brackets:
        if atom_label_sq_brackets != "[H]":
            if atom_label_sq_brackets in adduct_smiles[:len_strained_smiles]:
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


def gen_orca_inp(mol, name, opt=False, sp=False, pal=1):
    inp_filename = name + ".inp"
    with open(inp_filename, "w") as inp_file:
        if opt:
            keyword_line = method_dict[level][0]
            keyword_line += " PAL" + str(pal)
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
    opt_xyzs, energy, gibbs_corr = [], None, None

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

    s_trans, s_total, energy = None, None, None

    for line in out_lines:

        if 'Translational entropy' in line:
            s_trans = float(line.split()[3])

        if 'Total enthalpy' in line:
            s_total = float(line.split()[3])

        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])

    # If any of the energies are not found return None
    if any([e is None for e in [s_trans, s_total, energy]]):
        return None, None

    gibbs_corr = (s_trans + s_total) - energy

    return energy, gibbs_corr


def get_orca_sp_energy(out_lines):

    for line in out_lines[::-1]:
        if 'FINAL SINGLE POINT ENERGY' in line:
            return float(line.split()[4])             # e.g. line = 'FINAL SINGLE POINT ENERGY     -4143.815610365798'


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


def print_output(process, name, state):
    return print("{:<30s}{:<50s}{:>10s}".format(process, name, state))


class Molecule(object):

    def calc_gibbs(self):
        self.optimise()
        self.single_point()
        self.set_gibbs()

    def optimise(self):
        print_output('Optimisation of', self.name, 'Running')
        inp_filename = gen_orca_inp(mol=self, name=self.name + "_opt", opt=True)
        orca_output_lines = run_orca(inp_filename, out_filename=inp_filename.replace(".inp", ".out"))
        if len(self.xyzs) == 1:
            self.energy, self.gibbs_corr = get_orca_gibbs_corr_energy_single_atom(out_lines=orca_output_lines)
        else:
            self.xyzs, self.energy, self.gibbs_corr = get_orca_opt_xyzs_energy(out_lines=orca_output_lines)
        print_output('', '', 'Done')

    def single_point(self):
        print_output('Single point of of', self.name, 'Running')
        if method_dict[level][1] is not None:
            inp_filename = gen_orca_inp(mol=self, name=self.name + "_sp", sp=True)
            orca_output_lines = run_orca(inp_filename, out_filename=inp_filename.replace(".inp", ".out"))
            self.energy = get_orca_sp_energy(out_lines=orca_output_lines)
        print_output('', '', 'Done')

    def set_gibbs(self):
        if self.energy is None or self.gibbs_corr is None:
            self.gibbs = None
        else:
            self.gibbs = self.energy + self.gibbs_corr

    def __init__(self, smiles, charge=0, mult=1, name="strained"):
        print_output('Molecule object for', name, 'Generating')

        self.smiles = smiles
        self.charge = charge
        self.mult = mult
        self.name = name
        self.energy = None
        self.gibbs_corr = None
        self.gibbs = None
        self.obj = make_mol_obj(self.smiles)
        self.xyzs = gen_conformer_xyzs(mol_obj=self.obj, conf_ids=[0])[0]

        self.calc_gibbs()
        print_output('', '', '')


def calc_dG_addition(strained, probe, adduct):
    if any(gibbs is None for gibbs in [adduct.gibbs, strained.gibbs, probe.gibbs]):
        return None
    return (adduct.gibbs - (strained.gibbs + probe.gibbs))*conversion_factor


def calc_dG_isodesmic(probeH, adduct, probe, adductH):
    if any(gibbs is None for gibbs in [probeH.gibbs, adduct.gibbs, probe.gibbs, adductH.gibbs]):
        return None
    return ((probe.gibbs + adductH.gibbs) - (probeH.gibbs + adduct.gibbs))*conversion_factor


def get_xs_ys_not_none(xs, ys):
    xs_not_none, ys_not_none = [], []
    for i in range(len(xs)):
        if xs[i] is not None and ys[i] is not None:
            xs_not_none.append(xs[i])
            ys_not_none.append(ys[i])
    return xs_not_none, ys_not_none


def get_xs_to_zero(xs):

    if all([x < 0 for x in xs]):
        return list(sorted(xs)) + [0]
    else:
        return list(sorted(xs))


def calc_dGs(general_adduct_smiles, charge_on_probe, probe_name, mult, strained):
    probe = Molecule(smiles=probe_dict[probe_name][0], name=probe_name + str(charge_on_probe),
                     charge=charge_on_probe, mult=mult)
    probeH = Molecule(smiles=probe_dict[probe_name][1], name=probe_name + "H")
    adduct = Molecule(smiles=general_adduct_smiles + probe_dict[probe_name][2],
                      name=strained.name + "_" + probe.name, charge=charge_on_probe, mult=mult)
    adductH = Molecule(smiles=add_h_to_adduct(adduct_smiles=adduct.smiles),
                       name=strained.name + "_" + probe.name + "H")
    dG_addition = calc_dG_addition(strained, probe, adduct)
    dG_isodesmic = calc_dG_isodesmic(probeH, adduct, probe, adductH)

    return dG_addition, dG_isodesmic


def calc_strain_graph(strained_smiles, general_adduct_smiles, charge_on_probe):
    mult = 1
    if charge_on_probe == 0:
        mult = 2
    strained = Molecule(smiles=strained_smiles)
    general_adduct_smiles = modify_adduct_smiles(smiles_string=general_adduct_smiles)

    with Pool(processes=n_procs) as pool:
        results = [pool.apply_async(calc_dGs, (general_adduct_smiles, charge_on_probe, probe_name, mult, strained))
                   for probe_name in probe_dict.keys()]

        dGs = [res.get(timeout=None) for res in results]

    xs = [val[0] for val in dGs]
    ys = [val[1] for val in dGs]

    return xs, ys


def plot_strain_graph(strained_smiles, general_adduct_smiles, charge_on_probe):

    xs, ys = calc_strain_graph(strained_smiles, general_adduct_smiles, charge_on_probe)
    plt.scatter(xs, ys)

    xs_not_None, ys_not_None = get_xs_ys_not_none(xs, ys)
    m, c, r, p, err = linregress(xs_not_None, ys_not_None)
    plt.annotate("gradient = " + str(np.round(m,2)) + "\nstrain relief = " + str(np.round(c,1)) + "\n$r^2$ = "
                 + str(np.round(np.square(r),3)), (0.8*min(xs_not_None), 0.2*max(ys_not_None)), ha='center', va='center')

    xs_to_zero = get_xs_to_zero(xs=xs_not_None)
    plt.plot(xs_to_zero, np.array(xs_to_zero)*m + c, color = 'black', linestyle = 'dashed')
    plt.xlabel("$\Delta G_{addition}$ / kcal mol$^{-1}$")
    plt.ylabel("$\Delta G_{isodesmic}$ / kcal mol$^{-1}$")
    plt.axhline(y=0, color='k', linewidth = '0.5')
    plt.axvline(x=0, color='k', linewidth = '0.5')

    return plt.savefig("strain_graph.png")


if __name__ == "__main__":
    args = get_args()
    level = args.level
    maxcore = args.max_core
    n_procs = args.number_processors

    plot_strain_graph(strained_smiles=args.strained, general_adduct_smiles=args.adduct,
                      charge_on_probe=args.charge_on_probe)
