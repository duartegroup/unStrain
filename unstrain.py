from rdkit import Chem
from rdkit.Chem import AllChem


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
    string_list = list(smiles_string)
    for i in range(len(string_list)-2):
        if string_list[i] == "[" and string_list[i+1] == "*" and string_list[i+2] == "]":
            string_list[i] = "%"
            string_list[i+1] = "9"
            string_list[i+2] = "9"
            break
    new_string = "".join(string_list)
    return new_string


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


class Molecule(object):

    def __init__(self, smiles):

        self.smiles = smiles
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

    temp_product = Chem.RemoveHs(adduct.obj)
    product = Chem.AddHs(temp_product)
    



