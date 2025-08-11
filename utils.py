import torch
from Bio.PDB import PDBParser

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(pseudo_beta, torch.Tensor):
        pseudo_beta = torch.tensor(pseudo_beta, dtype=torch.float32)
    if not isinstance(pseudo_beta_mask, torch.Tensor):
        pseudo_beta_mask = torch.tensor(pseudo_beta_mask, dtype=torch.float32)

    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        # device=logits.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def pdb_to_tensor(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    resname = residue.get_resname()
                    if resname == "GLY":
                        atom_name = "CA"
                    else:
                        atom_name = "CB"

                    if atom_name in residue:
                        atom = residue[atom_name]
                        coords.append(atom.coord)
                    else:
                        continue
                else:
                    for atom in residue:
                        if atom.element != "H":
                            coords.append(atom.coord)

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    return coords_tensor

def af3_json(pdb_sequence=None, chain_id=[], name=None, seed=None, single=True, ligandccd=None, ligand_id=None, modify=None):
    '''
    This script only protein & ligand, and no gly and modify.
    seed is a list
    single is a bool whether run data pipeline
    proteinchain: eg.["A", "B"]
    ligandchain: eg.["C", "D"]
    ligandccd: eg. ["RET"]
    LYR is a number means pdb file have LYR, LYR is the number of LYR
    modify  [{"ptmType": "LYR","ptmPosition": 263}]
    '''
    af3_dic = {}
    af3_dic['dialect'] = "alphafold3"
    af3_dic["version"] = 1
    af3_dic['name'] = name
    af3_dic['sequences'] = []
    af3_dic['modelSeeds'] = list(seed)
    af3_dic["bondedAtomPairs"] = None #[[["A",143,"NZ"],["B",1,"C15"]]]
    af3_dic["userCCD"]= None

    chains = str(pdb_sequence).split(',')
    for _, i in enumerate(chain_id):
        protein = {}
        protein['id'] = list(i)
        protein['sequence'] = str(chains[_])
        if modify != None:
            protein['modifications'] = modify
        if single:
            protein["unpairedMsa"]=""
            protein["pairedMsa"]=""
            protein["templates"]=[]
        af3_dic['sequences'].append({"protein":protein})

    if ligandccd != None:
        ligand ={}
        ligand["ccdCodes"] = list(ligandccd)
        ligand["id"] = list(ligand_id)
        af3_dic['sequences'].append({"ligand":ligand})

    return af3_dic