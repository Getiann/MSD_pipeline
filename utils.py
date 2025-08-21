import torch
from Bio.PDB import PDBParser
from Bio.PDB import is_aa
from Bio.SeqUtils import seq1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.log(logits),
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

    errors = cross_entropy(
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

def distogram_prob_loss(
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
    true_bins = true_bins.unsqueeze(2) # L,L,1
    select_prob = logits.gather(2, true_bins) # L,L,1
    select_prob = select_prob.squeeze(2)

    return select_prob.mean()

def pdb_to_tensor(pdb_file, include_hetero=True):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    coords = []
    res_id = []
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
                        # if atom.get_altloc() in ("A", " "):

                        res_id.append(residue.id[1])
                        coords.append(atom.coord)
                    else:
                        continue
                elif residue.id[0] == "H" and include_hetero:
                    for atom in residue:
                        if atom.element != "H":
                            coords.append(atom.coord)

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    return coords_tensor

def extract_sequences(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("PDB", pdb_file)

    sequences = {}
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == ' ']
            sequence = "".join([seq1(res.resname) for res in residues])
            sequences[chain.id] = sequence
    return sequences

def af3_json(pdb_sequence=None, chain_id=[], version=1,
             name=None, seed=None, single=True, ligandccd=None, ligand_id=None, modify=None,
             userCCD_path=None, bonds=[]):
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
    af3_dic["version"] = version
    af3_dic['name'] = name
    af3_dic['sequences'] = []
    af3_dic['modelSeeds'] = list(seed)
    af3_dic["bondedAtomPairs"] = bonds #[[["A",143,"NZ"],["B",1,"C15"]]]
    af3_dic["userCCDPath"]= userCCD_path

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



def mean_max_prob(arr: np.ndarray) -> float:

    if arr.ndim != 3 or arr.shape[0] != arr.shape[1] :
        raise ValueError("输入必须是 (L, L, C) 维度的 numpy array")

    max_probs = arr.max(axis=-1)   # 在最后一维取最大值 => (L, L)
    mean_value = max_probs.mean()  # 对整个 (L, L) 求平均

    return mean_value

def plot_contact_3d(P, name: str,pseudo_beta=None ,out_dir: str = "./"):
    """
    P: (L, L, 64) contact map
    index_map: (L, L, 1) 指定每个位置取哪一维,如果没有指定，则每个维度取最大概率
    """
    L = P.shape[0]
    assert P.shape[-1] == 64, "最后一维必须是64"
    min_bin=2.3125
    max_bin=21.6875
    no_bins=64

    bin_centers = torch.linspace(
    min_bin,
    max_bin,
    no_bins - 1,)
    bin_centers = torch.tensor(bin_centers, dtype=torch.float32)
    if pseudo_beta is not None:
        if not isinstance(pseudo_beta, torch.Tensor):
            pseudo_beta = torch.tensor(pseudo_beta, dtype=torch.float32)
        dists = torch.sum(
            (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]),
            dim=-1,
            keepdims=True,
        )
        index_map = torch.sum(dists > bin_centers, dim=-1)
        print(index_map)
        index_map = index_map.unsqueeze(2)
    else:
        index_map = np.argmax(P, axis=-1)[..., None] # 取最大概率对应的索引


    Z = np.zeros((L, L))
    C = np.zeros((L, L))
    bin_centers = np.append(bin_centers, bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]))
    for i in range(L):
        for j in range(L):
            bin_idx = int(index_map[i, j, 0])
            Z[i, j] = bin_centers[bin_idx]  # 高度 = bin 中心距离
            C[i, j] = P[i, j, bin_idx]      # 颜色 = 对应概率值


    X, Y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(C),
                        rstride=1, cstride=1, linewidth=0, antialiased=False)

    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(C)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Probability')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Distance")
    ax.set_title(f"{name} (Height = Distance, Color = Probability)")
    out_file = f"{out_dir}/{name}_contactmap.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
