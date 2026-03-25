import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data


def load_data(path="data/dilirank.csv"):
    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue

    raise ValueError("파일 인코딩을 읽을 수 없음")


# -------------------------
# GNN용: SMILES → Graph
# -------------------------
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # node feature: atomic number
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetAtomicNum()])

    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# -------------------------
# RF/XGB용: fingerprint
# -------------------------
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)