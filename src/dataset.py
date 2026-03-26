import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def load_data(path="data/dilirank.csv"):
    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue

    raise ValueError("파일 인코딩을 읽을 수 없음")


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)
