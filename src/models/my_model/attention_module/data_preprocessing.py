"""
data_preprocessing.py  (Step 1 — 데이터 전처리 모듈)
============================================================
DILIrank binary CSV → Global(MACCS+Morgan FP) + Local(BRICS Substructure) 벡터 추출

① SMILES 로드 및 유효성 검증
② Path A (Global) : MACCS Keys(167) + Morgan ECFP6(1024) → 1191-dim
③ Path B (Local)  : BRICS 분해 → Bag-of-Fragments OHE (vocab 기반)
④ Train/Test Split 후 vocab 고정 (데이터 누수 방지)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
import threading

from rdkit import Chem
from rdkit.Chem import (
    MACCSkeys,
    AllChem,
    BRICS,
    rdFingerprintGenerator,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import config

# Morgan Generator (Deprecation 경고 제거)
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.MORGAN_RADIUS,
    fpSize=config.MORGAN_NBITS
)

# BRICS timeout 설정 (초)
_BRICS_TIMEOUT = 3
# BRICS를 적용할 최대 원자 수 (초과 시 빠른 스킵, 모델의 복잡도 줄이기 위함. 이후 논문 참고하여 바꾸어도 됨)
_MAX_BRICS_ATOMS = 60


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def smiles_to_mol(smiles: str):
    """SMILES → RDKit Mol 변환. 실패 시 None 반환."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Path A : Global Fingerprint (MACCS + Morgan)
# ─────────────────────────────────────────────────────────────

def mol_to_global_fp(mol) -> np.ndarray:
    """
    분자 → Global Fingerprint 벡터 (1191-dim)
    = MACCS Keys (167) ‖ Morgan ECFP6 (1024)
    MorganGenerator 사용으로 Deprecation 경고 제거.
    """
    maccs  = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)   # (167,)
    morgan = np.array(_morgan_gen.GetFingerprint(mol), dtype=np.float32) # (1024,)
    return np.concatenate([maccs, morgan], axis=0)                       # (1191,)


# ─────────────────────────────────────────────────────────────
# Path B : Local Substructure (BRICS + Bemis-Murcko)
# ─────────────────────────────────────────────────────────────

def _brics_with_timeout(mol, result_holder: list):
    """별도 스레드에서 BRICS를 실행하여 결과를 result_holder[0]에 저장."""
    try:
        frags = list(BRICS.BRICSDecompose(mol))
        clean = []
        for f in frags:
            fm = Chem.MolFromSmiles(f)
            if fm is not None:
                clean.append(Chem.MolToSmiles(fm, canonical=True))
        result_holder.append(clean)
    except Exception:
        result_holder.append([])


def mol_to_fragments(mol) -> list[str]:
    """
    분자 → Substructure Fragment 목록
    ① BRICS 분해: 원자 수 _MAX_BRICS_ATOMS 초과 시 스킵,
                   _BRICS_TIMEOUT 초 내 미완료 시 강제 중단
    ② Bemis-Murcko 스캐폴드 (골격 구조 추가)
    """
    fragments = []

    # ── BRICS 분해 (timeout + 원자 수 제한) ──
    if mol.GetNumHeavyAtoms() <= _MAX_BRICS_ATOMS:
        result_holder = []
        t = threading.Thread(target=_brics_with_timeout, args=(mol, result_holder), daemon=True)
        t.start()
        t.join(timeout=_BRICS_TIMEOUT)
        if result_holder:
            fragments.extend(result_holder[0])
        # timeout 초과 시 result_holder가 비어있으므로 자동 스킵

    # ── Bemis-Murcko Scaffold ──
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None and scaffold.GetNumAtoms() > 0:
            fragments.append(Chem.MolToSmiles(scaffold, canonical=True))
    except Exception:
        pass

    return fragments


def build_fragment_vocab(fragment_lists: list[list[str]],
                         max_vocab: int = config.SUB_VOCAB_INIT_DIM) -> dict:
    """
    Train 데이터의 Fragment 빈도 기반 Vocabulary 생성.
    max_vocab 개수만큼만 허용하여 희소성(Sparsity) 제한.
    """
    counter = Counter()
    for frags in fragment_lists:
        counter.update(frags)
    # 빈도 상위 max_vocab 개 선택
    most_common = [frag for frag, _ in counter.most_common(max_vocab)]
    vocab = {frag: idx for idx, frag in enumerate(most_common)}
    return vocab


def fragments_to_vector(fragments: list[str], vocab: dict) -> np.ndarray:
    """
    Fragment 목록 → Bag-of-Fragments 빈도 벡터 (vocab_size,)
    vocab에 없는 Fragment는 무시 (UNK 처리)
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    for f in fragments:
        if f in vocab:
            vec[vocab[f]] += 1.0
    # L1 정규화(분자 크기 편향 제거)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


# ─────────────────────────────────────────────────────────────
# 메인 전처리 파이프라인
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(data_path: str = config.DATA_PATH):
    """
    CSV 파일 로드 → 전처리 → Train/Test 분리 →
    (fp_train, sub_train, y_train), (fp_test, sub_test, y_test), vocab 반환
    """
    print(f"[1/5] 데이터 로드: {data_path}")
    df = pd.read_csv(data_path)

    # 필수 컬럼 확인
    assert config.SMILES_COL in df.columns, f"'{config.SMILES_COL}' 컬럼이 없습니다."
    assert config.LABEL_COL  in df.columns, f"'{config.LABEL_COL}' 컬럼이 없습니다."

    # RDKit Mol 변환 & 유효성 필터
    print("[2/5] SMILES 유효성 검증 및 Mol 변환")
    df["mol"] = df[config.SMILES_COL].apply(smiles_to_mol)
    before = len(df)
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    print(f"      유효 분자: {len(df)} / {before} ({before - len(df)}개 제거)")

    labels  = df[config.LABEL_COL].values.astype(np.int64)
    smiles  = df[config.SMILES_COL].values

    # ── Train / Test Split (vocab 누수 방지를 위해 먼저 분리) ──
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=labels
    )
    mols_all = df["mol"].tolist()

    # ── Path A: Global Fingerprint ──
    print("[3/5] Global Fingerprint 추출 (MACCS + Morgan ECFP6)")
    fp_all = np.stack([mol_to_global_fp(m) for m in mols_all])  # (N, 1191)
    fp_train, fp_test = fp_all[tr_idx], fp_all[te_idx]

    # ── Path B: Local Substructure ──
    print("[4/5] Local Substructure 추출 (BRICS + Bemis-Murcko) — 분자당 최대 3초")
    frags_all = [
        mol_to_fragments(m)
        for m in tqdm(mols_all, desc="  Substructure", ncols=80, unit="mol")
    ]
    frags_train  = [frags_all[i] for i in tr_idx]

    # Train 기반 Vocab 구축 (Test에는 동일 vocab 적용)
    vocab = build_fragment_vocab(frags_train, max_vocab=config.SUB_VOCAB_INIT_DIM)
    print(f"      Vocab 크기: {len(vocab)}")

    sub_all   = np.stack([fragments_to_vector(f, vocab) for f in frags_all])   # (N, vocab)
    sub_train = sub_all[tr_idx]
    sub_test  = sub_all[te_idx]

    y_train, y_test = labels[tr_idx], labels[te_idx]

    print(f"[5/5] 완료 | Train: {len(tr_idx)}개, Test: {len(te_idx)}개")
    print(f"      Global FP dim : {fp_train.shape[1]}")
    print(f"      Local Sub dim : {sub_train.shape[1]}")
    print(f"      class 분포(train) — 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")

    return (fp_train, sub_train, y_train), (fp_test, sub_test, y_test), vocab


if __name__ == "__main__":
    (fp_tr, sub_tr, y_tr), (fp_te, sub_te, y_te), vocab = load_and_preprocess()
    print("fp_train shape :", fp_tr.shape)
    print("sub_train shape:", sub_tr.shape)
