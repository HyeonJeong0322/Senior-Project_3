"""
data_preprocessing.py  (Step 1 — 데이터 전처리 모듈)
============================================================
DILIrank binary CSV → Global(MACCS+Morgan FP) + Local(BRICS Substructure) 벡터 추출
[NEW] GCN을 위한 분자 그래프(원자 피처 + 인접 행렬) 추출 함수 추가

① SMILES 로드 및 유효성 검증
② Path A (Global) : MACCS Keys(167) + Morgan ECFP6(512) → 679-dim  ← RF 후 FP_SELECT_DIM 축소
③ Path B (Local/BoF): BRICS 분해 → Bag-of-Fragments OHE (vocab 기반, 기존 호환 유지)
④ Path C (Local/GCN): 분자 그래프 → 원자 피처 행렬 + 인접 행렬 (위상 보존)
⑤ Train/Test Split 후 vocab 고정 (데이터 누수 방지)
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
    분자 → Global Fingerprint 벡터 (679-dim)
    = MACCS Keys (167) ‖ Morgan ECFP6 (512)
    MorganGenerator 사용으로 Deprecation 경고 제거.
    """
    maccs  = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)   # (167,)
    morgan = np.array(_morgan_gen.GetFingerprint(mol), dtype=np.float32) # (512,)
    return np.concatenate([maccs, morgan], axis=0)                       # (679,)


# ─────────────────────────────────────────────────────────────
# Path B : Local Substructure (BRICS + Bemis-Murcko) [기존 유지]
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
# [NEW] Path C : GCN 그래프 데이터 추출
# ─────────────────────────────────────────────────────────────

def mol_to_graph(mol) -> tuple[np.ndarray, np.ndarray]:
    """
    RDKit Mol → (node_feats, adj_matrix) 반환.
    위상(Topology)을 보존하며 1-Layer GCN 입력 형식으로 변환.

    - node_feats : (N, GCN_NODE_FEAT_DIM) — 원자 피처 행렬 (N = 중원자 수)
    - adj_matrix : (N, N) — self-loop 포함 대칭 인접 행렬

    ※ attention_module.atom_features()와 반드시 동일한 피처 정의를 사용해야 함.
    """
    from .attention_module import atom_features  # 순환참조 방지를 위해 지역 import

    # 수소 추가 없이 중원자(Heavy Atom)만 대상
    n = mol.GetNumHeavyAtoms()

    # 원자 피처 행렬
    feat_list = [atom_features(atom) for atom in mol.GetAtoms()]
    node_feats = np.array(feat_list, dtype=np.float32)   # (N, 48)

    # 인접 행렬 (self-loop 포함)
    adj = np.eye(n, dtype=np.float32)   # self-loop
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0   # 무방향 그래프

    return node_feats, adj   # (N, feat_dim), (N, N)


def collate_graph_batch(graph_list: list[tuple]) -> tuple:
    """
    서로 다른 원자 수(N_i)를 가진 그래프 목록을
    최대 원자 수(max_N)로 패딩하여 단일 배치 텐서로 변환.

    Args:
        graph_list: list of (node_feats (N_i, F), adj (N_i, N_i)) tuples

    Returns:
        x    : (B, max_N, F)          — 패딩된 원자 피처 텐서
        adj  : (B, max_N, max_N)      — 패딩된 인접 행렬
        mask : (B, max_N) bool tensor — 유효 원자 True, 패딩 False
    """
    import torch
    feat_dim = config.GCN_NODE_FEAT_DIM
    max_n    = max(nf.shape[0] for nf, _ in graph_list)
    B        = len(graph_list)

    x_batch   = np.zeros((B, max_n, feat_dim), dtype=np.float32)
    adj_batch = np.zeros((B, max_n, max_n),    dtype=np.float32)
    mask      = np.zeros((B, max_n),           dtype=bool)

    for i, (nf, ad) in enumerate(graph_list):
        n = nf.shape[0]
        x_batch[i, :n, :]   = nf
        adj_batch[i, :n, :n] = ad
        mask[i, :n]           = True

    return (
        torch.tensor(x_batch,   dtype=torch.float32),
        torch.tensor(adj_batch, dtype=torch.float32),
        torch.tensor(mask,      dtype=torch.bool),
    )


# ─────────────────────────────────────────────────────────────
# 메인 전처리 파이프라인
# ─────────────────────────────────────────────────────────────

def load_all_data(data_path: str = config.DATA_PATH):
    """
    CSV 파일 로드 → 전처리 →
    전체 환자(Mols)에 대한 fp_all, frags_all, graphs_all, labels 반환.
    K-Fold 교차 검증을 위해 Train/Test 분리 로직은 훈련 스크립트로 위임합니다.
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

    # 출처(ref) 컬럼이 있으면 소스별 분포 출력
    if "ref" in df.columns:
        print("      [출처 분포]", dict(df["ref"].value_counts()))

    labels   = df[config.LABEL_COL].values.astype(np.int64)
    mols_all = df["mol"].tolist()

    # ── Path A: Global Fingerprint ──
    print("[3/5] Global Fingerprint 추출 (MACCS + Morgan ECFP6)")
    fp_all = np.stack([mol_to_global_fp(m) for m in mols_all])  # (N, 679)

    # ── Path B: Local Substructure (BoF) ──
    # [최적화] frags_all은 현재 GCN 파이프라인에서 사용하지 않으므로 생략.
    # BRICS 분해는 분자당 최대 3초 소요 → 전체 실행 시간의 핵심 낭비 요인.
    # 레거시 DILIAttentionModel 사용 시에는 아래 주석을 해제하고 반환값에 frags_all 추가.
    # frags_all = [
    #     mol_to_fragments(m)
    #     for m in tqdm(mols_all, desc="  Substructure", ncols=80, unit="mol")
    # ]

    # ── Path C: GCN 그래프 데이터 ──
    print("[4/4] GCN 그래프 추출 (원자 피처 + 인접 행렬, 위상 보존)")
    graphs_all = [
        mol_to_graph(m)
        for m in tqdm(mols_all, desc="  Graph", ncols=80, unit="mol")
    ]

    print(f"[완료] 총 {len(df)}개 샘플")
    print(f"      Global FP dim : {fp_all.shape[1]}")
    print(f"      class 분포  0:{(labels==0).sum()}  1:{(labels==1).sum()}")

    return fp_all, graphs_all, labels


if __name__ == "__main__":
    fp_all, graphs_all, labels = load_all_data()
    print("fp_all shape :", fp_all.shape)
    print("그래프 샘플 node_feats shape:", graphs_all[0][0].shape)
    print("그래프 샘플 adj shape:", graphs_all[0][1].shape)


