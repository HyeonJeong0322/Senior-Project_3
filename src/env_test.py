import importlib

packages = [
    "numpy",
    "pandas",
    "torch",
    "rdkit",
    "torch_geometric",
    "sklearn",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "tqdm",
    "nbconvert",      # stackdili 노트북 실행용
]

print("=== 환경 체크 시작 ===\n")

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK]   {pkg}")
    except ImportError:
        print(f"[FAIL] {pkg}")

print("\n=== 상세 버전 ===")

try:
    import torch
    print("torch       :", torch.__version__)
except Exception:
    pass

try:
    import torch_geometric
    print("torch_geometric OK")
except Exception:
    pass

try:
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    print("rdkit       : OK" if mol is not None else "rdkit       : FAIL")
except Exception:
    pass

try:
    import xgboost
    print("xgboost     :", xgboost.__version__)
except Exception:
    pass

try:
    import lightgbm
    print("lightgbm    :", lightgbm.__version__)
except Exception:
    pass

try:
    import matplotlib
    print("matplotlib  :", matplotlib.__version__)
except Exception:
    pass

print("\n=== 완료 ===")
