import importlib

packages = [
    "numpy",
    "pandas",
    "torch",
    "rdkit",
    "torch_geometric",
    "sklearn",
    "xgboost"
]

print("=== 환경 체크 시작 ===\n")

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except ImportError:
        print(f"[FAIL] {pkg}")

print("\n=== 상세 체크 ===")

# torch
try:
    import torch
    print("torch version:", torch.__version__)
except:
    pass

# rdkit
try:
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    print("rdkit test:", mol is not None)
except:
    pass

# torch_geometric
try:
    import torch_geometric
    print("torch_geometric OK")
except:
    pass

# xgboost
try:
    import xgboost
    print("xgboost version:", xgboost.__version__)
except:
    pass

print("\n=== 완료 ===")