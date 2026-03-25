import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader

from dataset import load_data, mol_to_graph, smiles_to_fp
from registry import MODEL_REGISTRY

from torch_geometric.loader import DataLoader
import torch


def evaluate(y_true, y_pred):
    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
    }


def run_gnn(df):
    graphs = []
    for _, row in df.iterrows():
        g = mol_to_graph(row["smiles"])
        if g is None:
            continue
        g.y = torch.tensor([row["label"]])
        graphs.append(g)

    train_data, test_data = train_test_split(graphs, test_size=0.2)

    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = MODEL_REGISTRY["gnn"]()
    model.train(train_loader)

    preds = model.predict(test_loader)
    y_true = [d.y.item() for d in test_data]

    return y_true, preds


def run_ml(df, model_name):
    X = np.array([smiles_to_fp(s) for s in df["smiles"]])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MODEL_REGISTRY[model_name]()
    model.train((X_train, y_train))

    preds = model.predict(X_test)

    return y_test, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    df = load_data()

    if args.model == "stackdili":
        model = MODEL_REGISTRY["stackdili"]()
        model.train(None)
        return
    else:
        y_true, preds = run_ml(df, args.model)

    metrics = evaluate(y_true, preds)

    print(f"\n=== {args.model} ===")
    print(metrics)


if __name__ == "__main__":
    main()