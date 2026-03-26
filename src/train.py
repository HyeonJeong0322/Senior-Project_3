import argparse
from registry import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = MODEL_REGISTRY[args.model]()
    model.run()


if __name__ == "__main__":
    main()
