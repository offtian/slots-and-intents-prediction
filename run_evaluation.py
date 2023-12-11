import argparse
import yaml

from semantic_parsing_dialog.evaluate import evaluate_predictions_from_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TOP-representation predictions"
    )
    parser.add_argument(
        "pred", type=str, help="file with each row having a single TOP-representation"
    )
    args = parser.parse_args()

    metrics = evaluate_predictions_from_files("data/test.tsv", args.pred)
    print(yaml.dump(metrics, allow_unicode=True, default_flow_style=False))
