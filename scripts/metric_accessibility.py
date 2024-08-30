import json
from sklearn.metrics import classification_report, accuracy_score

dirname = "playground/accessibility_data"

def predict(model_name, files):
    # files = ["sample_test_aaaa"]
    # model_name = "llama3_lora"

    labels = ['negative', 'neutral', 'positive', 'unrelated']

    total_y_test = []
    total_y_pred = []
    print(f"Model: {model_name}")
    print(f"Files: {files}")

    for file in files:
        with open(f"{dirname}/{file}.jsonl", 'r') as infile:
            lines = infile.readlines()
        
        y_test = [json.loads(line.strip())["annotation"] for line in lines]

        with open(f"{dirname}/{file}-predict-{model_name}.jsonl", 'r') as infile:
            predict_lines = infile.readlines()
        
        y_pred = [json.loads(line.strip())["label"] for line in predict_lines]

        total_y_test.extend(y_test)
        total_y_pred.extend(y_pred)

    print(classification_report(y_test, y_pred, target_names=labels))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test-files", type=str, nargs="+", default=[])
    args = parser.parse_args()
    predict(model_name=args.model_name, files=args.test_files)
    