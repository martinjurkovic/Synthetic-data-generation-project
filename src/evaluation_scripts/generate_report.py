import os
import json
import pandas as pd

def insert_single_table_metrics(json_obj, method_name, metric_name, df, table):
    dataset_name = json_obj["dataset_name"]
    metrics = json_obj["metrics"]["single_table"][table]
    
    # Create an empty DataFrame with column names
    
    # Iterate over the metrics and insert them into the DataFrame
    metric_values = metrics[metric_name]
    mean = metric_values["mean"]
    std = metric_values["std"]
    min_val = metric_values["min"]
    max_val = metric_values["max"]
    median = metric_values["median"]
    
    # Create a dictionary with the metric values
    row = {
        "Method": method_name,
        "Mean": mean,
        "Std": std,
        "Min": min_val,
        "Max": max_val,
        "Median": median
    }
        
    # Append the row to the DataFrame using concat
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df


def load_results(dataset, method):
    results_path = f"metrics_report/{dataset}_{method}.json"
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r") as f:
        json_obj = json.load(f)
    return json_obj


def pretty(method):
    method = method.replace('_', ' ')
    words = method.split(' ')
    words = [word.capitalize() for word in words]
    return ' '.join(words)

metric_name = 'xgboost_detection'
datasets = ['biodegradability', 'rossmann-store-sales', 'coupon-purchase-prediction', 'telstra-competition-dataset', 'zurich', 'mutagenesis']
methods = ['SDV', 'RealTabFormer', 'RCTGAN', 'mostlyai', 'gretel']
for dataset in datasets:
    metadata_path = f"data/metadata/{dataset}_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    for table in metadata["tables"]:
        # Create a DataFrame with the metric values
        df = pd.DataFrame(columns=["Mean", "Std", "Min", "Max", "Median"])
        for method in methods:
            json_obj = load_results(dataset, method)
            if json_obj is None:
                continue
            df = insert_single_table_metrics(json_obj, method, metric_name, df, table)
        df.sort_values(by=['Mean'], inplace=True, ascending=False)
        print(df.to_latex(caption = f'{dataset.capitalize()} - {table.capitalize()} - {pretty(metric_name)}', index=False))
        print(df.to_markdown(index=False))
        
        