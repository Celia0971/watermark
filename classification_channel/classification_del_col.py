import os
import argparse
import fnmatch
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import csv


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", default="", help="the folder that contains all csv files that need to classify")
    parser.add_argument("--test_set", default="", help="path to test set")
    parser.add_argument("--result_dir", default='', type=str, help="dir to save classification results")
    parser.add_argument("--result_file", default='', type=str, help="path to output file to save classification results")
    parser.add_argument("--dataset", default='fct', help="'fct', 'winequality', 'hive', 'bioresponse', 'higgs', 'gsad'")
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'NR', 'GAHSW', 'IP'])

    args = parser.parse_args()
    return args


def row_exists(row, rows):
    # Check if the row exists in the list of rows
    for existing_row in rows:
        if row == existing_row:
            return True
    return False


def record_results(result_file, info, metrics):
    file_exists = os.path.isfile(result_file)

    with open(result_file, 'a', newline='') as csv_file:
        fieldnames = list(info.keys()) + list(metrics.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Read the existing rows and store them in a list
        existing_rows = []
        with open(result_file, 'r') as existing_file:
            reader = csv.reader(existing_file)
            existing_rows = list(reader)

        # Combine the dictionaries and write the data to the CSV file
        combined_dict = {**info, **metrics}
        if not row_exists(combined_dict, existing_rows):
            writer.writerow(combined_dict)
        else:
            print('row exists')


def get_classification_report(y_test, predictions, average="macro"):
    acc = accuracy_score(y_test, predictions)
    pre = precision_score(y_test, predictions, average=average, zero_division=1)
    rec = recall_score(y_test, predictions, average=average)
    f1 = f1_score(y_test, predictions, average=average)

    metrics = {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1 score": f1,
    }
    return metrics


def classification_del_col(train_set, test_set):
    train_data = pd.read_csv(train_set)
    X_train = train_data.iloc[:, 1:-1]
    y_train = train_data.iloc[:, -1]

    # re_sample, unbalanced class distribution
    # sm = SMOTE(random_state=0)
    # X, y = sm.fit_resample(X, y)

    # classifier
    classifier = LogisticRegression(max_iter=1200)
    classifier.fit(X_train, y_train)

    metrics = {}

    # predict on train set
    y_train_pred = classifier.predict(X_train)
    metrics_train = get_classification_report(y_train, y_train_pred)
    print('Train set performance. Accuracy {:.4f}, F1 Score {:.4f}'.
          format(metrics_train['accuracy'], metrics_train['f1 score']))

    for key, value in metrics_train.items():
        metrics['train_' + key] = value

    # predict on test set
    col_del_list = parse_del_col_idx(train_set)
    test_data = process_test_set(test_set, col_del_list)
    X_test = test_data.iloc[:, 1:-1]
    y_test = test_data.iloc[:, -1]

    y_test_pred = classifier.predict(X_test)
    metrics_test = get_classification_report(y_test, y_test_pred)
    print('Test set performance. Accuracy {:.4f}, F1 Score {:.4f}'.
          format(metrics_test['accuracy'], metrics_test['f1 score']))

    for key, value in metrics_test.items():
        metrics['test_' + key] = value

    return metrics


def parse_del_col_idx(file):
    csv_file_name = os.path.basename(file).split('.')[0]
    method = csv_file_name.split('_')[-1]
    match = re.search(rf'column_([\d_]+)_{method}', csv_file_name)
    if match:
        numbers_str = match.group(1)
        numbers_list = [int(num) for num in numbers_str.split('_')]
    else:
        print("Attention, No match found")
        numbers_list = []
        exit()
    # print(f'numbers_list {numbers_list}')
    return numbers_list


def process_test_set(file, col_del_list):
    df = pd.read_csv(file)
    # print('df(test set)', df)

    # Drop the specified columns based on their indexes
    df.drop(df.columns[col_del_list], axis=1, inplace=True)

    return df


def parse_info_del_col(file_path):
    attack_type = Path(file_path).parent.parent.name
    attack_add_info = Path(file_path).parent.name
    file_name = Path(file_path).stem
    info = file_name.split('_')
    dataset = info[0]
    method = info[-1]

    if "wm" not in info:  # clean data
        watermarked = "false"
        key_idx = "none"
        if attack_type == "wm_attacks":  # no watermark, no attack
            attack_type = "none"
            attack_add_info = "none"
            method = "none"
            alpha = 0
            beta = 0
        elif attack_type == "":
            attack_type = "none"
            attack_add_info = "none"
            method = "none"
            alpha = 0
            beta = 0
    else:
        watermarked = "true"
        match = re.search(r'keyid(\d+)', file_name)
        if match:
            key_idx = match.group(1)
            if method == "wm":
                method = "Ours"
        else:
            if method == "SF":
                key_idx = info[2]
            else:
                key_idx = "none"
                method = "SF"

    if attack_type == "embed" and attack_add_info == "wm_datasets":  # watermarked, no attack
        attack_type = "none"
        attack_add_info = "none"
        alpha = 0
        beta = 0
        if method == "SF":
            key_idx = info[2]

    valid_attacks = ['delete', 'alter', 'insert', 'noise', 'none']
    assert attack_type in valid_attacks, "Error, check attacks"

    valid_methods = ['Ours', 'OBT', 'SF', "none", "NR", "GAHSW"]
    assert method in valid_methods, file_path

    col_del_list = parse_del_col_idx(file_path)
    col_del_str = ",".join(map(str, col_del_list))

    info_dict = {
        "dataset": info[0],
        "attack": attack_type,
        "add_info": attack_add_info,
        "alpha": col_del_str,
        "beta": 0,
        "method": method,
        "watermarked": watermarked,
        "file_name": file_name,
        "key_idx": key_idx,
        "file_path": file_path
    }
    # print('info_dict', info_dict)
    return info_dict



if __name__ == "__main__":
    args = args_parse()

    file_list = []
    if fnmatch.fnmatch(args.target_dir, "*.csv"):  # a single file
        file_list.append(args.target_dir)
    else:  # a dir
        for file_name in os.listdir(args.target_dir):
            if fnmatch.fnmatch(file_name, "*.csv"):
                file_list.append(os.path.join(args.target_dir, file_name))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    for each_file in file_list:
        print(f'\n==> classification, dataset: {os.path.basename(each_file)}')
        metrics = classification_del_col(each_file, args.test_set)
        info = parse_info_del_col(each_file)
        record_results(os.path.join(args.result_dir, args.result_file), info, metrics)
