import os
import argparse
import fnmatch
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv
import re


def args_parse():
    def true_or_false(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", default="", help="the folder that contains all csv files that need to classify")
    parser.add_argument("--test_set", default="", help="path to test set")
    parser.add_argument("--result_dir", default='', type=str, help="dir to save classification results")
    parser.add_argument("--result_file", default='', type=str,
                        help="path to output file to save classification results")
    parser.add_argument("--dataset", default='fct',
                        choices=['fct', 'winequality', 'bioresponse', 'higgs', 'gsad', 'hive'])
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'IP', 'GAHSW', 'NR'])
    parser.add_argument("--withid", type=true_or_false, default=False, help="")
    args = parser.parse_args()
    return args


def classification(train_set, test_set, dataset='fct', withid=False):
    train_data = pd.read_csv(train_set)
    if withid and re.search(r'_keyid(\d+)', train_set):
        X_train = train_data.iloc[:, 1:-1]
    else:
        X_train = train_data.iloc[:, 0:-1]

    y_train = train_data.iloc[:, -1]

    # Count the number of rows before outlier removal
    initial_row_count = X_train.shape[0]

    # re_sample, unbalanced class distribution
    # sm = SMOTE(random_state=0)
    # X, y = sm.fit_resample(X, y)

    # # data preprocessing
    # if dataset == "fct":
    #     pass
    # else:
    #     # Remove outliers (you can replace this with your preferred outlier detection method)
    #     z_scores = ((X_train - X_train.mean()) / X_train.std()).abs()
    #     X_train = X_train[(z_scores < 3).all(axis=1)]
    #     y_train = y_train[(z_scores < 3).all(axis=1)]
    #
    #     # Count the number of rows after outlier removal
    #     final_row_count = X_train.shape[0]
    #
    #     # Calculate and print the number of outliers removed
    #     outliers_removed = initial_row_count - final_row_count
    #     # print(f"Number of outliers removed: {outliers_removed}")  # z_scores smaller, more points remove

    # Min-max feature scaling
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)

    # classifier
    classifier = LogisticRegression(max_iter=5000)

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
    test_data = pd.read_csv(test_set)
    X_test = test_data.iloc[:, 0:-1]
    # X_test = scaler.transform(X_test)
    y_test = test_data.iloc[:, -1]

    y_test_pred = classifier.predict(X_test)
    metrics_test = get_classification_report(y_test, y_test_pred)
    print('Test set performance. Accuracy {:.4f}, F1 Score {:.4f}'.
          format(metrics_test['accuracy'], metrics_test['f1 score']))

    for key, value in metrics_test.items():
        metrics['test_' + key] = value

    return metrics


def get_classification_report(y_test, predictions, average="macro"):
    acc = accuracy_score(y_test, predictions)
    pre = precision_score(y_test, predictions, average=average, zero_division=1)
    rec = recall_score(y_test, predictions, average=average)
    f1 = f1_score(y_test, predictions, average=average)

    # Confusion Matrix
    # cm = confusion_matrix(y_test, predictions)
    # cm_np = np.array(cm)
    # cm_vector = cm_np.flatten()
    # Reshape the vector back into a matrix
    # new_matrix = np.reshape(vector, matrix.shape)

    metrics = {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1 score": f1,
        # "confusion matrix": cm_vector
    }

    return metrics


def record_res(train_set, metrics_dict, filename):
    info_dict = parse_info(train_set)

    all_info = {**info_dict, **metrics_dict}
    headers = list(all_info.keys())

    # Check if file already exists
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write header row if file is newly created
        if not file_exists:
            writer.writeheader()

        writer.writerow(all_info)


def parse_info(file_path, dataset, method):
    attack_type = Path(file_path).parent.parent.name
    attack_add_info = Path(file_path).parent.name
    file_name = Path(file_path).stem
    info = file_name.split('_')

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
        else:
            key_idx = 0

        alpha = info[-3]
        beta = info[-2]

    if attack_type == "embed" and attack_add_info == "wm_datasets":  # watermarked, no attack
        attack_type = "none"
        attack_add_info = "none"
        alpha = 0
        beta = 0

    info_dict = {
        "dataset": dataset,
        "attack": attack_type,
        "add_info": attack_add_info,
        "alpha": float(alpha),
        "beta": float(beta),
        "method": method,
        "watermarked": watermarked,
        "file_name": file_name,
        "key_idx": key_idx,
        "file_path": os.path.basename(file_path)
    }
    # print('info_dict', info_dict)
    return info_dict


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


if __name__ == "__main__":
    args = args_parse()

    file_list = []
    if fnmatch.fnmatch(args.target_dir, "*.csv"):
        file_list.append(args.target_dir)
    else:  # a dir
        for file_name in os.listdir(args.target_dir):
            if fnmatch.fnmatch(file_name, "*.csv"):
                file_list.append(os.path.join(args.target_dir, file_name))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    for each_file in sorted(file_list):
        print(f'\n==> classification, dataset: {os.path.basename(each_file)}')
        metrics = classification(each_file, args.test_set, args.dataset, args.withid)
        info = parse_info(each_file, args.dataset, args.method)
        record_results(os.path.join(args.result_dir, args.result_file), info, metrics)

