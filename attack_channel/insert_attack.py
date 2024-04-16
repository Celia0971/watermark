import random
import os
import argparse
import fnmatch
from tqdm import tqdm
import pandas as pd
import numpy as np


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_file", default="", type=str, help="path to clean file (not watermarked)")
    parser.add_argument("--wm_folder", default="", type=str, help="folder, watermarked dataset")
    parser.add_argument("--remain_file", default="", type=str, help="path to remain set")
    parser.add_argument("--alpha_list", nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--beta_list", nargs='+', type=float, default=[1.0], help="1.0, cannot be 1")
    parser.add_argument("--output_dir", default='', type=str, help="path to output folder")
    parser.add_argument("--insert_type", default='concatenate', choices=['replicate', 'generate', 'concatenate'])
    parser.add_argument("--dataset", default='fct', help="'fct', 'gsad', 'winequality', 'hive', 'higgs'")
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'IP', 'NR', 'GAHSW'])
    parser.add_argument("--random_seed", default=42, type=int, help="random seed, so far, apply to numpy, noise matrix")
    parser.add_argument('--column_range', type=lambda x: tuple(map(int, x.split(','))), default=(10, 19),
                        help="column range to embed watermark， closed interval")
    args = parser.parse_args()
    return args


def alpha_beta_insert(csv_file, output_file, alpha, beta, col_range, random_seed):
    df = pd.read_csv(csv_file)
    n_existing = df.shape[0]

    n_new = int(float(alpha) * n_existing)
    random_rows = df.sample(n=n_new, replace=True, random_state=random_seed)
    copied_rows = random_rows.copy()

    for column in df.columns[col_range[0]: col_range[1]+1]:
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - (beta * std)
        upper_bound = mean + (beta * std)
        copied_rows[column] = np.random.uniform(lower_bound, upper_bound, size=n_new)

    df_processed = pd.concat([df, copied_rows], ignore_index=True)
    df_processed = df_processed.sample(frac=1.0, random_state=random_seed)
    df_processed.to_csv(output_file, index=False)


def alpha_selected_insert(csv_file, output_file, alpha):
    df = pd.read_csv(csv_file)
    ori_rows = len(df)
    num_rows = int(float(alpha) * ori_rows)
    random_rows = df.sample(n=num_rows, replace=True)
    df_processed = pd.concat([df, random_rows], ignore_index=True)
    df_processed.to_csv(output_file, index=False)


def generate_insert(input_file, alpha, beta, output_file):
    df = pd.read_csv(input_file)

    num_existing_tuples = len(df)
    num_generated_tuples = int(alpha * num_existing_tuples)

    new_tuples = []
    for i in range(num_generated_tuples):
        # Generate random values for the second to tenth columns
        rand_values = []
        for j in range(1, 11):
            mu = df.iloc[:, j].mean()
            sigma = df.iloc[:, j].std()
            rand_value = np.random.uniform(mu - beta * sigma, mu + beta * sigma)
            rand_values.append(rand_value)

        # Generate random values for the eleventh to fifty-fifth columns
        rand_binary_values = np.random.randint(0, 2, size=45)

        # Generate a random value for the fifty-sixth column
        rand_int_value = np.random.randint(0, 8)

        # Create the new tuple with the generated values
        new_tuple = [max(df.index) + i + 1] + rand_values + rand_binary_values.tolist() + [rand_int_value]

        # Add the new tuple to the list of new tuples
        new_tuples.append(new_tuple)

    # Append the new tuples to the original dataframe
    df = pd.concat([df, pd.DataFrame(new_tuples, columns=df.columns)], ignore_index=True)

    # Save the dataframe to a new CSV file
    df.to_csv(output_file, index=False)


def insert_attack(csv_file, remain_file, output_file, alpha):
    df = pd.read_csv(csv_file)
    remain_data = pd.read_csv(remain_file)
    n = int(alpha * len(df))
    idx = np.random.choice(len(remain_data), size=n, replace=False)
    sample = remain_data.iloc[idx]

    result = pd.concat([df, sample], ignore_index=True)
    result.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = args_parse()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_list = []
    for file_name in os.listdir(args.wm_folder):
        if fnmatch.fnmatch(file_name, "*.csv"):
            file_list.append(os.path.join(args.wm_folder, file_name))
    if args.clean_file != "":
        file_list.append(args.clean_file)
    else:
        print("Attention, clean file is null")

    print(f"\nInsertion attack, type:  {args.insert_type}. Processing ...")
    for each_file in tqdm(file_list):
        csv_file_name = os.path.basename(each_file).split('.')[0]

        if args.insert_type == "replicate":
            for alpha in args.alpha_list:
                file_name = "{}_insert_{}_{}_0.00_{}.csv".format(csv_file_name, args.insert_type, alpha, args.method)
                if not os.path.exists(os.path.join(args.output_dir, file_name)):
                    alpha_selected_insert(each_file, os.path.join(args.output_dir, file_name), alpha)
                else:
                    print(f'file {file_name} exists')
        elif args.insert_type == "concatenate":
            for alpha in args.alpha_list:
                file_name = "{}_insert_{}_{}_0.00_{}.csv".format(csv_file_name, args.insert_type, alpha, args.method)

                # assert args.remain_file == "", "remain_file cannot be null"
                if not os.path.exists(os.path.join(args.output_dir, file_name)):
                    insert_attack(each_file, args.remain_file, os.path.join(args.output_dir, file_name), alpha)
                else:
                    print(f'file {file_name} exists')
        elif args.insert_type == "generate":
            for alpha in args.alpha_list:
                for beta in args.beta_list:
                    file_name = "{}_insert_{}_{}_{}_{}.csv".format(csv_file_name, args.insert_type, alpha, beta, args.method)
                    if not os.path.exists(os.path.join(args.output_dir, file_name)):
                        alpha_beta_insert(each_file, os.path.join(args.output_dir, file_name), alpha, beta, args.column_range, args.random_seed)
                    else:
                        print(f'file {file_name} exists')
        else:
            assert False, "check insert_type"
