import random
import os
import argparse
import fnmatch
from tqdm import tqdm
import pandas as pd
import json
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
    parser.add_argument("--clean_file", default="", type=str, help="path to clean file (not watermarked)")
    parser.add_argument("--wm_folder", default="", type=str, help="folder, watermarked dataset")
    parser.add_argument("--alpha_list", nargs='+', default=[0.1, 0.2, 0.3, 0.4], type=float, help="the percentage to delete")
    parser.add_argument("--output_dir", default='', type=str, help="path to output folder")
    parser.add_argument("--feat_rank_json", default='', type=str, help="path to json file, which store feature ranking")
    parser.add_argument("--delete_type", default='row', choices=['row', 'column'])
    parser.add_argument("--del_col_idx", nargs='+', default=[9], type=int, help="the index of the column to delete")
    parser.add_argument("--num_col_to_del", default=1, type=int, help="the number of columns to delete")
    parser.add_argument("--num_col_to_embed", default=1, type=int, help="the number of columns to emb")
    parser.add_argument("--dataset", default='fct', help="'fct', 'winequality', 'hive', 'bioresponse', 'higgs', 'gsad'")
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'NR', 'GAHSW', 'IP'])
    parser.add_argument("--random_seed", default=42, type=int, help="random seed, so far, apply to numpy, noise matrix")
    parser.add_argument("--param_ans_d", type=true_or_false, default=False, help="If parameteric analyses for d")
    args = parser.parse_args()
    return args


def row_delete_attack(csv_file, output_file, alpha, random_seed):
    df = pd.read_csv(csv_file)
    # alpha is the percentage to delete, n is the number to sample
    n = int(len(df) * (1-alpha))
    sample = df.sample(n, random_state=random_seed)
    sample.to_csv(output_file, index=False)


def delete_column(csv_file, output_file, col_idx):
    df = pd.read_csv(csv_file)
    df.drop(df.columns[col_idx], axis=1, inplace=True)
    df.to_csv(output_file, index=False)


def get_del_col_name_feat_rank(json_file, num_col_del):
    # Load the sorted dictionary from the JSON file
    with open(json_file, 'r') as json_file:
        sorted_dict = json.load(json_file)

    # Get the last 'num_col_del' keys from the sorted dictionary
    last_keys = list(sorted_dict.keys())[-num_col_del:]

    return last_keys


def get_del_col_idx_list(num_col_emb, num_col_del):
    del_col_idx = []
    for i in range(0, num_col_del):
        if num_col_emb - i > 0:
            del_col_idx.append(f"V{num_col_emb - i}")
    return del_col_idx


def get_del_col_idx_list_fct(num_col_emb, num_col_del, headers):
    del_col_idx = []
    for i in range(1, num_col_del + 1):
        if num_col_emb - i > 0:
            del_col_idx.append(headers[num_col_emb - i])
    return del_col_idx


def get_del_col_idx_list_param_d(file_name, num_col_del):
    match = re.search(r'_emb(\d+)', file_name)
    if match:
        num_col_emb = int(match.group(1))
        del_col_idx = []
        # print(f'param ans d: num_col_embed {num_col_emb}, num_col_del {num_col_del}')
        for i in range(1, num_col_del+1):
            if num_col_emb - i > 0:
                del_col_idx.append(f"V{num_col_emb - i}")
            if num_col_emb - i == 0:
                return del_col_idx
        # print(f'del_col_idx {del_col_idx}')
        return del_col_idx


def delete_columns_feat_rank(input_file, output_file, columns_to_delete):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Delete the specified columns
    for col_name in columns_to_delete:
        df.drop(columns=col_name, inplace=True)

    # print('After column deletion, df.shape', df.shape)
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = args_parse()
    random.seed(args.random_seed)
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

    print(f"\ndeletion attack, type:  {args.delete_type}. Processing ...")
    for each_file in tqdm(file_list):
        csv_file_name = os.path.basename(each_file).split('.')[0]

        if args.delete_type == "row":
            for alpha in args.alpha_list:
                file_name = "{}_delete_{}_{:.2f}_0.00_{}.csv".format(csv_file_name, args.delete_type, alpha, args.method)
                if not os.path.exists(os.path.join(args.output_dir, file_name)):
                    row_delete_attack(each_file, os.path.join(args.output_dir, file_name), alpha, args.random_seed)
                else:
                    print(f'file {file_name} exists')
        elif args.delete_type == "column":  # column
            file_name = "{}_delete_{}_{}_{}.csv".format(csv_file_name, args.delete_type, args.num_col_to_del, args.method)
            # version 1
            # columns_to_delete = get_del_col_name_feat_rank(args.feat_rank_json, args.num_col_to_del)

            # version 2
            # Assume the dataset is sorted based on feature importance by default
            if args.param_ans_d:
                columns_to_delete = get_del_col_idx_list_param_d(csv_file_name, args.num_col_to_del)
            else:
                if args.dataset == 'gsad':
                    columns_to_delete = get_del_col_idx_list(args.num_col_to_embed, args.num_col_to_del)
                elif args.dataset == 'fct':
                    headers = list(pd.read_csv(args.clean_file).columns)
                    columns_to_delete = get_del_col_idx_list_fct(args.num_col_to_embed, args.num_col_to_del, headers)
                else:
                    columns_to_delete = get_del_col_idx_list(args.num_col_to_embed, args.num_col_to_del)

            # print(f'columns_to_delete {columns_to_delete}')

            if not os.path.exists(os.path.join(args.output_dir, file_name)):
                if args.param_ans_d:
                    match_num_col_to_embed = re.search(r'_emb(\d+)', file_name)
                    if match_num_col_to_embed:
                        num_col_to_embed = int(match_num_col_to_embed.group(1))
                    else:
                        num_col_to_embed = 0
                    if args.num_col_to_del < num_col_to_embed:
                        try:
                            delete_columns_feat_rank(each_file, os.path.join(args.output_dir, file_name),
                                                     columns_to_delete)
                        except:
                            pass
                    else:
                        pass
                else:
                    delete_columns_feat_rank(each_file, os.path.join(args.output_dir, file_name), columns_to_delete)
            else:
                print(f'file {file_name} exists')
        else:
            assert False, "check delete_type"
