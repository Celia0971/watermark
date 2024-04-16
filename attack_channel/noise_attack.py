import fnmatch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import random


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_file", default="", type=str, help="path to clean file (not watermarked)")
    parser.add_argument("--wm_folder", default="", type=str, help="folder, watermarked dataset")
    parser.add_argument("--noise_mean", default=0, type=float, help="noise_mean")
    parser.add_argument("--noise_std", nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], type=float, help="Gaussian noise std")
    parser.add_argument("--noise_range", nargs='+', default=[0.1, 0.2], type=float, help="Uniform noise U[-x, x]")
    parser.add_argument("--noise_constants_list", nargs='+', type=float, default=[0.2, 0.4])
    parser.add_argument("--noise_list", nargs='+', type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], help='rand noise')
    parser.add_argument("--output_dir", default='', type=str, help="path to output folder")
    parser.add_argument("--noise_type", default='gaussian', choices=['gaussian', 'uniform', 'rand', 'constant'])
    parser.add_argument("--dataset", default='fct', help="'fct', 'gsad', 'winequality', 'hive', 'higgs'")
    parser.add_argument('--column_range', type=lambda x: tuple(map(int, x.split(','))), default=(10, 19), help="column range to embed watermark， closed interval")
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'IP', 'GAHSW', 'NR'])
    parser.add_argument("--random_seed", default=42, type=int, help="random seed, so far, apply to numpy, noise matrix")
    args = parser.parse_args()
    return args


def add_gaussian_noise(df, cols_to_add_noise, output_file, noise_mean, noise_std):
    df[cols_to_add_noise] += np.random.normal(noise_mean, noise_std, size=df[cols_to_add_noise].shape)
    df.to_csv(output_file, index=False)


def add_uniform_noise(df, cols_to_add_noise, output_file, noise_range_l, noise_range_r):
    df[cols_to_add_noise] += np.random.uniform(low=noise_range_l, high=noise_range_r, size=df[cols_to_add_noise].shape)
    df.to_csv(output_file, index=False)


def noise_addition(df, cols_to_add_noise, output_file, noise_norm):
    # produce noise matrix
    noise_mat = np.random.rand(*df[cols_to_add_noise].shape)
    noise_mat = noise_mat - np.mean(noise_mat, axis=1, keepdims=True)
    noise_mat = noise_mat / np.sqrt(np.sum(noise_mat ** 2, axis=1, keepdims=True))
    noise_mat = noise_norm * noise_mat

    df[cols_to_add_noise] += noise_mat
    df.to_csv(output_file, index=False)


def add_constant_noise(df, cols_to_add_noise, output_file, each_constant):
    noise_mat = np.ones(shape=df[cols_to_add_noise].shape)
    noise_mat = each_constant * noise_mat
    df[cols_to_add_noise] += noise_mat
    df.to_csv(output_file, index=False)


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

    print(f"\nnoise addition attack, noise type: {args.noise_type}. Processing ...")
    for each_file in tqdm(file_list):
        csv_file_name = os.path.basename(each_file).split('.')[0]
        df = pd.read_csv(each_file)

        cols_to_add_noise = df.columns[args.column_range[0]:args.column_range[1] + 1]
        if args.noise_type == 'gaussian':
            for each_std in args.noise_std:
                if each_std < 0.0001:
                    file_name = "{}_noise_{}_{:.2f}_{:.5f}_{}.csv".format(csv_file_name, args.noise_type, args.noise_mean,
                                                                      each_std, args.method)
                else:
                    file_name = "{}_noise_{}_{:.2f}_{}_{}.csv".format(csv_file_name, args.noise_type, args.noise_mean, each_std, args.method)
                output_file = os.path.join(args.output_dir, file_name)
                if not os.path.exists(output_file):
                    add_gaussian_noise(df, cols_to_add_noise, output_file, args.noise_mean, each_std)
                else:
                    print(f'file {file_name} exists')
        elif args.noise_type == 'uniform':
            for each_param in args.noise_range:
                if each_param < 0.0001:
                    file_name = "{}_noise_{}_{:.5f}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_param,
                                                                    args.method)
                else:
                    file_name = "{}_noise_{}_{}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_param, args.method)

                output_file = os.path.join(args.output_dir, file_name)
                if not os.path.exists(output_file):
                    add_uniform_noise(df, cols_to_add_noise, output_file, -each_param, each_param)
                else:
                    print(f'file {file_name} exists')
        elif args.noise_type == 'rand':
            for each_norm in args.noise_list:
                if each_norm < 0.0001:
                    file_name = "{}_noise_{}_{:.5f}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_norm, args.method)
                else:
                    file_name = "{}_noise_{}_{}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_norm,
                                                                    args.method)

                output_file = os.path.join(args.output_dir, file_name)
                if not os.path.exists(output_file):
                    noise_addition(df, cols_to_add_noise, output_file, each_norm)
                else:
                    print(f'file {file_name} exists')
        elif args.noise_type == 'constant':
            for each_constant in args.noise_constants_list:
                if each_constant < 0.0001:
                    file_name = "{}_noise_{}_{:.5f}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_constant, args.method)
                else:
                    file_name = "{}_noise_{}_{}_0.00_{}.csv".format(csv_file_name, args.noise_type, each_constant, args.method)
                output_file = os.path.join(args.output_dir, file_name)
                if not os.path.exists(output_file):
                    add_constant_noise(df, cols_to_add_noise, output_file, each_constant)
                else:
                    print(f'file {file_name} exists')
        else:
            assert False, "check noise_type"





