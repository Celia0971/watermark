import argparse
import os
import time
import fnmatch
import re

from core import extract_watermark, extract_wm_col_del_feat_rank
from utils import new_dir, get_keyidx_from_storedkey, get_params_from_susp_file, record_extract_results
from classification_channel.classification_del_col import parse_del_col_idx


def args_parse():
    def true_or_false(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="./experiments", type=str, help="experimental root")
    parser.add_argument("--exp_tag", default='test', help="experimental tag")
    parser.add_argument("--target_dir", default='embed', help="the directory that saves the suspicious files")
    parser.add_argument("--dataset", default='covtype', help="winequality, covtype, hive, bioresponse, higgs, gas")
    parser.add_argument("--attack", default="watermarked", type=str, help="insert, delete, noise, watermarked")
    parser.add_argument("--add_info", default="wm", type=str, help="")
    parser.add_argument("--susp_file", default="", type=str, help="the path to suspicious file")
    parser.add_argument('--col_to_embed', type=lambda x: tuple(map(int, x.split(','))), default=(0, 9), help="column range to embed watermark，closed interval") # Attention!
    parser.add_argument("--ori_file", default='', type=str)
    parser.add_argument("--param_ans_d", type=true_or_false, default=False, help="If parameteric analyses for d")
    parser.add_argument("--param_ans_b", type=true_or_false, default=False, help="If parameteric analyses for bin width")

    args = parser.parse_args()
    return args


def extract_pipeline(args):
    # collect all suspicious files
    csv_paths = []
    if args.susp_file != '':
        csv_paths.append(args.susp_file)
    elif args.target_dir != '':
        file_list = os.listdir(args.target_dir)
        file_list = sorted(file_list, key=lambda x: x.split("/")[-1])

        for filename in file_list:
            if fnmatch.fnmatch(filename, "*.csv"):
                csv_paths.append(os.path.join(args.target_dir, filename))
    else:
        print('Please specify target_dir or suspicious file.')
        exit()
    print(f'\n==> there are {len(csv_paths)} suspicious files to check.')

    # collect all candidate keys
    key_list = os.listdir(args.keys_folder)
    key_list = sorted(key_list, key=lambda x: x.split("/")[-1])
    candidate_keys_paths = []
    for keyname in key_list:
        if fnmatch.fnmatch(keyname, "*.csv"):
            candidate_keys_paths.append(os.path.join(args.keys_folder, keyname))
    print(f'==> there are {len(candidate_keys_paths)} candidate keys.\n')

    # extract watermark from each suspicious file using each key
    duration_collect = []
    for k in range(len(candidate_keys_paths)):  # for each key
        start_time = time.time()
        one_key = candidate_keys_paths[k]
        key_idx = get_keyidx_from_storedkey(one_key)

        for i in range(len(csv_paths)):  # for each susp_file
            one_file = csv_paths[i]
            one_file_name = os.path.splitext(os.path.basename(one_file))[0]

            fig_name = 'ext_res_using_key' + key_idx + '_' + one_file_name + '.png'

            if one_file_name.find("_delete_column_") != -1:  # contains
                del_col_idx = parse_del_col_idx(one_file_name)
                num_col_to_del = del_col_idx[0]
                # print('num_col_to_del', num_col_to_del)

                pattern = r'keyid(\d+)'
                match = re.search(pattern, one_file_name)
                if match:
                    keyid_number = int(match.group(1))
                else:
                    keyid_number = 'none'

                susp_file_info = {
                    'wm_tag': keyid_number,
                    'attack_param1': ",".join(map(str, del_col_idx)),
                    'attack_param2': 0
                }
                result = extract_wm_col_del_feat_rank(susp_file=one_file, stroedkey=one_key,
                                                      fig_file=os.path.join(args.figures_folder, fig_name),
                                                      num_col_to_del=num_col_to_del, col_range=args.col_to_embed,
                                                      param_ans_d=args.param_ans_d)
            else:
                susp_file_info = get_params_from_susp_file(one_file_name)
                result = extract_watermark(susp_file=one_file, stroedkey=one_key,
                                           fig_file=os.path.join(args.figures_folder, fig_name), column_range=args.col_to_embed)

            print('\n==> extract result')
            print(f'secret key: {os.path.basename(one_key)} \nsuspicious dataset: {one_file_name} \nextract result (probability): {result}')
            record_extract_results(os.path.join(args.res_folder, args.results_file), result, one_file, key_idx, susp_file_info)
            end_time = time.time()
            duration_collect.append(end_time - start_time)

    mean_duration = sum(duration_collect) / len(duration_collect)
    print('\n==> Extracting watermark takes {:.2f}s in average for each suspicious file. (extract {} times)'.format(mean_duration, len(duration_collect)))  # 2.51s


if __name__ == "__main__":
    args = args_parse()
    if args.target_dir == 'embed':
        args.target_dir = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed', 'wm_datasets')
    else:
        args.target_dir = os.path.join(args.exp_root, args.exp_tag, args.dataset, args.attack, args.add_info)

    args.keys_folder = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed', 'storedkeys')
    args.res_folder = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'extract', args.attack, args.add_info)
    new_dir(args.res_folder)
    args.figures_folder = os.path.join(args.res_folder, 'figures')
    new_dir(args.figures_folder)
    args.results_file = f"extract_res_{args.dataset}_{args.attack}_{args.add_info}.csv"
    extract_pipeline(args)
