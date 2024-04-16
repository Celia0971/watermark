import os
import argparse
import time
import tqdm

from utils import load_data, save_data, load_eigenvectors, load_wm_params, new_dir
from core import embed_watermark


def args_parse():
    def true_or_false(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()
    parser.add_argument("--org_file", default="", type=str, help="path to clean file (not watermarked)")
    parser.add_argument("--eigenvec_file", default="", type=str, help="path to eigen vectors")
    parser.add_argument("--wm_params_json", default="", type=str, help="json file to save watermark parameters")
    parser.add_argument("--exp_root", default="./experiments", type=str, help="root of experiments, all experimental results will be saved in this folder")
    parser.add_argument("--exp_tag", default='test', help="experimental tag, the name of experiment")
    parser.add_argument("--dataset", default='covtype', help="winequality, covtype, hive, bioresponse, higgs, gas")
    parser.add_argument('--col_to_embed', type=lambda x: tuple(map(int, x.split(','))), default=(0, 9), help="column range to embed watermark，closed interval") # Attention!
    parser.add_argument("--num_wm_sets", type=int, default=12, help='the number of watermarked datasets, list(range(1,num_wm_sets+1))')
    parser.add_argument("--save_bin_info", type=true_or_false, default=False, help="If save bin info")
    parser.add_argument("--param_ans_d", type=true_or_false, default=False, help="If parameteric analyses for d")

    args = parser.parse_args()
    return args


def get_num_col_to_emb(col_to_embed):
    num_col_to_emb = col_to_embed[1]-col_to_embed[0]+1
    return num_col_to_emb


def embed_pipeline(args):
    print(f'\n==> embed watermark, save result at {os.path.join(args.exp_root, args.exp_tag, args.dataset, "embed")}')

    feat_to_emb = load_data(args.org_file, args.col_to_embed)
    eigenvectors = load_eigenvectors(feat_to_emb, args.eigenvec_file)
    embed_path = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed')
    wm_save_path = os.path.join(embed_path, 'wm_datasets')
    keys_save_path = os.path.join(embed_path, 'storedkeys')

    new_dir(wm_save_path)
    new_dir(keys_save_path)

    assert feat_to_emb.shape[1] == eigenvectors[1].shape[0]
    duration_collect = []

    if args.param_ans_d:
        wm_data_idxes = list(range(1, args.num_wm_sets + 1))
        for wm_idx in tqdm.tqdm(wm_data_idxes):
            start_time = time.time()
            wm_params = load_wm_params(args.wm_params_json, str(wm_idx), eigenvectors)
            key_idx = wm_params['key_id']
            num_col_to_emb = get_num_col_to_emb(args.col_to_embed)
            storedfile = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_keys.csv'
            data_wm = embed_watermark(feat_to_emb, wm_params, os.path.join(keys_save_path, storedfile),
                                      args.save_bin_info)
            wm_file = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_emb' + str(num_col_to_emb) + '_wm.csv'
            save_data(args.org_file, data_wm, os.path.join(wm_save_path, wm_file), args.col_to_embed)
            end_time = time.time()
            duration_collect.append(end_time - start_time)
    else:
        key_idxes = list(range(1, args.num_wm_sets+1))
        for key_idx in tqdm.tqdm(key_idxes):
            start_time = time.time()
            wm_params = load_wm_params(args.wm_params_json, str(key_idx), eigenvectors)
            storedfile = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_keys.csv'
            data_wm = embed_watermark(feat_to_emb, wm_params, os.path.join(keys_save_path, storedfile), args.save_bin_info)
            wm_file = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_wm.csv'
            save_data(args.org_file, data_wm, os.path.join(wm_save_path, wm_file), args.col_to_embed)
            end_time = time.time()
            duration_collect.append(end_time-start_time)

    mean_duration = sum(duration_collect) / len(duration_collect)
    print('\n==> Embedding watermark into the dataset takes {:.2f}s in average.'.format(mean_duration))  # 2.51s


if __name__ == "__main__":
    args = args_parse()
    embed_pipeline(args)




