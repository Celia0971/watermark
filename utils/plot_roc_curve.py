import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import auc
from pathlib import Path

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_res", type=str, default="", help="The csv file that saves the extract results of our method")
    parser.add_argument("--extract_res_obt", type=str, default="")
    parser.add_argument("--extract_res_nr", type=str, default="")
    parser.add_argument("--extract_res_ip", type=str, default="")
    parser.add_argument("--extract_res_gahsw", type=str, default="")
    parser.add_argument('--thresholds', nargs='+', type=int, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--attack_params', nargs='+', type=float, default=[0.7, 0.8, 0.9])
    parser.add_argument("--fig_folder", type=str, default="", help="The path to save the output figures")
    parser.add_argument("--method", type=str, default='Ours', choices=["OBT", "Ours"])
    parser.add_argument("--dataset", type=str, default='covtype', help="winequality, covtype")
    parser.add_argument("--num_wm", type=int, default=5, help="")
    parser.add_argument("--attack", default="watermarked", type=str, help="insert, delete, noise, watermarked")
    parser.add_argument("--add_info", default="wm", type=str, help="")

    args = parser.parse_args()
    return args


def cal_scores(method, res, t, num_wm, alpha):
    if method == "Ours" or method == "NR":
        fp_mask = (res['extract_res'] >= t) & (res['key_idx'] != res['watermark_tag']) & (res['attack_param1'] == alpha)
        fn_mask = (res['extract_res'] < t) & (res['key_idx'] == res['watermark_tag']) & (res['attack_param1'] == alpha)
        tp_mask = (res['extract_res'] >= t) & (res['key_idx'] == res['watermark_tag']) & (res['attack_param1'] == alpha)
        tn_mask = (res['extract_res'] < t) & (res['key_idx'] != res['watermark_tag']) & (res['attack_param1'] == alpha)
    else:
        fp_mask = (res['similarity'] >= t) & (res['key_idx'] != res['watermark_tag']) & (res['attack_param1'] == alpha)
        fn_mask = (res['similarity'] < t) & (res['key_idx'] == res['watermark_tag']) & (res['attack_param1'] == alpha)
        tp_mask = (res['similarity'] >= t) & (res['key_idx'] == res['watermark_tag']) & (res['attack_param1'] == alpha)
        tn_mask = (res['similarity'] < t) & (res['key_idx'] != res['watermark_tag']) & (res['attack_param1'] == alpha)


    fp = fp_mask.sum()
    fn = fn_mask.sum()
    tp = tp_mask.sum()
    tn = tn_mask.sum()

    sum_constant = num_wm * (num_wm+1)
    assert fp + fn + tp + tn == sum_constant, f'fp {fp}, fn {fn}, tp {tp}, tn {tn}, sum {fp + fn + tp + tn}, sum_constant {sum_constant}'

    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    scores = {
        "method": method,
        "threshold": t,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "f1_score": f1_score
    }
    print('Method {}, threshold={}, fp={}, fn={}, tp={}, tn={}, fpr={}, tpr={}'.format(method, t, fp, fn, tp, tn, fpr, tpr))
    return scores


def get_plot_info(res_file, thresholds, num_wm, method, alpha):
    res = pd.read_csv(res_file)
    fp_list = []
    fn_list = []
    tp_list = []
    tn_list = []
    precision_values = []
    recall_values = []
    fpr_values = []
    tpr_values = []

    for t in thresholds:  # for each threshold
        scores_dict = cal_scores(method, res, t, num_wm, alpha)
        fp_list.append(scores_dict['fp'])
        fn_list.append(scores_dict['fn'])
        tp_list.append(scores_dict['tp'])
        tn_list.append(scores_dict['tn'])
        precision_values.append(scores_dict['precision'])
        recall_values.append(scores_dict['recall'])
        fpr_values.append(scores_dict['fpr'])
        tpr_values.append(scores_dict['tpr'])

    # correct (ROC)
    if tpr_values[-1] != 0:
        tpr_values.append(0)
        fpr_values.append(0)
    if fpr_values[0] != 1:
        tpr_values.insert(0, 1)
        fpr_values.insert(0, 1)

    roc_auc = auc(fpr_values, tpr_values)
    results = {
        "fp_list": fp_list,
        "fn_list": fn_list,
        "tp_list": tp_list,
        "tn_list": tn_list,
        "tpr_values": tpr_values,
        "fpr_values": fpr_values,
        "roc_auc": roc_auc,
        "precision_values": precision_values,
        "recall_values": recall_values
    }

    return results


def plot_ROC_curve(attack, add_info, res_file, thresholds, num_wm, save_folder, attack_nums, method):
    for alpha in attack_nums:
        results = get_plot_info(res_file, thresholds, num_wm, method, alpha)
        plt.plot(results['fpr_values'], results['tpr_values'], label=f"{method}, AUC={results['roc_auc']:.2f}", marker='x')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve, attack: {}, type: {}, alpha: {}'.format(attack, add_info, alpha))
        plt.legend()
        save_fig_file = "ROC_curves_{}_{}_{}_{}.png".format(method, attack, add_info, alpha)
        plt.savefig(os.path.join(save_folder, save_fig_file))
        plt.close()
        # plt.show()


def plot_ROC_curves(attack, add_info, res_file_ours, res_file_obt, res_file_nr, res_file_ip, res_file_gahsw, thresholds, num_wm, save_folder, attack_nums):
    for alpha in attack_nums:
        res_ours = get_plot_info(res_file_ours, thresholds, num_wm, "Ours", alpha)
        plt.plot(res_ours['fpr_values'], res_ours['tpr_values'], label=f"Ours, AUC={res_ours['roc_auc']:.2f}", marker='x')

        res_obt = get_plot_info(res_file_obt, thresholds, num_wm, "OBT", alpha)
        plt.plot(res_obt['fpr_values'], res_obt['tpr_values'], label=f"OBT, AUC={res_obt['roc_auc']:.2f}", marker='x')

        res_nr = get_plot_info(res_file_nr, thresholds, num_wm, "NR", alpha)
        plt.plot(res_nr['fpr_values'], res_nr['tpr_values'], label=f"NR, AUC={res_nr['roc_auc']:.2f}", marker='x')

        res_ip = get_plot_info(res_file_ip, thresholds, num_wm, "IP", alpha)
        plt.plot(res_ip['fpr_values'], res_ip['tpr_values'], label=f"IP, AUC={res_ip['roc_auc']:.2f}", marker='x')

        res_gahsw = get_plot_info(res_file_gahsw, thresholds, num_wm, "GAHSW", alpha)
        plt.plot(res_gahsw['fpr_values'], res_gahsw['tpr_values'], label=f"GAHSW, AUC={res_gahsw['roc_auc']:.2f}", marker='x')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve, attack: {}, type: {}, alpha: {}'.format(attack, add_info, alpha))
        plt.legend()
        save_fig_file = "ROC_curves_{}_{}_{}.png".format(attack, add_info, alpha)
        plt.savefig(os.path.join(save_folder, save_fig_file))
        plt.close()
        # plt.show()


if __name__ == "__main__":
    args = args_parse()
    if not os.path.exists(args.fig_folder):
        os.makedirs(Path(args.fig_folder))

    attack = os.path.basename(args.extract_res)[0:-4].split('_')[-2]
    add_info = os.path.basename(args.extract_res)[0:-4].split('_')[-1]
    print(f'attack {attack}, add into {add_info}')

    # one
    plot_ROC_curve(attack, add_info, args.extract_res, args.thresholds, args.num_wm, args.fig_folder, args.attack_params, args.method)

    # several
    # plot_ROC_curves(attack, add_info, args.extract_res, args.extract_res_obt, args.extract_res_nr,
    #                 args.extract_res_ip, args.extract_res_gahsw, args.thresholds,
    #                 args.num_wm, args.fig_folder, args.attack_params)

