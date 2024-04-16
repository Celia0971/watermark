import re
import os


def get_keyidx_from_storedkey(key_path):
    key_name = os.path.splitext(os.path.basename(key_path))[0]
    match = re.search(r'keyid(\d+)_', key_name)
    key_idx = match.group(1)
    return key_idx


def get_params_from_susp_file(file_name):
    if '_wm' in file_name:
        idx_num = re.search(r'keyid(\d+)', file_name).group(1)
        wm_tag = int(idx_num)
    else:
        wm_tag = 0

    pattern = r'(\d+\.\d+)'
    noise_norm_match = re.findall(pattern, file_name)
    if noise_norm_match:
        # convert the substring to a floating-point number
        attack_param1 = float(noise_norm_match[0])  # noise norm
        attack_param2 = float(noise_norm_match[1])
    else:
        attack_param1 = 0.0
        attack_param2 = 0.0

    params = {
        'wm_tag': wm_tag,
        'attack_param1': attack_param1,
        'attack_param2': attack_param2
    }

    return params
