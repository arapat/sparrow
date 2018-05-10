import pandas as pd
import os
import pickle
import sys
import numpy as np
from io import StringIO

BASE_DIR = "/mnt/rust-boost"


def create_dataframe(csv_string, col_names):
    return pd.read_csv(StringIO(csv_string),
                       names=col_names, header=None)


def filter_logs(csv_file, keyword):
    with open(csv_file) as f:
        return ''.join([line for line in f if keyword in line]).strip()


def remove_dict(raw_input):
    assert('"' in raw_input)
    return '\n'.join([
        line[:line.find(', "')].strip() for line in raw_input.split('\n')
    ])


def get_new_tree_info(log_file):
    logs = filter_logs(log_file, "new-tree-info")
    # logs = remove_dict(logs)

    formatted = []
    for line in logs.split('\n'):
        line = line.strip()
        line = line[:line.find(', "')]
        formatted.append(line)
    logs = '\n'.join(formatted)
    cols = ["level", "time", "module", "log-type",
            "tree-id", "num-scanned", "gamma", "sum-gamma-squared"]  # , "tree"]
    return create_dataframe(logs, cols)


def get_performance(log_file):
    logs = filter_logs(log_file, "boosting_speed")

    cols = ["level", "time", "module", "log-type",
            "overall-duration", "overall-count", "overall-speed",
            "learner-duration", "learner-count", "learner-speed"]
    return create_dataframe(logs, cols)


def get_true_z(log_file):
    logs = filter_logs(log_file, "validate-only")

    cols = ["level", "time", "module", "log-type",
            "tree-id", "true-z", "auprc"]
    return create_dataframe(logs, cols)


def get_network_out(log_file):
    logs = filter_logs(log_file, "network-to-send-out")

    cols = ["level", "time", "module", "log-type",
            "local-name", "local-index", "score"]
    return create_dataframe(logs, cols)


def get_network_in(log_file):
    logs = filter_logs(log_file, "message-received")
    logs = remove_dict(logs)
    cols = ["level", "time", "module", "log-type",
            "local-name", "local-index", "remote-name", "remote-idx",
            "remote-ip", "score", "json-len"]  # , "model"]
    return create_dataframe(logs, cols)


def get_model_replace(log_file):
    logs = filter_logs(log_file, "model-replaced")

    cols = ["level", "time", "module", "log-type",
            "remote-score", "local-score", "remote-model-len", "local-model-len"]
    return create_dataframe(logs, cols)


def get_df(base_path, idx):
    log_path = os.path.join(base_path, "run-network.log")
    validate_path = os.path.join(base_path, "validate.log")

    if not os.path.isfile(log_path):
        print("Missing logging file on node {}".format(idx))
        return

    if not os.path.isfile(validate_path):
        print("Missing validation file on node {}".format(idx))
        return

    trees = get_new_tree_info(log_path)
    trees["node"] = idx

    true_z = get_true_z(validate_path)
    true_z["node"] = idx

    model_replace = get_model_replace(log_path)
    model_replace["node"] = idx

    # speed = get_performance(log_path)
    # speed["node"] = idx

    # network_in = get_network_in(log_path)
    # network_in["node"] = idx

    # network_out = get_network_out(log_path)
    # network_out["node"] = idx


    trees["estimated-z"] = np.exp(-trees["sum-gamma-squared"])
    trees = trees.drop(columns=["level", "module", "log-type", "sum-gamma-squared"])
    true_z = true_z.drop(columns=["level", "module", "log-type", "time"])
    model_replace = model_replace.drop(columns=["level", "module", "log-type"])

    tree_info = trees.merge(true_z, on=["tree-id"], how="outer") \
                 .merge(
                     model_replace.rename(columns={"remote-model-len": "tree-id"}),
                     on=["tree-id"], how="outer"
                 ).sort_values(by=['tree-id'])
    tree_info["time"] = tree_info["time_x"].fillna(tree_info["time_y"])
    tree_info["time"] -= tree_info["time"].iloc[0]
    tree_info = tree_info.drop(
        columns=["time_x", "time_y"])
    tree_info[["estimated-z", "true-z", "auprc"]] = \
        tree_info[["estimated-z", "true-z", "auprc"]].fillna(method="pad")

    # speed = speed.drop(columns=["level", "module", "log-type"])
    # network_in = network_in.drop(columns=["level", "module", "log-type"])
    # network_out = network_out.drop(columns=["level", "module", "log-type"])

    to_save = [trees, true_z, model_replace, tree_info]
    names = ["trees", "true_z", "model_repalce", "tree_info"]
    for var, name in zip(to_save, names):
        path = os.path.join(base_path, "{}-{}.pkl".format(name, idx))
        with open(path, 'wb') as f:
            pickle.dump(var, f)


def main(base_dir, idx):
    get_df(base_dir, idx)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: gen-pandas.py <machine-id>")
        sys.exit(1)
    idx = int(sys.argv[1])
    main(BASE_DIR, idx)
