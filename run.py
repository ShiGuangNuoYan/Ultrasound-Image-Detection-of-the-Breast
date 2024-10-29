from run_cla import run_classification
from run_fea import run_feature
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=False, type=str, help="feature dir")
    parser.add_argument("--cla_dir", required=False, type=str, help="image classification dir")
    parser.add_argument("--fea_model_path", required=True, type=str, help="input model path")
    parser.add_argument("--cla_model_path", required=True, type=str, help="input model path")
    args = parser.parse_args()
    run_classification(args)
    run_feature(args)
