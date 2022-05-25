import argparse


def get_argparse_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/path/to/model", type=str)
    parser.add_argument("--model_name", default="t5", type=str)
    parser.add_argument("--dataset", default="instruction", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--min_length", default=30, type=int)
    parser.add_argument("--sample_num", default=128, type=int)
    parser.add_argument("--eval_num", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--data_dir", default="/path/to/dataset/", type=str)
    parser.add_argument("--max_source_length", default=512, type=int)
    parser.add_argument("--max_target_length", default=100, type=int)
    parser.add_argument("--best_metric", default="micro_f1", type=str)
    parser.add_argument("--split_dataset", action="store_true", dest="split_dataset",
                        help="whether split(update) train/dev dataset")
    parser.add_argument("--format_data", action="store_true", dest="format_data",
                        help="whether format data to instruction from raw data")
    return parser
