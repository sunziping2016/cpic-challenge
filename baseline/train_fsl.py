import os
import sys
import time
import random
import logging

import transformers
from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import is_main_process

from utils.tuning_argparse import get_argparse_base
from utils.utils import write2json, seed_everything, load_json
from metrics.metrics import datatype2metrics
from instructions.instruction_model import init_baseline_model
from instructions.instruction_templates import instruction_format
from instructions.instruction_utils import DataTrainingArguments, ModelArguments, format_data

args = get_argparse_base()
args = args.parse_args()
arg_dict = args.__dict__

logger = logging.getLogger(__name__)

dataset_name = arg_dict["dataset"]
model_name = arg_dict["model_name"]
outdir_1 = "../results"
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)

outdir = outdir_1 + "/" + time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_{}_{}".format(model_name.upper(), dataset_name.upper())
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed_everything(arg_dict["seed"])

# 如果需要切分训练-验证集，该参数设置为True
# 以instruction-tuning的形式格式化数据
# 注意修改要切分的数据源和存储文件
# 可以根据需要自定义切分验证集的方式
if arg_dict["split_dataset"]:
    train_data = load_json(os.path.join(arg_dict["data_dir"], "train_data.json"))
    train_data = instruction_format(train_data)
    EVAL_NUM = arg_dict["eval_num"]
    random.shuffle(train_data)
    write2json(train_data[:-EVAL_NUM], os.path.join(arg_dict["data_dir"], "{}_train.json".format(dataset_name)), "train data")
    write2json(train_data[-EVAL_NUM:], os.path.join(arg_dict["data_dir"], "{}_dev.json".format(dataset_name)), "dev data")
    test_data = load_json(os.path.join(arg_dict["data_dir"], "test_data_A.json"))
    test_data = instruction_format(test_data)
    write2json(test_data, os.path.join(arg_dict["data_dir"], "{}_test.json".format(dataset_name)), "test data")

args = [
    "--model_name_or_path", arg_dict["model_path"],
    "--do_train", "--do_eval", "--do_predict",
    "--train_file", os.path.join(arg_dict["data_dir"], "{}_train.json".format(dataset_name)),
    "--validation_file", os.path.join(arg_dict["data_dir"], "{}_dev.json".format(dataset_name)),
    "--test_file", os.path.join(arg_dict["data_dir"], "{}_test.json".format(dataset_name)),
    "--output_dir", outdir,
    "--per_device_train_batch_size", str(arg_dict["batch_size"]),
    "--per_device_eval_batch_size", str(arg_dict["batch_size"]),
    "--overwrite_output_dir",
    "--max_source_length", str(arg_dict["max_source_length"]),
    "--max_target_length", str(arg_dict["max_target_length"]),
    "--predict_with_generate=1",
    "--seed", str(arg_dict["seed"]),
    "--num_train_epochs", str(arg_dict["epoch"]),
    "--save_strategy", "no",
    "--evaluation_strategy", "epoch",
    "--learning_rate", str(arg_dict["lr"]),
    "--metric_for_best_model", arg_dict["best_metric"],
    "--save_total_limit", "1"
]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

datasets = {}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    datasets[key] = format_data(data_files[key])
eval_data_raw = load_json(data_files["validation"])
test_data_raw = load_json(data_files["test"])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small target:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/Evaluation parameters %s", training_args)

tokenizer, model = init_baseline_model(model_args, model_name.upper(), data_args)

instruction_column = "instruction"
target_column = "target"
column_names = datasets["train"].column_names
max_target_length = data_args.val_max_target_length
padding = False


def preprocess_function(examples):
    inputs = examples[instruction_column]
    targets = examples[target_column]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if training_args.do_train:
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets["validation"]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)
eval_times = 0


def get_metrics_results(eval_preds):
    global eval_times

    def reconstruct_predictions(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eval_res = []
        for raw_data, pred in zip(eval_data_raw, decoded_preds):
            raw_data["prediction"] = pred.strip().replace(" ", "")
            eval_res.append(raw_data)
        return eval_res

    def get_golden_predictions(data_list, data_type):
        goldens = dict()
        predictions = dict()
        for sample in data_list:
            if data_type.lower() == "ner":
                goldens[sample["ID"]] = sample["entities"]
                _predictions = sample["prediction"].split("，")
                tmp = []
                for prediction_spans in _predictions:
                    for pred in prediction_spans.split(","):
                        tmp.append(pred)
                predictions[sample["ID"]] = list(set(tmp))
            else:
                goldens[sample["ID"]] = sample["target"]
                predictions[sample["ID"]] = sample["prediction"]
        return goldens, predictions

    predictions = reconstruct_predictions(eval_preds)
    dataname_metrics = dict()
    write2json(predictions, os.path.join(training_args.output_dir, "eval_results_{}.json".format(eval_times)), "{} times eval results".format(eval_times))
    eval_times += 1
    for sample in predictions:
        dataname = "_".join(sample["ID"].split("_")[:-1])
        data_predictions = dataname_metrics.get(dataname, {
            "metric": 0.,
            "data_type": sample["data_type"],
            "data_list": [],
        })
        data_predictions["data_list"].append(sample)
        dataname_metrics[dataname] = data_predictions
    all_metrics = {
        "macro_f1": 0.,
        "micro_f1": 0.,
        "eval_num": 0,
    }
    for dataname, data_predictions in dataname_metrics.items():
        goldens, predictions = get_golden_predictions(data_predictions["data_list"], data_predictions["data_type"])
        metric = datatype2metrics[dataname_metrics[dataname]["data_type"]]()
        dataname_metrics[dataname]["metric"] = metric.calc_metric(golden=goldens, predictions=predictions)
        print("{}-{}: {:.5f}({} samples)".format(
            data_predictions["data_type"],
            dataname,
            dataname_metrics[dataname]["metric"],
            len(data_predictions["data_list"])
        ))
        all_metrics["macro_f1"] += dataname_metrics[dataname]["metric"]
        all_metrics["micro_f1"] += dataname_metrics[dataname]["metric"]*len(data_predictions["data_list"])
        all_metrics["eval_num"] += len(data_predictions["data_list"])
        all_metrics[dataname] = dataname_metrics[dataname]["metric"]
    all_metrics["macro_f1"] /= len(dataname_metrics.keys())
    all_metrics["micro_f1"] /= all_metrics["eval_num"]
    return all_metrics


def save_generate_results(trainer, file_name="test_generations"):
    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
            test_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            test_res = []
            for raw_data, pred in zip(test_data_raw, test_preds):
                raw_data["prediction"] = pred.strip().replace(" ", "")
                test_res.append(raw_data)
            output_test_preds_file = os.path.join(training_args.output_dir, "{}.json".format(file_name))
            write2json(test_res, output_test_preds_file, file_name)


class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        last_metric = state.log_history[-1]["eval_micro_f1"]
        if state.best_metric is None or state.best_metric < last_metric:
            # save
            state.best_metric = last_metric
            control.should_save = True
        else:
            control.should_save = False
        return control


print("Training Args: {}".format(training_args))
# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=get_metrics_results if training_args.predict_with_generate else None,
    callbacks=[TestCallback],
)
try:
    print(trainer._max_length)
except Exception as e:
    pass
try:
    print(trainer.model.config.max_length)
except Exception as e:
    pass
print("*"*99)


# Training
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    save_generate_results(trainer, "trained_generate_results")
