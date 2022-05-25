import os
import json
import time
import codecs
import random
import logging
import unicodedata
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger()


def draw_data(data_list, signal_list, target_file, data_names):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    data_lines = []
    for data, signal, data_name in zip(data_list, signal_list, data_names):
        data_line = plt.plot(data, signal, label=data_name)
        data_lines.append(data_line)
    plt.legend(data_lines, data_names)
    plt.savefig(target_file)
    plt.close()
    print("{}'s figure saved into {}".format(data_names, target_file))


def makedirs(name):
    import errno
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            pass
        else:
            raise


def pretty_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{:d}:{:2d}:{:2d}".format(int(h), int(m), int(s))


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


def write2json(data_list, data_path, data_name):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data_list, ensure_ascii=False, indent=2))
        print("{}({}) saved into {}".format(data_name, len(data_list), data_path))


def load_json_by_line(file):
    data = []
    with open(file, "r", encoding="utf8") as f:
        reader = f.readlines()
        for line in reader:
            # print(line)
            sample = json.loads(line.strip())
            data.append(sample)
    return data


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print("load {} from {}".format(len(data), json_file))
    return data


def process_escape(text):
    try:
        decoded_string = codecs.escape_decode(bytes(text, "utf-8"))[0].decode("utf-8")
    except Exception as e:
        print(e)
        print(text)
        # input()
        decoded_string = text
    return decoded_string


def is_bad_char(char):
    cat = unicodedata.category(char)
    if cat.startswith("C") or cat.startswith("Zs"):
        return True
    return False


def process_text(text: str):
    text = text.strip().lower()
    text = process_escape(text)
    return text
    processed_text = ""
    for i in range(len(text)):
        if is_bad_char(text[i]):
            # print("{} has {} at position {}-ORD is {}".format(text, text[i], i, ord(text[i])))
            continue
        else:
            processed_text += text[i]
    return processed_text

