from .instruction_prompts import *


dataset2instruction = {
    "senti": {
        "prompt": "【{}】这篇文章的情感态度是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "cls": {
        "prompt": "【{}】这篇文章的类别是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "app": {
        "prompt": "【{}】这篇文章的类别是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "news": {
        "prompt": "【{}】这篇文章的类别是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "intent": {
        "prompt": "【{}】这句话的意图是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "nli": {
        "prompt": "【{}】和【{}】，以上两句话的逻辑关系是什么？{}",
        "keys_order": ["text_a", "text_b", "verbalizer"],
        "instruction": NLIInstruction,
        "data_type": "classification",
    },
    "sts": {
        "prompt": "【{}】和【{}】，以上两句话的内容是否相似？{}",
        "keys_order": ["text_a", "text_b", "verbalizer"],
        "instruction": STSInstruction,
        "data_type": "classification",
    },
    "para": {
        "prompt": "【{}】和【{}】，以上两句话的内容是否相似？{}",
        "keys_order": ["text_a", "text_b", "verbalizer"],
        "instruction": PARAInstruction,
        "data_type": "classification",
    },
    "mrc": {
        "prompt": "阅读文章【{}】问题【{}】的答案是什么？",
        "keys_order": ["context", "question"],
        "instruction": MRCInstruction,
        "data_type": "mrc",
    },
    "ner": {
        "prompt": "找出【{}】这篇文章中所有【{}】类型的实体？",
        "keys_order": ["context", "entity_type"],
        "instruction": NERInstruction,
        "data_type": "ner",
    },
    "summ": {
        "prompt": "【{}】这篇文章的摘要是什么？",
        "keys_order": ["passage"],
        "instruction": SUMMInstruction,
        "data_type": "summ",
    },
    "keys": {
        "prompt": "【{}】这篇文章的关键词是什么？",
        "keys_order": ["text_a"],
        "instruction": KEYSInstruction,
        "data_type": "keys",
    },
    "wsc": {
        "prompt": "文章【{}】中【{}】的是【{}】吗？{}",
        "keys_order": ["text", "target/span2_text", "target/span1_text", "verbalizer"],
        "instruction": WSCInstruction,
        "data_type": "classification",
    },
    "yesno": {
        "prompt": "阅读文章【{}】问题【{}】？{}",
        "keys_order": ["text_b", "text_a", "verbalizer"],
        "instruction": MRCInstruction,
        "data_type": "classification",
    },
    "c3": {
        "prompt": "阅读文章【{}】问题【{}】{}",
        "keys_order": ["context", "question", "choice"],
        "instruction": C3Instruction,
        "data_type": "classification",
    },
    "weibo_emotion": {
        "prompt": "【{}】这篇文章的情感态度是什么？{}",
        "keys_order": ["text_a", "verbalizer"],
        "instruction": WeiboEmotionInstruction,
        "data_type": "classification",
    },
    "lsht": {
        "prompt": "【{}】这篇文章的类别是什么？{}",
        "keys_order": ["content", "verbalizer"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    }
}


def instruction_format(data_dict: Dict) -> List[Dict]:
    special_datasets = {
        "dureader_yesno": "yesno",
        "c3": "c3",
        "NLPCC2014_Weibo_Emotion_classification": "weibo_emotion",
        "NLPCC2014_LSHT_sample": "lsht"
    }
    instruction_data = []
    for data_type, type_dict in data_dict.items():
        for data_name, data_info in type_dict.items():
            label_mappings = data_info.get("label_mappings")
            data_list = data_info["data_list"]
            format_info = dataset2instruction[special_datasets.get(data_name, data_type)]
            instruction_processor = format_info["instruction"](
                data_list,
                label_mappings,
                format_info["prompt"],
                format_info["keys_order"],
                format_info["data_type"]
            )
            instruction_data.extend(instruction_processor.transform2instruction())
            print(instruction_data[-1])

    return instruction_data
