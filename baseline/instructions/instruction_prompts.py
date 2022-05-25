from typing import List, Dict


class Instruction(object):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        self.data_list = data_list
        self.verbalizer = verbalizer
        self.instruction = instruction
        self.keys_order = keys_order
        self.data_type = data_type

    def transform2instruction(self):
        raise NotImplementedError


class NERInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(NERInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = "，".join(example["entities"]) if "entities" in example and type(example["entities"]) is list else ""
            example["entity_type"] = self.verbalizer[example["entity_type"]]
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class MRCInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str, NO_ANSWER="无答案"):
        super(MRCInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)
        self.NO_ANSWER = NO_ANSWER
        self.YESNO_SET = {"yes", "no", "depends"}

    def process_answer(self, answer_text):
        if len(answer_text) == 0:
            return self.NO_ANSWER
        answer_text = answer_text[0] if type(answer_text) is list else answer_text
        if answer_text == "":
            return self.NO_ANSWER
        if answer_text.lower() in self.YESNO_SET:
            return self.verbalizer.get(answer_text.lower(), answer_text.lower())
        return answer_text

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            if len(self.verbalizer) == 3:  # yesno
                example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
                example["target"] = self.process_answer([example.get("label", "")])
            else:
                example["target"] = self.process_answer(example.get("answer", ""))
            slots = [example[k] for k in self.keys_order]
            example["instruction"] = self.instruction.format(*slots)
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class C3Instruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str, NO_ANSWER="无答案"):
        super(C3Instruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)
        self.NO_ANSWER = NO_ANSWER

    def process_answer(self, answer_text):
        if answer_text == "":
            return self.NO_ANSWER
        return answer_text

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.process_answer(example["answer"][0])
            example["choice"] = "/".join(example["choice"])
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class SUMMInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(SUMMInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = example["summary"]
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class KEYSInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(KEYSInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = "，".join(example["keys"])
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class NLIInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(NLIInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer[example["label"]] if "label" in example and example["label"] != "" else ""
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class STSInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(STSInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer[str(example["label"])] if "label" in example and example["label"] != "" else ""
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class PARAInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(PARAInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            label = example["label"]
            example["target"] = self.verbalizer[str(label)]
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class ClassificationInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(ClassificationInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer[str(example["label"])] if "label" in example and example["label"] != "" else ""
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class WSCInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(WSCInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            slots = [
                example["text"],
                example["target"]["span2_text"],
                example["target"]["span1_text"],
                example["verbalizer"],
            ]
            example["target"] = self.verbalizer[example["label"]]
            example["instruction"] = self.instruction.format(*slots)
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class WeiboEmotionInstruction(Instruction):
    def __init__(self, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(WeiboEmotionInstruction, self).__init__(data_list, verbalizer, instruction, keys_order, data_type)
        self.verbalizer = self.verbalizer["label_list_1"]

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer[example["label_1"]]
            example["verbalizer"] = "/".join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples
