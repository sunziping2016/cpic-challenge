from typing import List, Dict


class Metric(object):
    def __init__(self):
        super(Metric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        """
        params:
        - golden: dictionary contains samples, each sample is a key-value indexed by ID
        - predictions: dictionary contains samples, each sample is a key-value indexed by ID
        """
        raise NotImplementedError


class ClassificationMetric(Metric):
    """
    use Accuracy/Micro-F1 as classification metric
    """
    def __init__(self):
        super(ClassificationMetric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        correctness = 0.
        total = len(golden.keys())
        for golden_k, golden_v in golden.items():
            correctness += 1. if golden_v == predictions[golden_k] else 0.
        return correctness/total


class SummarizationMetric(Metric):
    """
    use Rouge-L as summarization metric
    """
    def __init__(self):
        super(SummarizationMetric, self).__init__()
        from rouge import Rouge
        self.rouge = Rouge()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        labels = []
        preds = []
        for k in golden.keys():
            labels.append(" ".join(list(golden[k])))
            preds.append(" ".join(list(predictions[k])))
        rouge_scores = self.rouge.get_scores(preds, labels, avg=True)
        return rouge_scores["rouge-l"]["f"]


class MRCMetric(Metric):
    """
    use string F1 score as MRC metric
    """
    def __init__(self):
        super(MRCMetric, self).__init__()

    def tokenize_chinese_chars(self, text):
        """
        :param text: input text, unicode string
        :return:
            tokenized text, list
        """

        def _is_chinese_char(cp):
            """Checks whether CP is the codepoint of a CJK character."""
            # This defines a "chinese character" as anything in the CJK Unicode block:
            #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            #
            # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            # despite its name. The modern Korean Hangul alphabet is a different block,
            # as is Japanese Hiragana and Katakana. Those alphabets are used to write
            # space-separated words, so they are not treated specially and handled
            # like the all of the other languages.
            if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                    (cp >= 0x3400 and cp <= 0x4DBF) or  #
                    (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                    (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                    (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                    (cp >= 0x2B820 and cp <= 0x2CEAF) or
                    (cp >= 0xF900 and cp <= 0xFAFF) or  #
                    (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
                return True

            return False

        output = []
        buff = ""
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or char == "=":
                if buff != "":
                    output.append(buff)
                    buff = ""
                output.append(char)
            else:
                buff += char

        if buff != "":
            output.append(buff)

        return output

    def normalize(self, in_str):
        """
        normalize the input unicode string
        """
        in_str = in_str.lower()
        sp_char = [
            u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
            u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
            u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
        ]
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)

    def find_lcs(self, s1, s2):
        """find the longest common subsequence between s1 ans s2"""
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        max_len = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > max_len:
                        max_len = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - max_len:p], max_len

    def _calc_f1_em(self, answer, prediction, is_impossible):
        if is_impossible:
            if prediction.lower() == "":
                return 1.0, 1.0
            else:
                return 0.0, 0.0
        ans_norm = self.normalize(answer)
        pred_norm = self.normalize(prediction)
        ans_tok = self.tokenize_chinese_chars(ans_norm)
        pred_tok = self.tokenize_chinese_chars(pred_norm)
        lcs, lcs_len = self.find_lcs(ans_tok, pred_tok)
        if lcs_len == 0:
            return 0., 0.
        prec = 1.0*lcs_len/len(pred_tok)
        rec = 1.0*lcs_len/len(ans_tok)
        f1 = (2*prec*rec)/(prec+rec)
        em = 1. if ans_norm == pred_norm else 0.
        return f1, em

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        f1 = 0.
        em = 0.
        for k in golden.keys():
            pred_span = predictions[k]
            answer_span = golden[k]
            is_impossible = True if answer_span == "" else False
            _f1, _em = self._calc_f1_em(answer_span, pred_span, is_impossible)
            f1 += _f1
            em += _em
        return f1/len(golden.keys())


class NERMetric(Metric):
    def __init__(self):
        super(NERMetric, self).__init__()

    def _compute(self, label, pred, hit):
        recall = 0 if label == 0 else (hit / label)
        precision = 0 if pred == 0 else (hit / pred)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        f1 = 0.
        for k in golden.keys():
            hit_entities = [e for e in predictions[k] if e in golden[k]]
            _recall, _precision, _f1 = self._compute(
                len(golden[k]),
                len(predictions[k]),
                len(hit_entities)
            )
            f1 += _f1
        return f1/len(golden.keys())


datatype2metrics = {
    "classification": ClassificationMetric,
    "mrc": MRCMetric,
    "ner": NERMetric,
    "summ": SummarizationMetric,
    "keys": MRCMetric,
}
