from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 需要每个单词的唯一索引用作以后网络的输入和目标。
SOS_token = 0
EOS_token = 1


# 用一个名为的辅助类Lang
# 包含 word → index ( word2index) 和 index → word ( index2word) 字典，以及每个单词的计数word2count，
# 稍后将用于替换稀有单词。
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 这些文件都是 Unicode 格式的，为了简化，我们将把 Unicode 字符转换为 ASCII，使所有内容都小写，并修剪大部分标点符号。
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 统一小写、修剪和删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


'''
为了读取数据文件，我们将文件分割成行，然后将行分割成对。
文件都是英语→其他语言，所以如果我们想从其他语言→英语翻译，我添加了reverse 标志来反转对。
'''
def readLangs(language1, language2, reverse=False):
    print("Reading lines...")

    # 分行
    lines = open('data/%s-%s.txt' % (language1, language2), encoding='utf-8'). \
        read().strip().split('\n')

    # 将每行分割成语言对，并正则化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 反转语言对
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(language2)
        output_lang = Lang(language1)
    else:
        input_lang = Lang(language1)
        output_lang = Lang(language2)

    return input_lang, output_lang, pairs


'''
由于有很多例句，我们想快速训练一些东西，我们将把数据集修剪成只相对较短和简单的句子。
此处的最大长度为 10 个单词（包括结尾标点符号），只考虑以“我是”或“他是”等形式开头的句子。
'''
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


'''
准备数据的完整过程是：
1.读取文本文件并拆分成行，将行拆分成对
2.标准化文本，按长度和内容过滤
3.从成对的句子制作单词列表
'''
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
