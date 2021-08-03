import torch
import torch.nn as nn
from datapre import MAX_LENGTH
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
编码器:
seq2seq 网络的编码器是一个 RNN，它为输入句子中的每个单词输出一些值。
对于每个输入词，编码器输出一个向量和一个隐藏状态，并将隐藏状态用于下一个输入词。
'''
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  # 维度调整为1*1*n
        output = embedded
        output, hidden = self.gru(output, hidden)  # 获取每个GRU的输出和隐藏状态，用于后续计算attention
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

e = EncoderRNN(10, 256)
print(e)
'''
解码器:
解码器是另一个RNN，它采用编码器输出向量并输出一系列单词来创建翻译。
'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


'''
注意力解码器
'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # 全连接层
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # 先把输入embedding
        embedded = self.embedding(input).view(1, 1, -1)
        # dropout防止过拟合
        embedded = self.dropout(embedded)
        # 计算注意力权重
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 矩阵相乘，用注意力权重乘以编码输出
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 将输入的embedding层和注意力层拼接，按维数1拼接（横着拼）
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # 拼好后加个全连接层然后压缩维度0。
        output = self.attn_combine(output).unsqueeze(0)
        # 激活函数
        output = F.relu(output)
        # 输入GRU
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

d = AttnDecoderRNN(256, 10)
print(d)