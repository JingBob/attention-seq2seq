import torch
import torch.nn as nn
from torch import optim
from datapre import prepareData
import random
from util import timeSince, showPlot
from seq2seq_model import EncoderRNN, AttnDecoderRNN
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 需要每个单词的唯一索引用作以后网络的输入和目标。
SOS_token = 0
EOS_token = 1
# 规定句子的最大长度
MAX_LENGTH = 10

'''
训练:
为了训练，对于每一对，我们需要一个输入张量（输入句子中单词的索引）和目标张量（目标句子中单词的索引）。
在创建这些向量时，我们会将 EOS 令牌附加到两个序列。
'''

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


# 获取句子中每个单词的索引，返回的是索引序列
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# 根据索引序列建立张量
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# 输入张量是输入句子中单词的索引，输出张量是目标句子中单词的索引
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5

'''
开始训练:
为了训练，我们通过编码器运行输入句子，并跟踪每个输出和最新的隐藏状态。
然后解码器被赋予<SOS>令牌作为它的第一个输入，编码器的最后一个隐藏状态作为它的第一个隐藏状态。
'''
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # 获取编码器的每个输出和隐藏状态，用于计算注意力权重
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    # 解码器第一个隐藏状态是编码器输出的隐藏状态
    decoder_hidden = encoder_hidden

    # 训练可以使用“Teacher forcing”策略：使用真实目标输出作为下一个输入，而不是使用解码器的猜测作为下一个输入。
    # 使用Teacher forcing会使模型收敛更快，但使用训练得到的网络时，可能会表现出不稳定。
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 将目标单词作为下一个解码输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # 用预测结果作为下一个解码输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            # 遇到终止符号就退出解码
            if decoder_input.item() == EOS_token:
                break

    # 反向传播
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


'''
整个训练过程：
1.启动计时器
2.初始化优化器和标准
3.创建一组训练对
4.开始绘制损失数组
'''
'''
@函数名：迭代训练
@参数说明：
    encoder：编码器
    decoder：解码器
    n_iters：训练迭代次数
    print_every：多少代输出一次训练信息
    plot_every：多少代绘制一下图
    learning_rate：学习率
'''
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # 优化器用SGD
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    # 因为模型的输出已经进行了log和softmax，因此这里损失韩式只用NLL，三者结合起来就算二元交叉熵损失
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch:%d  %s (%d%%) loss:%.4f' % (iter, timeSince(start, iter / n_iters),
                                          iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# 评估
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    # 训练吧
    hidden_size = 256  # 隐藏层维度设置为256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 75000,5000
    trainIters(encoder1, attn_decoder1, 5000, print_every=100)
    evaluateRandomly(encoder1, attn_decoder1)
    # 注意力可视化
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())

    # 输入一些句子测试下
    evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    evaluateAndShowAttention("elle est trop petit .")
    evaluateAndShowAttention("je ne crains pas de mourir .")
    evaluateAndShowAttention("c est un jeune directeur plein de talent .")