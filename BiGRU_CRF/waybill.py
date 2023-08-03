import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
# 预训练的词向量
from paddlenlp.embeddings import TokenEmbedding


# 1 数据加载
for i, line in enumerate(open('data/train.txt', encoding='utf-8')):
    if 0 < i < 5:
        print('%d: ' % i, line.split()[0])
        print('   ', line.split()[1])


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


train_ds, dev_ds, test_ds = load_dataset(datafiles=('data/train.txt', 'data/dev.txt', 'data/test.txt'))


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab


label_vocab = load_dict('data/tag.dic')
word_vocab = load_dict('data/word.dic')


#2 数据转id
def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids


def convert_example(example):
    tokens, labels = example
    # OOV 表示空
    token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')

    label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
    return token_ids, len(token_ids), label_ids


train_ds.map(convert_example)
dev_ds.map(convert_example)
test_ds.map(convert_example)


# 3
batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV'), dtype='int'),  # token_ids
        Stack(dtype='int'),  # seq_len 二维矩阵
        Pad(axis=0, pad_val=label_vocab.get('O'), dtype='int')  # label_ids
    ): fn(samples)


train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=True, # 分批后最后一组非32，丢弃
        return_list=True,
        collate_fn=batchify_fn)

dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=32,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=32,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

#4 设计网络模型构建，BiGRU双向RNN，CRF输出

class BiGRUWithCRF(nn.Layer):
    # 超参数初始化模型
    def __init__(self,
                 emb_size,
                 hidden_size,
                 word_num, # 多少个词
                 label_num, # 多少个标签
                 use_w2v_emb=False):
        super(BiGRUWithCRF, self).__init__()

        #1 词嵌入层
        if use_w2v_emb:
            # 如果使用预训练的词向量，已经大规模语料训练过，则编码更好
            self.word_emb = TokenEmbedding(extended_vocab_path='data/word.dic', unknown_token='OOV')

        else:
            # 词嵌入数量，嵌入维度（把每个词编码成多长维度的向量）
            self.word_emb = nn.Embedding(word_num, emb_size)

        # 2 gru层
        self.gru = nn.GRU(emb_size,
                          hidden_size,  # 隐藏节点
                          num_layers=2, #2个隐层
                          direction='bidirectional' # 双向
                          )
        # 3 全连接层 两个方向
        self.fc = nn.Linear(hidden_size * 2, label_num + 2)  # BOS EOS ？begin of sentence , end of sentence

        # 4 crf 输出层， 前提需要知道有多少个标签
        self.crf = LinearChainCrf(label_num)
        # 最终结果解码
        self.decoder = ViterbiDecoder(self.crf.transitions)

    # 把前面那些层都连起来
    def forward(self, x, lens):
        embs = self.word_emb(x)
        # 序列标注，
        output, _ = self.gru(embs)
        output = self.fc(output)

        # 把output输出解码
        _, pred = self.decoder(output, lens)
        return output, lens, pred

#5 构建模型
# Define the model netword and its loss
network = BiGRUWithCRF(300, 300, len(word_vocab), len(label_vocab))
model = paddle.Model(network)

#准备优化器
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
# 最小化的损失
crf_loss = LinearChainCrfLoss(network.crf)
# 评估器
chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
model.prepare(optimizer, crf_loss, chunk_evaluator) # 设置模型


model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=1,
              save_dir='./results',
              log_freq=1)

model.evaluate(eval_data=test_loader, log_freq=1)


def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


outputs, lens, decodes = model.predict(test_data=test_loader)
preds = parse_decodes(test_ds, decodes, lens, label_vocab)

print('\n'.join(preds[:5]))


