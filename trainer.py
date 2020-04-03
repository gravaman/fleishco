import time
import math
import torch
import torch.nn as nn
import torchtext
from TransformerModel import TransformerModel
from torchtext.data.utils import get_tokenizer


def train(model, text, chunk_size=35, log_interval=200):
    # turn on train mode and setup
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(text.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0)-1, chunk_size)):
        data, targets = get_batch(train_data, i, chunk_size=chunk_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d}.batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // chunk_size,
                      scheduler.get_lr()[0], elapsed*1000/log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source, chunk_size=35):
    eval_model.eval()  # turn on eval mode
    total_loss = 0.0
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0)-1, chunk_size):
            data, targets = get_batch(data_source, i, chunk_size=chunk_size)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source)-1)


def batchify(text, data, bsz):
    """
    params:
    text: used to numericalize the data
    data: contains text to numericalize
    bsz: batch size to split data into
    """
    # divide dataset into bsz parts and trim extra elements
    data = text.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data[:nbatch*bsz]
    # divide into batch size and make contiguous in C order memory
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(data_source, i, chunk_size=35):
    """
        Subdivides given data_sounce into input and target sequences
        for the transformer model. Input is the initial sequence
        and target is the subsequent sequence of chunk_size length.
    """
    seq_len = min(chunk_size, len(data_source)-1-i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len].view(-1)
    return data, target


# device setup
if torch.cuda.is_available():
    print('GPU enabled')
    device_type = 'cuda'
else:
    device_type = 'cpu'

device = torch.device(device_type)


# get text data
TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)

# data setup
batch_size = 20
eval_batch_size = 10
train_data = batchify(TEXT, train_txt, batch_size)
val_data = batchify(TEXT, val_txt, eval_batch_size)
test_data = batchify(TEXT, test_txt, eval_batch_size)

# model setup
ntokens = len(TEXT.vocab.stoi)  # vocabulary size
emsize = 200  # embedding dimension
nhid = 200  # dimension of TransformerEncoder feedforward network
nlayers = 2  # number of TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in multiheadattention model
dropout = 0.2  # dropout value
model = TransformerModel(ntokens, emsize, nhead,
                         nhid, nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# train model
epochs = 3
chunk_size = 35
best_val_loss = float('inf')
best_model = None
for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    train(model, TEXT, chunk_size=chunk_size)
    val_loss = evaluate(model, val_data, chunk_size)
    epoch_time = time.time()-epoch_start_time
    val_ppl = math.exp(val_loss)
    print('-'*89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, epoch_time, val_loss, val_ppl))
    print('-'*89)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    scheduler.step()

# eval model
test_loss = evaluate(best_model, test_data, chunk_size=chunk_size)
test_ppl = math.exp(test_loss)
print('='*89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, test_ppl))
print('='*89)
