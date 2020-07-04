import torch
import torch.nn as nn
import torch.optim as optim
from nmt_utils import load_dataset, preprocess_data
import torch.nn.functional as F
from pudb import set_trace
import numpy as np

Tx = 30  # human time-steps 30
Ty = 10  # machine time-steps 10
EMBEDDING_DIM_PRE_ATTN = 50
HIDDEN_DIM_PRE_ATTN_LSTM = 32  # hidden size of pre-attention Bi-LSTM; output is twice of this
HIDDEN_DIM_POST_ATTN_LSTM = 64
LEARNING_RATE = 0.1


class EncoderRNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.bi_dir_lstm = nn.LSTM(embedding_dim, hidden_dim,
                                   bidirectional=True)


    def forward(self, sentence):
        embeds = self.char_embeddings(sentence)
        lstm_out, _ = self.bi_dir_lstm(embeds.view(len(sentence), 1, -1))
        #lstm_out holds the backward and forward hidden states in the final layer
        #lstm_out dim, [sent len, batch size, hid dim * n directions]

        return lstm_out


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,
                 max_length=Ty):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn = nn.Linear(3840, 30)
        # self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        repeat_hidden = torch.repeat_interleave(hidden, repeats=30, dim=0).view(1, 1, -1)
        attn_weights = F.softmax(self.attn(torch.cat((input.view(1, 1, -1)[0],
                                                      repeat_hidden[0]), 1)),
                                 dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), input.view(1, 30, -1))

        output, hidden = self.gru(attn_applied, hidden)

        set_trace()
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden
 
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    
# We'll train the model on a dataset of 10000 human readable dates
# and their equivalent, standardized, machine readable dates.
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print('Human vocab', human_vocab)
print('Machine vocab', machine_vocab)
X, Y = zip(*dataset)

# Xoh[0] shape - 30 time-steps, 37 long
# Yoh[0] shape - 10 time-steps, 11 long
X, Y, _, _ = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)


def train(input_tensor, target_tensor,
          encoder_rnn, attn_decoder_rnn,
          encoder_optimizer, attn_decoder_optimizer,
          criterion, target_length=Ty):

    encoder_optimizer.zero_grad()
    attn_decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs = encoder_rnn(input_tensor)
    attn_decoder_input = encoder_outputs
    attn_decoder_hidden = attn_decoder_rnn.init_hidden()

    for d_indx in range(target_length):
        attn_decoder_output, attn_decoder_hidden = attn_decoder_rnn(attn_decoder_input, attn_decoder_hidden)

        loss += criterion(attn_decoder_output, target_tensor[d_indx].unsqueeze(0))

    loss.backward()

    encoder_optimizer.step()
    attn_decoder_optimizer.step()

    return loss.item() / target_length


encoder_rnn = EncoderRNN(EMBEDDING_DIM_PRE_ATTN, HIDDEN_DIM_PRE_ATTN_LSTM, len(human_vocab))
attn_decoder_rnn = AttnDecoderRNN(HIDDEN_DIM_POST_ATTN_LSTM, len(machine_vocab))

encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(attn_decoder_rnn.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

for i in range(10000):
    print(i)
    input = torch.from_numpy(X[i]).long()
    target = torch.from_numpy(Y[i]).long()
    loss = train(input, target, encoder_rnn, attn_decoder_rnn, encoder_optimizer, decoder_optimizer, criterion)
    print(loss)
