import torch
import torch.nn as nn
from nmt_utils import load_dataset, preprocess_data
# from pudb import set_trace
import numpy as np

Tx = 30  # human time-steps 30
Ty = 10  # machine time-steps 10

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

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
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
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
Xoh = np.expand_dims(Xoh, axis=2)  # shape (10000, 30, 1, 37)

EMBEDDING_DIM = 50
HIDDEN_DIM = 32  # hidden size of pre-attention Bi-LSTM; output is twice of this


def train(input_tensor, target_tensor,
          encoder_rnn, attn_decoder,
          encoder_optimizer, attn_decoder_optimizer,
          criterion, max_length=Ty):

    encoder_rnn = EncoderRNN(EMBEDDING_DIM, HIDDEN_DIM, len(human_vocab))

 
    encoder_rnn.zero_grad()
    lstm_out = encoder_rnn(input)
    encoder_hidden = encoder_rnn.initHidden()

    encoder_optimizer.zero_grad()
    attn_decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    encoder_outputs = encoder_rnn(input_tensor)
    attn_decoder_input = torch.tensor([[SOS_token]])

    attn_decoder_input = encoder_outputs


    for d_indx in range(target_length):
        attn_decoder_output, attn_decoder_hidden, decoder_attention = attn_decoder(
            attn_decoder_input, attn_decoder_hidden, encoder_outputs)
        topv, topi = attn_decoder_output.topk(1)
        attn_decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(attn_decoder_output, target_tensor[d_indx])

    loss.backward()

    encoder_optimizer.step()
    attn_decoder_optimizer.step()

    return loss.item() / target_length


input = torch.from_numpy(X[0]).long()
target = torch.from_numpy(Y[0]).long()
train(input, target)