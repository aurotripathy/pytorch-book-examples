import torch
import torch.nn as nn
import torch.optim as optim
from nmt_utils import load_dataset, preprocess_data
import torch.nn.functional as F
import numpy as np
import random
import argparse
from pudb import set_trace

seq_len_human = 30  # human time-steps 30
seq_len_machine = 10  # machine time-steps 10
EMBEDDING_DIM_PRE_ATTN = 50
HIDDEN_DIM_PRE_ATTN_LSTM = 32  # hidden size of pre-attention Bi-LSTM; output is twice of this
HIDDEN_DIM_POST_ATTN_LSTM = 64
LEARNING_RATE = 0.01
NB_EPOCHS = 4


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
        # lstm_out holds the backward and forward hidden states in the final layer
        # lstm_out dim, [sent len, batch size, hid dim * n directions]

        return lstm_out


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn_weighted = nn.Linear((HIDDEN_DIM_PRE_ATTN_LSTM * 2 +  # 64
                                        HIDDEN_DIM_POST_ATTN_LSTM) * # 64
                                       seq_len_human, seq_len_human)  # 64 x 2 x 30
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)  # TODO convert to LSTM
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, time_step):
        cat_input_hidden = torch.stack([torch.cat((input[i], hidden[0]), 1) for i in range(30)])
        attn_weights = F.softmax(self.attn_weighted(cat_input_hidden.view(1, 1, -1)), dim=1)
        attn_applied = torch.bmm(attn_weights, input.view(1, 30, -1))

        set_trace()
        output, hidden = self.gru(attn_applied, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden  # TODO n three parameters?
 
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    

def evaluate(input_tensor, encoder_rnn, attn_decoder_rnn, target_length=seq_len_machine):

    with torch.no_grad():
        encoder_outputs = encoder_rnn(input_tensor)
        attn_decoder_hidden = attn_decoder_rnn.init_hidden()

        decoded_date = []
        for time_step in range(target_length):
            attn_decoder_output, attn_decoder_hidden = attn_decoder_rnn(encoder_outputs, attn_decoder_hidden, time_step)
            topv, topi = attn_decoder_output.data.topk(1)
            decoded_date.append(inv_machine_vocab[topi.item()])

    return ''.join(decoded_date)


def train(input_tensor, target_tensor,
          encoder_rnn, attn_decoder_rnn,
          encoder_optimizer, attn_decoder_optimizer,
          criterion, target_length=seq_len_machine):

    encoder_optimizer.zero_grad()
    attn_decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs = encoder_rnn(input_tensor)
    attn_decoder_hidden = attn_decoder_rnn.init_hidden()

    for time_step in range(target_length):
        attn_decoder_output, attn_decoder_hidden = attn_decoder_rnn(encoder_outputs, attn_decoder_hidden, time_step)

        loss += criterion(attn_decoder_output, target_tensor[time_step].unsqueeze(0))

    loss.backward()

    encoder_optimizer.step()
    attn_decoder_optimizer.step()

    return loss.item() / target_length

parser = argparse.ArgumentParser(description='Either train or evaluate attn model for normalizing dates')
parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'eval'],
                    help="pick mode - train or eval")
args = parser.parse_args()

# We'll train the model on a dataset of 10000 human readable dates
# and their equivalent, standardized, machine readable dates.
nb_samples = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(nb_samples)
print('Human vocab', human_vocab)
print('Machine vocab', machine_vocab)
print('Inverse machine vocab', inv_machine_vocab)

X, Y = zip(*dataset)
X, Y, _, _ = preprocess_data(dataset, human_vocab, machine_vocab, seq_len_human, seq_len_machine)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_rnn = EncoderRNN(EMBEDDING_DIM_PRE_ATTN, HIDDEN_DIM_PRE_ATTN_LSTM,
                         len(human_vocab)).to(device)
attn_decoder_rnn = AttnDecoderRNN(HIDDEN_DIM_POST_ATTN_LSTM, len(machine_vocab)).to(device)

X = torch.from_numpy(X).long().to(device)
Y = torch.from_numpy(Y).long().to(device)
#  TODO split train/test

if args.mode == 'train':
    encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(attn_decoder_rnn.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    total_loss = 0
    for iters in range(NB_EPOCHS):
        for i in range(1, nb_samples):
            total_loss += train(X[i - 1], Y[i - 1], encoder_rnn, attn_decoder_rnn, encoder_optimizer, decoder_optimizer, criterion)
            if i % 1000 == 0:
                print(i, total_loss/1000)
                total_loss = 0
        torch.save(encoder_rnn.state_dict(), 'encoder_rnn_state.dict')
        torch.save(attn_decoder_rnn.state_dict(), 'attn_decoder_rnn_state.dict')
else:  # evaluate
    print('loading models...')
    encoder_rnn.load_state_dict(torch.load('encoder_rnn_state.dict'))
    attn_decoder_rnn.load_state_dict(torch.load('attn_decoder_rnn_state.dict'))
    
    for _ in range(100):
        i = random.choice(range(nb_samples))
        machine_date = evaluate(X[i], encoder_rnn, attn_decoder_rnn)
        print('Input Human Date:', dataset[i][0])
        print('Predicted Machine Date:', machine_date,
              'Actual Machine Date:', dataset[i][1],
              'matches' if machine_date == dataset[i][1] else 'mismatch')
