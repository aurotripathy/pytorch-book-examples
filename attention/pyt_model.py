import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_softmax import SoftMax
from pudb import set_trace

# following https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412

class Attn(torch.nn.Module):
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, batch_size):
        super(Attn, self).__init__()
        """
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"
        """
        self.Ty = Ty
        self.Tx = Tx
        self.n_a = n_a
        self.n_state = n_s
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size
        self.batch_size = batch_size

        self.densor1 = nn.Linear(128, 10)  

        self.densor2 = nn.Linear(10, 1)
        

        self.bi_d_lstm = nn.LSTM(self.human_vocab_size, self.n_a, 1,  # in size, out size, num layers
                                 batch_first=False, bidirectional=True)


        self.post_activation_LSTM_cell = nn.LSTMCell(self.n_state, self.n_state)  # input size, hidden size

        self.output_layer = nn.Linear(self.n_state, machine_vocab_size)  # Fix

        self.h_state = torch.zeros(self.batch_size, self.n_state, requires_grad=True)
        self.c_state = torch.zeros(self.batch_size, self.n_state, requires_grad=True)

        
    def _one_step_attention(self, a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """

        print('s_prev shape', s_prev.shape)
        s_prev = s_prev.repeat(self.Tx, 1,1)
        print('s_prev shape', s_prev.shape)
        
        concat = torch.cat((a, s_prev), dim=-1)
        print('concat shape', concat.shape)

        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = F.tanh(self.densor1(concat))
        print('e shape', e.shape)

        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = F.relu(self.densor2(e))

        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = F.softmax(energies)
        print('alphas shape', alphas.shape)
        print('a shape', a.size())

        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context =  torch.matmul(alphas.transpose(1,0).transpose(2,1), a.transpose(1,0))

        return context


    def forward(self, x):
        """
        Arguments:
        x -- input
        Returns:
        model -- Pytorch model instance
        """
        # Initialize empty list of outputs
        outputs = []

        # Step 1: Define your pre-attention Bi-LSTM. 
        a, _ = self.bi_d_lstm(x)  # output shape (seq_len, batch, num_directions * hidden_size) 30, 100, 32*2

        # Step 2: Iterate for Ty steps
        for _ in range(self.Ty):

            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
            attn_context = self._one_step_attention(a, self.c_state)
            print('attn_context shape', attn_context.shape)
            set_trace()
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            (self.h_state, self.c_state) = self.post_activation_LSTM_cell(attn_context.view(self.batch_size, self.n_state),
                                                                          (self.h_state, self.c_state))

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = F.softmax(self.output_layer(self.h_state))
            print('output_layer shape', out.size())

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
            print('outputs length', len(outputs), 'outputs element type', outputs[0].type())

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        # model = Model(inputs=(X, s0, c0), outputs=outputs)

        return torch.stack(outputs).resize(10, 100, 11)  # TODO remove hard coding
