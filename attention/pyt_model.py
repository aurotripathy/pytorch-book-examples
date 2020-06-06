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
        ## TF # self.repeator = RepeatVector(Tx)  # Repeats the input Tx times.
        # self.repeator = torch.repeat(Tx)
        
        ## TF # self.concatenator = Concatenate(axis=-1)
        # self.concatenator = torch.cat(axis=-1)  # not needed
        
        ## TF # self.densor1 = Dense(10, activation = "tanh", name='Dense1')
        self.densor1 = nn.Linear(128, 10)  # TBD input?? 

        ## TF # self.densor2 = Dense(1, activation = "relu", name='Dense2')
        self.densor2 = nn.Linear(10, 1)
        
        # We are using a custom softmax(axis = 1) loaded in this notebook
        ## TF # self.activator = Activation(softmax, name='attention_weights')
        self.activator = SoftMax()


        ## TF # self.dotor = Dot(axes = 1)
        # self.dotor = torch.mm(axes = 1)

        self.bi_d_lstm = nn.LSTM(self.human_vocab_size, self.n_a, 1,  # in size, out size, num layers
                                 batch_first=False, bidirectional=True)
        self.post_activation_LSTM_cell = nn.LSTM(100, self.n_state)  # FIX input
        self.output_layer = nn.Linear(100, machine_vocab_size)  # Fix

        self.state = torch.zeros(self.batch_size, self.n_state,)
        self.context = torch.zeros(self.n_state,)

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

        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        # s_prev = self.repeator(s_prev)
        set_trace()
        print('s_prev shape', s_prev.shape)
        s_prev = s_prev.repeat(self.Tx, 1,1)
        print('s_prev shape', s_prev.shape)
        
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        concat = torch.cat((a, s_prev), dim=-1)
        print('concat shape', concat.shape)

        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = F.tanh(self.densor1(concat))
        print('e shape', e.shape)

        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = F.relu(self.densor2(e))
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = self.activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        # context = self.dotor([alphas, a])
        context = torch.mm(alphas, a)


        return context


    def forward(self, x):
        """
        Arguments:
        X -- input
        Returns:
        model -- Pytorch model instance
        """
        set_trace()
        # Initialize empty list of outputs
        outputs = []

        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
        ## TF # a = Bidirectional(LSTM(self.n_a, return_sequences=True, name='bidirectional_1'), merge_mode='concat')(X)
        # Bi-LSTM input is of shape (seq_len, batch, input_size) 30, 100, 37
        a, _ = self.bi_d_lstm(x)

        # Step 2: Iterate for Ty steps
        for t in range(self.Ty):

            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self._one_step_attention(a, self.state)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = self.post_activation_LSTM_cell(context, initial_state = [s, c])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = F.softmax(self.output_layer(s))

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model(inputs=(X, s0, c0), outputs=outputs)

        return model
