from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
from custom_softmax import softmax

class Attn(object):
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        self.Ty = Ty
        self.Tx = Tx
        self.n_a = n_a
        self.n_s = n_s
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size
        self.repeator = RepeatVector(Tx)  # Repeats the input Tx times.
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(10, activation = "tanh", name='Dense1')
        self.densor2 = Dense(1, activation = "relu", name='Dense2')
        
        # We are using a custom softmax(axis = 1) loaded in this notebook
        self.activator = Activation(softmax, name='attention_weights') 
        self.dotor = Dot(axes = 1)

        self.post_activation_LSTM_cell = LSTM(n_s, return_state = True)
        self.output_layer = Dense(machine_vocab_size, activation=softmax)


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
        s_prev = self.repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        concat = self.concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = self.densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = self.densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = self.activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = self.dotor([alphas, a])

        return context


    def model(self):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """

        # Define the inputs of your model with a shape (Tx,)
        # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
        X = Input(shape=(self.Tx, self.human_vocab_size), name= 'X')
        s0 = Input(shape=(self.n_s,), name='s0')
        c0 = Input(shape=(self.n_s,), name='c0')
        s = s0
        c = c0

        # Initialize empty list of outputs
        outputs = []

        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
        a = Bidirectional(LSTM(self.n_a, return_sequences=True, name='bidirectional_1'), merge_mode='concat')(X)

        # Step 2: Iterate for Ty steps
        for t in range(self.Ty):

            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self._one_step_attention(a, s)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = self.post_activation_LSTM_cell(context, initial_state = [s, c])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = self.output_layer(s)

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model(inputs=(X, s0, c0), outputs=outputs)

        return model
