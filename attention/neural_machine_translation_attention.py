import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import load_dataset, preprocess_data, string_to_int, to_categorical
from pyt_model import Attn
import torch.optim as optim
import torch
import torch.nn as nn
from pudb import set_trace


# We'll train the model on a dataset of 10000 human readable dates
# and their equivalent, standardized, machine readable dates. 
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

# X: a processed version of the human readable dates in the training set,
# where each character is replaced by an index mapped to the character via human_vocab.
# Each date is further padded to $T_x$ values with a special character (< pad >). X.shape = (m, Tx)

# Y: a processed version of the machine readable dates in the training set,
# where each character is replaced by the index it is mapped to in machine_vocab.
# You should have Y.shape = (m, Ty).

# Xoh: one-hot version of X, the "1" entry's index is mapped to the character thanks
# to human_vocab. Xoh.shape = (m, Tx, len(human_vocab))

# Yoh: one-hot version of Y, the "1" entry's index is mapped to the character thanks to machine_vocab.
# Yoh.shape = (m, Tx, len(machine_vocab)).
# Here, len(machine_vocab) = 11 since there are 11 characters ('-' as well as 0-9).

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
print("Xoh.shape after preprocessing (one-hot):", Xoh.shape)
print("Yoh.shape after preprocessing (one-hot):", Yoh.shape)
print("Human vocab", human_vocab)
print("Machine vocab", machine_vocab)
print("Human vocab", human_vocab)
print("Machine vocab", machine_vocab)
print("Length Human vocab", len(human_vocab))
print("Length Machine vocab", len(machine_vocab))


Xoh = torch.from_numpy(Xoh).float()
Yoh = torch.from_numpy(Yoh).float()

Xoh = Xoh.transpose(0,1)  # seq, batch, feature
Yoh = np.transpose(Yoh, (1, 0, 2))
print("Tensor Xoh.shape:", Xoh.size())
print("Numpy  Yoh.shape:", Yoh.size())

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 50
batch_size = 100
n_batches = Xoh.size()[1] // batch_size

n_a = 32  # hidden state size of the Bi-LSTM
n_s = 64  # hidden state size of the post-attention LSTM
model = Attn(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab), batch_size)

outputs = model(Xoh[:, 0:batch_size, :])  # Dim = 30 x 100 x 37

optimizer = optim.Adam(model.parameters(),
                       lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, epochs + 1):
    # TODO remember to shuffle data
    model.train()  # Turn on the train mode
    for i in range(n_batches):
        local_Xoh, local_Yoh = Xoh[:, i*n_batches:(i+1)*n_batches,], Yoh[:, i*n_batches:(i+1)*n_batches,]
        # local_Xoh, local_Yoh = local_Xoh.to(device), local_Yoh.to(device)
        optimizer.zero_grad()
        outputs = model(local_Xoh).transpose(1,0).transpose(2,1)
        print('shape of outputs', outputs.size())
        # TODO - https://discuss.pytorch.org/t/loss-functions-for-batches/20488
        targets = local_Yoh.argmax(2).transpose(1,0)
        print('numpy shape of targets', targets.shape)
        loss = criterion(outputs, targets)
        print('---loss---', loss)
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'models/attn-model.pt')

# Inference
model.load_weights('models/attn-model.h5')

EXAMPLES = ['Friday 3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.expand_dims(np.transpose(source), axis=0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))


