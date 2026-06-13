train.py, FileNotFoundError: [Errno 2] No such file or directory: 'config.json', the config.json was deleted and doesnt exist, created config.json with the necessary contents, 71b36a4c5de39986e3f38bc662f5c2152a7dfa93

train.py -> data.py, FileNotFoundError: [Errno 2] No such file or directory: 'data\\cells_data.pt', the files in the backup on the cloud do not have _data appended to their names, removed the "_data" addition from line 11, 2dfb7c011c152037b31403d4e2415d929f8a9c30

train.py -> fit.py -> models.py, RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x3072 and 2048x1024), the output of the convolutional layers is 192x4x4=3072 in size which does not match the fully connected layer at 2048 in dimension, changed the linear layer to be of size 3072x1024 to fit to the output of the convolutions in line 92, c36b71164d35a2b49ce09408a68da7b8aeb49475

train.py -> fit.py, RuntimeError: 0D or 1D target tensor expected, multi-target not supported, when the cross_entropy_loss receives the labels it received an unnessecary dimension of (,1) which does not fit the outputs that do not have this dimension, used the torch.squeeze method with labels.squeeze() in line 24, be0da2eb25c9cd572584822595b238d1067acc83

train.py -> fit.py, RuntimeError: 0D or 1D target tensor expected, multi-target not supported, when the cross_entropy_loss receives the labels it received an unnessecary dimension of (,1) which does not fit the outputs that do not have this dimension, used the torch.squeeze method with labels.squeeze() in line 46, 