import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SubsNN(T.nn.Module):
    def __init__(self):
        """Regression network
        Input is 11 as the number of X train rows 
        These are columns 0 to 11 from the output csv files
        from `processData.m` octave file
        """
        super(SubsNN, self).__init__()
        self.hid1 = T.nn.Linear(11, 16)  # 11-(16-16)-1
        self.hid2 = T.nn.Linear(16, 16)
        self.oupt = T.nn.Linear(16, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = self.oupt(z)  # could use sigmoid() here
        return z


class SubsNN2(T.nn.Module):
    def __init__(self):
        """Regression network
        Input is 11 as the number of X train rows
        These are columns 0 to 11 from the output csv files
        from `processData.m` octave file
        """
        super(SubsNN2, self).__init__()
        self.hid1 = T.nn.Linear(7, 14)
        self.hid2 = T.nn.Linear(14, 14)
        self.out = T.nn.Linear(14, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.out.weight)
        T.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = self.out(z)  # could use sigmoid() here
        return z


class FFNN(nn.Module):
    
    def __init__(self, input_size, num_hidden_layers, hidden_size, out_size=1):
        super(FFNN, self).__init__()
        print(input_size, num_hidden_layers, hidden_size)

        # Create input
        self.input = nn.Linear(input_size, hidden_size)
        
        # Create hidden layers
        self.hidden = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            hid = nn.Linear(hidden_size, hidden_size)

            # zero everything
            nn.init.xavier_uniform_(hid.weight)
            nn.init.zeros_(hid.bias)

            self.hidden.append(hid)
        
        # Create output
        self.output = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # Flatten image
        
        # Utilize hidden layers and apply activation function
        z = self.input(x)
        
        for layer in self.hidden:
            z = T.tanh(layer(z))
            # z = F.relu(z)
        
        # Get predictions
        z = self.output(z)
        return z

    def print(self):
        h1 = self.hidden[0]
        return f"FFNN:input_{self.input.in_features}x{self.input.out_features}"\
            f":hidden({len(self.hidden)})_{h1.in_features}x{h1.out_features}"\
            f":out_{self.output.in_features}x{self.output.out_features}"
