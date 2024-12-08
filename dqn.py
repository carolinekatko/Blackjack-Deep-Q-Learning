# Caroline Katko, Transy U
# Original code: https://github.com/johnnycode8/dqn_pytorch
# Modified by: Caroline Katko using Chat GPT and Codeium AI
# Source code altered (number of nodes and hidden layers) depending on test


#imports
import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim1=143, hidden_dim2=130, hidden_dim3=117, hidden_dim4=104, hidden_dim5=91, hidden_dim6=78, hidden_dim7=65, hidden_dim8=52, hidden_dim9=39, hidden_dim10=26, hidden_dim11=13):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.fc6 = nn.Linear(hidden_dim5, hidden_dim6)
        self.fc7 = nn.Linear(hidden_dim6, hidden_dim7)
        self.fc8 = nn.Linear(hidden_dim7, hidden_dim8)
        self.fc9 = nn.Linear(hidden_dim8, hidden_dim9)
        self.fc10 = nn.Linear(hidden_dim9, hidden_dim10)
        self.fc11 = nn.Linear(hidden_dim10, hidden_dim11)
        self.output = nn.Linear(hidden_dim11, action_dim)
     

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        return self.output(x)


if __name__ == '__main__':
    state_dim = 3
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)