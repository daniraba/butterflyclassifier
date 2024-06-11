import torch 
import torch.nn as nn

class INeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(7500,2000),
            nn.Sigmoid(),
            nn.Linear(2000,5),
            nn.Sigmoid()
        )

        self.loss_function=nn.MSELoss()
        self.optimiser=torch.optim.SGD(self.parameters(),lr=0.1)

    def forward(self,inputs):
        inputs=torch.FloatTensor(inputs)
        return self.model(inputs)
    
    def train(self,inputs,targets):
        targets=torch.FloatTensor(targets)
        outputs=self.forward(inputs)

        loss=self.loss_function(outputs,targets)#Error
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


        
    