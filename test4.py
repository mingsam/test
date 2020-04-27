
import torch
import torch.nn as nn

dtype = torch.float
device = torch.device("cuda:0")
N, D_in, H1, H2, D_out,learning_rate = 64, 1000,100, 100, 10, 1e-4

x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)


class SampleReLUNet(nn.Module):
    def __int__(self, D_in, H1, H2, D_out):
        super(SampleReLUNet, self).__init__()
        self.linear1=nn.Linear(D_in, H1)
        self.linear2=nn.Linear(H1, H2)
        self.linear3=nn.Linear(H2, D_out)

    def forward(self, x):
        h1_pred = self.linear1(x).clamp(min=0)
        h2_pred = self.linear2(h1_pred).clamp(min=0)
        y_pred = self.linear3(h2_pred)
        return y_pred

model = SampleReLUNet(D_in, H1, H2, D_out).to(device)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

for i in range(2500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(i,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

