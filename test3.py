
import torch
import torch.nn as nn

dtype = torch.float
device = torch.device("cuda:0")
N, D_in, H1, H2, D_out,learning_rate = 64, 1000,100, 100, 10, 1e-4

x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)

model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.ReLU(),
    nn.Linear(H1,H2),
    nn.ReLU(),
    nn.Linear(H2,D_out),
).to(device)

loss_fn=nn.MSELoss(reduction='sum')
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(2500):
    y_pred = model(x)

    loss = loss_fn(y_pred,y)
    print(i, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()