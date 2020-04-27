
import torch
dtype = torch.float
device = torch.device("cuda:0")

N, D_in, H1, H2, D_out,learning_rate = 64, 1000,100, 100, 10, 1e-6

x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)

model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2,D_out),
)

loss_fn=torch.nn.MSELoss(reduction='sum')

for i in range(2500):
    model.to(device)
    y_pred = model(x)

    loss=loss_fn(y_pred, y)
    print(i,loss.item())

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad