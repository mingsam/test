
import torch

dtype = torch.float
device = torch.device("cuda:0")

N, D_in, H1,H2, D_out,learning_rate = 64, 1000,100, 100, 10, 1e-6

x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)

w1=torch.randn(D_in, H1, device=device, dtype=dtype, requires_grad=True)
w2=torch.randn(H1, H2, device=device, dtype=dtype, requires_grad=True)
w3=torch.randn(H2, D_out, device=device, dtype=dtype, requires_grad=True)


#forwords
for i in range(5000):
    h_pred = x.mm(w1).clamp(min=0).mm(w2)
    y_pred = h_pred.clamp(min=0).mm(w3)

    loss = (y_pred-y).pow(2).sum()
    print(i,loss.item())

#backwords
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        w3 -= learning_rate*w3.grad
##clean the grad
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()