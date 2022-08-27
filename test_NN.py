import torch
import torch.optim 


batch, dim_in, dim_h, dim_out = 128, 2000, 200, 20 

input_X = torch.randn(batch, dim_in)

output_Y = torch.randn(batch, dim_out)

Adam_model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
) # neural network

loss_fn = torch.nn.MSELoss(reduction='sum') # loss function

rate_learning = 1e-4 # learning rate

optim = torch.optim.Adam(Adam_model.parameters(), lr=rate_learning)

for iter_it in range(500):

    pred_y = Adam_model(input_X)

    loss = loss_fn(pred_y, output_Y)

    if iter_it % 5 == 0:
      print(iter_it, loss.item())

    optim.zero_grad()

    loss.backward()

    step = optim.step()