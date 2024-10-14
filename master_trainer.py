from timecore import TimeGPT
from agens import LoadAgens
import numpy as np
import torch 
import matplotlib.pyplot as plt 
from timeit import default_timer as timer 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

if torch.cuda.is_available():
    device = torch.device("cuda")
    BATCH_SIZE = 16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    BATCH_SIZE = 8
else:
    device = torch.device("cpu")
    BATCH_SIZE = 4
loader = LoadAgens(B = BATCH_SIZE)

def set_timer(t: float = timer()) -> float:
    h = int(t / (60 * 60))
    m = int(t % (60 * 60) / 60)
    s = int(t % 60)
    return f"hrs: {h:02}, mins: {m:>02}, secs: {s:>05.2f}"

def master(model: TimeGPT = TimeGPT(),
           EPOCHS: int = 100000)->TimeGPT:
    """
    Args:
        model (TimeGPT, optional): _description_. Defaults to TimeGPT().
        EPOCHS (int, optional): _description_. Defaults to 100000.

    Returns:
        TimeGPT: _description_. Trained TimeGPT
    """
    model = model.to(device = device)
    model.train()
    
    for epoch in range(EPOCHS):
        
        xb, yb = loader.next_batch(split = "train")
        # Convert data back to np.ndarray for standerdization
        xb, yb = xb.detach().cpu().numpy(), yb.detach().cpu().numpy()
        xb, yb = scaler.fit_transform(xb), scaler.fit_transform(yb)
        # Re-convert to torch.Tensor
        xb = torch.from_numpy(xb).to(torch.float32)
        yb = torch.from_numpy(yb).to(torch.float32)
        # Ship to the available device
        xb, yb = xb.to(device = device), yb.to(device = device)
        
        xv, yv = loader.next_batch(split = "valid")
        # Convert data back to np.ndarray for standerdization
        xv, yv = xv.detach().cpu().numpy(), yv.detach().cpu().numpy()
        xv, yv = scaler.fit_transform(xv), scaler.fit_transform(yv)
        # Re-convert to torch.Tensor
        xv = torch.from_numpy(xv).to(torch.float32)
        yv = torch.from_numpy(yv).to(torch.float32)
        # Ship to the available device
        xv, yv = xv.to(device = device), yv.to(device = device)
        # Forward-pass
        loss, _ = model(x = xb, y = yb)
        if epoch % 1 == 0:
            # run evaluation loop
            vloss, _ = model.time_eval(x = xv, y = yv)
            print(f"\n{160 * '*'}\n")
            print(f"\n>>>> Epoch: {epoch}, Train Loss: {loss.item():.4f}, Validation Loss: {vloss.item():.4f}\n")
        
        lr = model.cosine_decay_lr(epoch) # Fetch the lr
        optimizer = model.optimizer_setup()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr 
        # zerograding
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # optimization 
        optimizer.step()
    return model 

def forecastings(model: TimeGPT) -> None:
    for i in range(3):
        if i % 3 == 0:
            # Use the last datapoints in 2018 for the initial sequence
            xf, yf = loader.next_batch(split = "valid")
   
    # Convert data back to np.ndarray for standerdization
    xf, yf = xf.detach().cpu().numpy(), yf.detach().cpu().numpy()
    xf, yf = scaler.fit_transform(xf), scaler.fit_transform(yf)
    # Re-convert to torch.Tensor
    xf = torch.from_numpy(xf).to(torch.float32)
    yf = torch.from_numpy(yf).to(torch.float32)
    # Ship to the available device
    xf, yf = xf.to(device = device), yf.to(device = device)
    # Get the forecasts
    fcs = model.time_forecasting(start_seq = xf)
    # Get the predictions
    _, preds = model.time_eval(x = xf, y = yf)
    # Rescale to original format 
    preds = preds.view(BATCH_SIZE, -1)
    yf, preds = yf.detach().cpu().numpy(), preds.detach().cpu().numpy()
    yf, preds = scaler.inverse_transform(yf), scaler.inverse_transform(preds)
    fcs = fcs.detach().cpu().numpy()
    fcs = scaler.inverse_transform(fcs) # shape ==> (BATCH_SIZE, horizon)
    plt.figure(figsize = (12, 4))
    fc_steps = np.arange(yf.shape[1], yf.shape[1] + fcs.shape[1])
    print(f">>>> yf shape: {yf.shape}, preds shape {preds.shape}, fcs shape: {fcs.shape}")
    plt.plot(np.arange(yf.shape[1]), yf[-1, :], color = "gray", label = "real")
    plt.plot(np.arange(yf.shape[1]), preds[-1,:], color = "fuchsia", label = "predictions")
    plt.plot(fc_steps, fcs[-1,:], color = "green", label = "forecasts")
    plt.title("TimeGPT evaluation and forecasts plot")
    plt.legend(loc = "best")
    plt.show()
    
if __name__ == "__main__":
    tic = timer()
    model = master()
    #forecastings(model = model)
    toc = timer()
    set_timer(toc - tic)    