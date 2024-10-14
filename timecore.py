from torch import nn 
from d_res import D_Residual
from setings import D_Settings
from typing import Tuple, List, Dict 
import torch, math
from agens import LoadAgens
loader = LoadAgens()
if torch.cuda.is_available():
    device = torch.device("cuda")
    BATCH_SIZE = 16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    BATCH_SIZE = 8
else:
    device = torch.device("cpu")
    BATCH_SIZE = 4

class TimeGPT(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
        - This class implement the GPT model for timeseries forecasting
          Training, evaluation, forecasting loops are embeded inside the
          model. Parameters initialization and optimization settings follows
          GPT2 and GPT3 to some extent. The forecast horizon is turned for
          a maximum of 192 time-stamps but can be extended depends on the 
          need.
    """
    def __init__(self, 
                 setings: D_Settings = D_Settings()) -> None:
        super().__init__()
        self.setings = setings 
        self.timeblock = nn.ModuleDict(
            dict(
                stamps = nn.Linear(setings.block_size, setings.num_embed),
                positions = nn.Embedding(setings.block_size, setings.num_embed),
                blocks = nn.ModuleList(D_Residual(
                    setings = setings) for _ in range(setings.num_layers)),
                ln_n = nn.LayerNorm(setings.num_embed)))
        self.final = nn.Linear(setings.num_embed, setings.input_dim, bias = False)
        
        self.apply(self.pars_init)
    
    def pars_init(self, module) -> None:
        """_summary_

        Args:
            module (_type_): _description_
            Initializes the parameters following GPT 2
            - All Linear layer matrices are initilaized as N(0, 0.02)
            - The weights for the embedding are initialized as N(0, 0.01)
            - All biases are initialized to zeros
            - We scale-down the variance for the residuals by the square root
              of the number of residual layers
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "TIME_GPT"):
                std *= (2 * self.setings.num_layers) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean = 0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.01)
    
    def forward(self, 
                x: torch.Tensor,
                y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_: raw input tensor:  shape (BATCH_SIZE, block_size)
            y (torch.Tensor): _description_: raw target tensor: shape (BATCH_SIZE, block_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_:
            (loss(sclar tensor), predictions(BATCH_SIZE, block_size))
        """
        _, T = x.size()
        assert T <= self.setings.block_size, print(f"\n>>>> Cannot forward more than the block size: {self.setings.block_size}\n")
        # Forwarding for the time-stamps
        tstamps = self.timeblock.stamps(x)# (B, T) @ (T, C) ==> (B, C)
        # Recover the block dimension
        tstamps = tstamps.unsqueeze_(dim = 1).repeat(1, T, 1)# shape ==> (B, T, C)
        # Initialize the position's tensor
        p = torch.arange(start = 0, end = T, dtype = torch.long, device = x.device)
        # Encoding the positions: [p.unsqueeze_(1).repeat(1, T)] @ positions
        pe = self.timeblock.positions(p) # (T, T)@(T, C) ==> (T, C)
        # adding time-stamps and the positions' encoding
        bl_in = tstamps + pe # shape ==> (B, T, C) + (T, C) ==> (B, T, C)
        # Forward for the residuals block
        
        for res in self.timeblock.blocks:
            r_out = res(bl_in) # shape ==> (B, T, C)
        # forward for the last LayerNorm layer
        time_out = self.timeblock.ln_n(r_out) # shape ==> (B, T, C)
        
        # Forward for the final linear layer to get the predictions
        preds = self.final(time_out) # shape ==> (B, T, input_dim = 1)
        
        # Computing the loss:
        loss = None
        if y is not None:
            B, T, C = preds.size()
            # Strech both tensors into a long vector
            preds, y = preds.view(B * T * C), y.view(B * T)
            loss = torch.nn.functional.mse_loss(preds, y) # scalar tensor
            
        return (loss, preds)
    
    def cosine_decay_lr(self, 
                        it: int,
                        max_lr: float = 6e-4,
                        warmup: int = 100,
                        max_iters:int = 1000
                        )-> float:
        """_summary_

        Args:
            it (int): _description_. Current iteration
            max_lr (float, optional): _description_. Defaults to 6e-4.
            warmup (int, optional): _description_. Defaults to 100.
            max_iters (int, optional): _description_. Defaults to 1000.

        Returns:
            float: _description_
        """
        min_lr = 0.1 * max_lr # set the minimum lr
        if it < warmup:
            # Increase the lr gradually up to the maximum value
            # at the biginning training stage
            lr = max_lr * (it + 1) / warmup
        # if we exceed the maximum iteration, keep trains with the least lr
        elif it > max_iters:
            lr = min_lr
        # implement the cosine decay (from the maximum lr)
        else:
            decay_rate = (it - warmup) / (max_iters - warmup)
            cosine_decay_rate = 0.5 * (1 + math.cos(math.pi * decay_rate))
            lr = min_lr + cosine_decay_rate * (max_lr - min_lr)
        return lr
    
    def optimizer_setup(self,
                        decay_rate: float = 0.1,
                        lr: float = 6e-4,
                        ) -> torch.optim:
        """
           __summary__. This method implement configuration of the torch.optim
            - We decay all high dimension tensors (>= 2)
            - Lower dimensional tensors such as biases are left unchanged
           Params:
           ----------------------------
           decay_rate: Optional, Default: float = 0.1
           lr: float: The lr at the current iteration
           
           Returns:
           -----------------------------
           torch.optim: A configured optimizer (with groups of parameters to decay and not to decay, and lr) 
        """
        # fetch all parameters with their respective names  
        pars: Dict[str, torch.Tensor] = {name : par for name, par in self.named_parameters()}
        # keep only the trainable parameters
        pars = {name: par for name, par in pars.items() if par.requires_grad} 
        
        par_to_decay: List[torch.Tensor] = [par for _, par in pars.items() if par.ndim >= 2]
        par_not_decay: List[torch.Tensor] = [par for _, par in pars.items() if par.ndim < 2]
        # print(f"\n{160 * '*'}\n")
        # print(f"\n>>>> There are {sum(p.numel() for p in par_to_decay):,} parameters to decay\n")
        # print(f"\n{160 * '*'}\n")
        # print(f"\n>>>> There are {sum(p.numel() for p in par_not_decay):,} parameters not to decay\n")
        # print(f"\n{160 * '*'}\n")
        # print(f"\n>>>> The {self.__class__.__name__} has the total of {sum(p.numel() for p in par_to_decay) + sum(p.numel() for p in par_not_decay):,} trainable parameters\n")
        # print(f"\n{160 * '*'}\n") 
        
        # Organize parameters' dictionaries 
        optim_groups: List[Dict] = [
            {"params": par_to_decay, "weight_decay": decay_rate},
            {"params": par_not_decay, "weight_decay": 0.0}
        ] 
        # configure the optimizer using the above setings 
        optimizer = torch.optim.AdamW(
            params = optim_groups, 
            lr = lr, 
            betas = (0.90, 0.95), eps = 1e-8) 
        return optimizer   
            
    @torch.no_grad()
    def time_eval(self, x: torch.Tensor,
                  y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_: Raw validation input tensor(BATCH_SIZE, block_size) 
            y (torch.Tensor): _description_: Raw validation target tensor(BATCH_SIZE, block_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_:
            val_loss(scalar tensor), preds(BATCH_SIZE, block_size)
        """
        # switch-off regularization layers like BatchNorm, DropOut etc
        self.eval()
        v_loss, preds = self(x = x, y = y)
        return (v_loss, preds)
    
    @torch.no_grad()
    def time_forecasting(self, 
                         start_seq: torch.Tensor,
                         horizon: int = 96 * 2,
                         max_seq: int = 10)-> torch.Tensor:
        """
          __summary__
          - This method generate the forecasts from the pre-trained TIME-GPT model
          ---------------
          params:
          start_seq: torch.Tensor(BATCH_SIZE, D): D can be of any-length starting 
                     from a single time-stamps to several time-stamps, but must not
                     execeed the transformer's block size
          horizon: This is the maximum time-frame tgenerate the forecast. If more 
                   than the block size, approximation by truncation will be done
          max_seq: This is the maximum number of sequences to generate for smoothing
                   the forecasts (depending on the computing resources)
        """
        self.eval()
        fcs: List[torch.Tensor] = [] # Container for all forcasts
        for _ in range(max_seq): 
            while start_seq.shape[1]< start_seq.shape[1] + horizon: # Generate until the horizon is reached
                start_seq_clip = start_seq[:,
                        -self.setings.block_size:] # clip to avoid confict in the GPT forward pass
                _, preds = self(x = start_seq_clip) # shape ==> (B, T, 1)
                # Fetch the forecasted value
                fc = preds[:, -1, :] # shape ==> (B, 1)
                # Append the generated forecasts to continue the initial sequence
                start_seq = torch.cat((start_seq, fc), dim = 1) # shape ==> (B, start_seq.shape[1]+)
            fcs.append(start_seq) # packing sequences (forecasts)
        forecasts = torch.stack(fcs, dim = 0) # stacking the sequences vertically
        out = forecasts.mean(dim = 0)
        return out # shape ==> (B, horizon)

if __name__ == "__main__":
    
    loader = LoadAgens(B = BATCH_SIZE)
    xb, yb = loader.next_batch(split = "train")
    xb, yb = xb.to(device = device), yb.to(device = device)
    model = TimeGPT().to(device = device)
    loss, preds = model(x = xb, y = yb)
    _ = model.optimizer_setup()
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> Loss: {loss.item():.4f}\n")
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> predictions shape: {preds.shape}\n")
    print(f"\n{160 * '*'}\n")
    fcs = model.time_forecasting(start_seq = xb)
    print(f"\n>>>> The first 10 forecasts: \n {fcs[0, :10]}\n")
    print(f"\n{160 * '*'}\n")
    
    
                