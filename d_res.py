import torch 
from torch import nn 
from setings import D_Settings
from comune import D_Commune
from compute import D_Compute

class D_Residual(nn.Module):
    """_summary_:
    - This class compute the residual block of the time-GPT
      model for timeseries forecasting. The architecture of this
      block resembles the GPT model. We introduce the regularization
      before the attention and FF-net layer to create a clean path
      for gradient flows.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, 
                 setings: D_Settings = D_Settings()) -> None:
        super().__init__()
        self.commune = D_Commune(setings = setings)
        self.ln_1 = nn.LayerNorm(setings.num_embed)
        self.compute = D_Compute(setings = setings)
        self.ln_2 = nn.LayerNorm(setings.num_embed)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_: input tensor(BATCH_SIZE, block_size, num_embed)

        Returns:
            torch.Tensor: _description_: output tensor(BATCH_SIZE, block_size, num_embed)
        """
        x = x + self.commune(self.ln_1(x)) # Normalize before the attention
        x = x + self.compute(self.ln_2(x)) # Normalize before the ff-net
        out = x # shape ==> (BATCH_SIZE, block_size, num_embed)
        return out

if __name__ == "__main__":
    
    B, T, C = 32, D_Settings.block_size, D_Settings.num_embed
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    x = torch.randn(size = (B, T, C), device = device)
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> Available device: {device}\n")
    print(f"\n{160 * '*'}\n")
    residual = D_Residual().to(device = device)
    out = residual(x)
    assert out.shape == (B, T, C), print(f"\n>>>> Shapes missmatch!\n")
    print(f"\n>>>> Okay!\n")
    print(f"\n{160 * '*'}\n")
