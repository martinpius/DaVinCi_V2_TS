import torch 
from torch import nn 
from setings import D_Settings

class D_Compute(nn.Module):
    
    def __init__(self, 
                 setings: D_Settings = D_Settings())-> None:
        super().__init__()
        self.c_fc = nn.Linear(setings.num_embed, 4 * setings.num_embed)
        self.act = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * setings.num_embed, setings.num_embed)
        # set the flag to scale down the variance of the residuals at initialization
        self.c_proj.TIME_GPT = 1
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_: Input tenso(BATCH_SIZE, block_size, num_embed)

        Returns:
            torch.Tensor: _description_: torch.Tensor(BATCH_SIZE, block_size, num_embed)
        """
        out = self.c_proj(self.act(self.c_fc(x))) # shape ==> (BATCH_SIZE, block_size, num_embed)
        return out

if __name__ == "__main__":
    B, T, C = 32, D_Settings.block_size, D_Settings.num_embed
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")
    
    x = torch.randn(size = (B, T, C), device = device)
    dcompute = D_Compute().to(device = device)
    out = dcompute(x)
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> Available device: {device}\n")
    print(f"\n{160 * '*'}\n")
    assert out.shape == (B, T, C), print(f"\n>>>> Shape missmatch!\n")
    print(f"\n>>>> Okay!\n")
    print(f"\n{160 * '*'}\n")
        