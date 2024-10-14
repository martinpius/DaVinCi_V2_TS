from torch import nn 
import torch 
from setings import D_Settings

class D_Commune(nn.Module):
    
    def __init__(self, 
                 setings: D_Settings = D_Settings()) -> None:
        super().__init__()
        assert setings.num_embed % setings.num_heads == 0, print(f"\n>>>> The embed size: {setings.num_embed} is not divisible by the number of heads: {setings.num_heads}\n")
        self.setings = setings 
        # Initializes the weights for the K, Q, and V tensors
        self.c_attn = nn.Linear(self.setings.num_embed,
                                3 * self.setings.num_embed)
        # Initializes the weights for processing the attention info
        self.c_proj = nn.Linear(self.setings.num_embed, 
                                self.setings.num_embed)
        # For scaling down the variance of the residuals at initialization
        self.c_proj.TIME_GPT = 1
        self.register_buffer("bias", 
                             torch.tril(
                             torch.ones(self.setings.block_size, 
                             self.setings.block_size)).view(
                             1, 1, self.setings.block_size, 
                             self.setings.block_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_: input tensor(BATCH_SIZE, block_size, num_embed)

        Returns:
            torch.Tensor: _description_ : output (attention weights)(BATCH_SIZE, block_size, num_embed)
        """
        B, T, C = x.size()
        assert T <= self.setings.block_size, print(f"\n>>>> Cannot forward more than the block size: {self.setings.block_size}\n")
        # Initialize the K,Q,V tensors in parallel
        kqv = self.c_attn(x)# (B, T, C) @ (C, 3C) ==> (B, T, 3C)
        # Spliting equally along the 3rd dimension
        K, Q, V = kqv.split(split_size = C, dim = 2)# Each with shape: (B, T, C)
        # Reset K, Q, and V into 4d for kernel-fusion
        K = K.view(B,
                   T, 
                   self.setings.num_heads,
                   C//self.setings.num_heads).transpose(1,2)# shape ==> (B, nh, T, hs)
        V = V.view(B,
                   T, 
                   self.setings.num_heads,
                   C//self.setings.num_heads).transpose(1,2)# shape ==> K.shape
        Q = Q.view(B, 
                   T, 
                   self.setings.num_heads,
                   C//self.setings.num_heads).transpose(1,2)# shape ==> K.shape
        # Compute the affinities between query and key tensors
        # (B, nh, T, hs) @ (B, nh, hs, T)
        aff = Q @ K.transpose(-2, -1) * (K.shape[3])**(-0.5) # shape ==> (B, nh, T, T)
        # Apply the auto-regressive mask
        aff = aff.masked_fill(self.bias[:, :, :T, :T]==0, float("-inf")) # shape (B, nh, T, T)
        # Normalize the affinities to have the prob-distribution
        wei = torch.nn.functional.softmax(aff, dim = 3) # shape ==> aff.shape
        # Get the attention weights
        attn = wei @ V # shape: (B, nh, T, T) @ (B, nh, T, hs) ==> (B, nh, T, hs)
        # Reshape back to the original shape
        attn = attn.transpose(1,2).contiguous().view(B, T, C) # (B, T, C)
        # Processing the pooled information
        out = self.c_proj(attn) # shape ==> (B, T, C)
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
    commune = D_Commune().to(device = device)
    out = commune(x)
    assert out.shape == (B, T, C), print(f"\n>>>> Shape missmatch!\n")
    print(f"\n>>>> Okay!\n")
    print(f"\n{160 * '*'}\n")