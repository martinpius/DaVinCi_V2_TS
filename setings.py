from dataclasses import dataclass 

@dataclass
class D_Settings:
    
    input_dim: int = 1
    block_size: int = 1024
    num_layers: int = 12 
    num_embed: int = 768
    num_heads: int = 12 
    dropout: float = 0.5

    
