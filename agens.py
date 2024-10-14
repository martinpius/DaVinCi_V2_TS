import pandas as pd 
import matplotlib.pyplot as plt 
from typing import Tuple
import torch 
from setings import D_Settings
web_url = "https://raw.githubusercontent.com/martinpius/LargeLaN/main/dfm_all.csv"

class LoadAgens:
    
    def __init__(self,
                 B: int = 4,
                 setings: D_Settings = D_Settings()):
        
        self.B = B 
        self.setings = setings 
        # Download and preprocess the data
        df = pd.read_csv(web_url)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace = True) # indexing the dataframe by the date column
        # set the time stamps to 15 minutes interval, averaging the data points
        # at each interval, and fill the missing values to obtain the full sequence
        df = df.resample(pd.Timedelta(15, "m"), closed = "right").mean().interpolate()
        self.data = df
        self.initial_pt = 0 
        
    def plotting_baseline(self) -> None:
        
        plt.figure(figsize = (12, 4))
        # Plot the baseline model predictions using 2018 data
        plt.plot(self.data.loc["2018-01-01 00:00:00":]['prediction'],
                 color = "gray", label = "prediction")
        plt.plot(self.data.loc["2018-01-01 00:00:00":]['measurement'],
                 color = "fuchsia", label = "real")
        plt.title("Performance Evaluation: Baseline model")
        plt.xlabel("2018 data")
        plt.ylabel("Electricty consuption in (KWhrs)")
        plt.legend(loc = "best")
        plt.show()
    
    def next_batch(self,
                   split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            split (str, optional): _description_. Defaults to "train".
            - This method build a simple custom pytorch generator to
              stream data in batches during training and valiation of TIME_GPT 
              model
            - First we split the data into training-validation set
            - We generate sequences of size (BATCH_SIZE * block_size)
            - We draw data sequentially without replacement.
            - If we complete one epoch we cycle back
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_. (X(B, T), y(B, T))
        """
        df = self.data
        if split == "train":
            # load up to July, 2018 for training
            data = df.loc[:"2018-07-31 23:45:00"]['measurement']
        elif split == "valid":
            # Take the rest for validation
            data = df.loc["2018-08-01 00:00:00":]['measurement']
        else:
            print(f"\n{160 * '*'}\n")
            print(f"\n>>>> Invalid split: {split}\n")
            print(f"\n{160 * '*'}\n")
            
        # print(f"\n{160 * '*'}\n")
        # print(f"\n>>>> The {split} split has the total of {len(data) // (self.B * self.setings.block_size)} batches")
        # print(f"\n{160 * '*'}\n")
        
        # convert to torch.Tensor
        data = torch.from_numpy(data.values).to(torch.float32)
        
        X_batch, Y_batch = [], []
        
        for _ in range(self.B):
            # If we have insuficient data for a complete block, we reset 
            if len(data) < (self.initial_pt + self.setings.block_size + 1):
                self.initial_pt = 0
            # Extract the input and target sequences
            x = data[self.initial_pt: self.initial_pt + self.setings.block_size]
            y = data[self.initial_pt + 1: self.initial_pt + 1 + self.setings.block_size]
            # We reset again if the data-points are not sufficient for a block_size
            if x.shape[0] < self.setings.block_size and y.shape[0] < self.setings.shape[0]:
                self.initial_pt = 0
                # Re-Extract input & target sequences
                x = data[self.initial_pt: self.initial_pt + self.setings.block_size]
                y = data[self.initial_pt + 1 : self.initial_pt + 1 + self.setings.block_size]
            
            # Packing the sequences
            X_batch.append(x)
            Y_batch.append(y)
            
            # sliding with a constant window for the next batch
            self.initial_pt += self.B * self.setings.block_size
        # Stacking the sequences vertically to obtain a tensor(self.B, self.setings.block_size)
        X = torch.stack(X_batch, dim = 0) # shape ==> (self.B, self.setings.block_size)
        Y = torch.stack(Y_batch, dim = 0)# shape ==> X.shape
        return (X, Y)
                
        

if __name__ == "__main__":
    loader = LoadAgens()
    loader.plotting_baseline()
    xtr_b, ytr_b = loader.next_batch(split = "train")
    xval_b, yval_b = loader.next_batch(split = "valid")
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> X_train shape: {xtr_b.shape}, Y_train shape: {ytr_b.shape}\n")
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> Xval shape: {xval_b.shape}, Yval shape: {yval_b.shape}\n")
    print(f"\n{160 * '*'}\n")
        