
from torch.utils.data import Dataset 

#--------------- Dataset object for multi-view data-------------------
class TimeDataset(Dataset):
  def __init__(self, raw_data, window_size, seq_start, seq_end): #
    self.raw_data = raw_data
    self.window_size = window_size
    
    self.seq_start = seq_start
    self.seq_end = seq_end
    
    #self.sequence_length = raw_data[0].shape[1]
    #self.num_sequences = self.sequence_length - self.window_size + 1
    
    #self.X_train,Y_train,X_val,Y_val = self.process(raw_data)
    self.X, self.Y = self.process(raw_data)
    
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):

    x = self.X[idx]#.double()
    y = self.Y[idx]#.double()

    return x, y
  
  def process(self, data):    
    
    #train_seq_end = int((1 - self.val_ratio)*(self.num_sequences))
    #val_seq_start = train_seq_end
    #val_seq_end = self.num_sequences - 1
    
    X, Y = [], []
    
    for i in range(self.seq_start, self.seq_end):           
     
      x_views, y_views = [], []
     
      for view in data :
       x_view = view[:, i:(i+self.window_size)]
       y_view = view[:,i+self.window_size]
       y_view = y_view.reshape(y_view.shape[0],-1)
    
       x_views.append(x_view)
       y_views.append(y_view)
     
      X.append(x_views)
      Y.append(y_views)
      
    return X,Y 

#--------------- Dataset object for single-view data-------------------
class SingleView_Dataset(Dataset):
  def __init__(self, raw_data, window_size, seq_start, seq_end): 
    self.raw_data = raw_data
    self.window_size = window_size
    
    self.seq_start = seq_start
    self.seq_end = seq_end
    
    self.X, self.Y = self.process(raw_data)
    
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):

    x = self.X[idx]
    y = self.Y[idx]

    return x, y
  
  def process(self, data):
    
    X, Y = [], []
    
    for i in range(self.seq_start, self.seq_end):           
     
      features = self.raw_data[:, i:(i+self.window_size)]
      target = self.raw_data[:,i+self.window_size]
      
      X.append(features)
      Y.append(target)
      
    return X,Y   
