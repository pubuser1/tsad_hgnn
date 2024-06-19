num_epochs = 200 
window_size = 10 
batch = 512 

learning_rate = 0.01 

val_ratio = 0.2 
test_ratio = 0.4 
train_seq_start = 1

best_val_loss = float('inf')
patience = 15
current_patience = 0

lamdas = (1, 1, 1, 0, 0) 
