
import torch
import torch.nn as nn

from utils.generic_utils import initial_adj_matrix
from utils.generic_utils import initialize_zero_tensor
from constants.hyper_params import *

#------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        super(Block, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.dropout(x)
        return x
      
#------------------------------------------------------------------------------------------
class MV_MODEL(nn.Module):
    def __init__(self, input_size, view_dims):
        super(MV_MODEL, self).__init__()
        
        self.view_dims = view_dims        
        
        # Number Of Views
        self.num_views = len(view_dims)
        
        # Overall view attentions
        #V = initial_adj_matrix(self.num_views)
        V = initialize_zero_tensor(self.num_views)
        self.view_attn = nn.Parameter(V) 
        
        # Per View Attentions and transformations
        self.per_view_attns = nn.ParameterList()
        
        #num_nodes_next_layer = input_size//2
        self.hidden_size = input_size//2
        
        views_blocks = []
        
        for view_dim in view_dims:
          #A_tensor = initial_adj_matrix(view_dim)
          A_tensor = initialize_zero_tensor(view_dim)
          per_view_attn = nn.Parameter(A_tensor)
          self.per_view_attns.append(per_view_attn)
          
          blocks = []
          block1 = Block(3*input_size, 2*input_size, view_dim)
          blocks.append(block1)
          block2 = Block(2*input_size, input_size, view_dim)
          blocks.append(block2)
          block3 = Block(input_size, self.hidden_size, view_dim)
          blocks.append(block3)
          
          fc4 = nn.Linear(self.hidden_size, input_size) 
          blocks.append(fc4)
          bn4 = nn.BatchNorm1d(view_dim)
          blocks.append(bn4)
          
          #block4 = Block(hidden_size, input_size, view_dim) 
          #blocks.append(block4)
          #fc8 = nn.Linear(input_size, input_size) 
          #blocks.append(fc8)
          
          block5 = Block(2*input_size, input_size, view_dim)
          blocks.append(block5)
          block6 = Block(input_size, self.hidden_size, view_dim)
          blocks.append(block6)
          fc7 = nn.Linear(self.hidden_size, 1)
          blocks.append(fc7)
          
          view_block = nn.ModuleList(blocks)
          views_blocks.append(view_block)
          
        self.views_layers = nn.ModuleList(views_blocks)
        self.view_ll1 = Block(input_size, self.hidden_size, self.num_views)
        self.view_ll2 = Block(self.hidden_size, input_size, self.num_views)
        #self.view_ll1 = nn.Linear(input_size, hidden_size)
        #self.view_ll2 = nn.Linear(hidden_size, input_size)
        
    #Adding the constraint that diagonal elements must be zeros for attentions
    def zero_diagonal(self):
        # Zero out the diagonal elements
        with torch.no_grad():
            # Constraint applied on attentions among views
            self.view_attn.fill_diagonal_(0)
            
            # Constraint applied on attentions within each view
            for k in range(self.num_views):
             self.per_view_attns[k].fill_diagonal_(0)
             
    def forward(self, h_views, x_views):
        
        node_embeddings = []
        view_embeddings = []
        node_predictions = []
        
        #print('bn_layers : ',self.bn_layers)
        
        for k in range(self.num_views):    
         
         N = self.view_dims[k]     
         A = self.per_view_attns[k]
         
         #h = h_views[k].to(A.device)  
         #x = x_views[k].to(A.device) 
         h = h_views[k]
         x = x_views[k]
         
         vec3 = torch.matmul(A, x)
         vec3_shape_0 = vec3.shape[0]
         vec1 = h  
         # To adjust the last batch size
         vec1 = vec1[:vec3_shape_0,:,:]
         vec2 = torch.matmul(A, vec1)
         
         h0 = torch.cat((vec1, vec2, vec3), dim=2) 
         #h2 = vec1 + vec2 # + vec3
         #print('h0 : ', h0.shape)         
         
         h1 = self.views_layers[k][0](h0)
         #print('h1 : ', h1.shape) 
         h2 = self.views_layers[k][1](h1)
         #print('h2 : ', h2.shape) 
         h3 = self.views_layers[k][2](h2)  
         #print('h3 : ', h3.shape) 
         #print('self.views_layers[k][3] : ',self.views_layers[k][3])
         #h3 = self.views_layers[k][2](h0)  
         h4 = self.views_layers[k][3](h3) 
         #print('self.views_layers[k][4] : ',self.views_layers[k][4])
         #print('h4 : ', h4.shape) 
         x_reconstructed = self.views_layers[k][4](h4) #h4 #
         
         #print('x_reconstructed : ',x_reconstructed.shape)
         
         #view_embedding = x_reconstructed.sum(dim=1, keepdim=True)
         view_embedding = x_reconstructed.sum(dim=1, keepdim=True)/N
         #view_embedding = self.views_layers[k][5](x_reconstructed.sum(dim=1, keepdim=True)/N)
         
         # Expanded vector is creating multiple copies of view embedding
         # the number of copies is the number of nodes in the view     
         view_embedding_expanded = view_embedding.repeat(1, N, 1)
         
         # Each view embedding is prepended before x_reconstructed(h) so that it is used in the 
         # prediction for each node in h
         h5 = torch.cat((view_embedding_expanded, x_reconstructed), dim=2)      
         
         # The [view embedding,x_reconstructed] is used for prediction using a 
         # fully connected layer
         
         # Old
         h6 = self.views_layers[k][5](h5)
         h7 = self.views_layers[k][6](h6)
         x_predicted = self.views_layers[k][7](h7)
         
         # New
         #h6 = self.views_layers[k][6](h5)
         #h7 = self.views_layers[k][7](h6)
         #x_predicted = self.views_layers[k][8](h7)
         
         node_predictions.append(x_predicted)          
         node_embeddings.append(x_reconstructed) 
         view_embeddings.append(view_embedding)
        
        view_embeddings = torch.cat(view_embeddings, dim=1)
        view_embeddings_attn = torch.matmul(self.view_attn, view_embeddings) 
        
        #Applying GIN to view embeddings
        v0 = view_embeddings + view_embeddings_attn # The eqn used for all exp till proposal
        #v0 = view_embeddings_attn
        #print('v0 : ', v0.shape)  
        #print('ll1 : ',self.view_ll1)  
        v1 = self.view_ll1(v0)
        #print('v1 : ', v1.shape)  
        v2 = self.view_ll2(v1)
        
        #print('v2 computed')
        return node_embeddings, node_predictions, view_embeddings, v2
        #return node_embeddings, node_predictions, view_embeddings, view_embeddings_attn 
      
    def compute_losses(self, h_curr, x_curr, x_pred, y, z, z_, lamdas):
        
        total_reconstruction_loss = torch.zeros((1, 1), requires_grad=True) 
        total_prediction_loss = torch.zeros((1, 1), requires_grad=True)
        view_recons_loss = torch.zeros((1, 1), requires_grad=True)
      
        criterion = nn.MSELoss()
        
        for k in range(self.num_views):
          
          reconstruction_loss = criterion(x_curr[k], h_curr[k])
          total_reconstruction_loss = total_reconstruction_loss + reconstruction_loss
          
          prediction_loss = criterion(y[k], x_pred[k])
          total_prediction_loss = total_prediction_loss + prediction_loss 
        
        view_recons_loss = criterion(z, z_)       
        total_reconstruction_loss = total_reconstruction_loss/self.num_views
        total_prediction_loss = total_prediction_loss/self.num_views
        
        # Computing norms
        view_attention_norm = torch.norm(self.view_attn, 1)
        
        per_view_attention_norms = 0
        for k in range(self.num_views):
         per_view_attention_norms = per_view_attention_norms + torch.norm(self.per_view_attns[k], 1)
          
        per_view_attention_norms = per_view_attention_norms/self.num_views
        
        #Total loss is a combination of all the losses plus norms
        total_loss = lamdas[0] * total_reconstruction_loss
        total_loss = total_loss + lamdas[1] * view_recons_loss 
        total_loss = total_loss + lamdas[2] * total_prediction_loss
        total_loss = total_loss + lamdas[3] * view_attention_norm
        total_loss = total_loss + lamdas[4] * per_view_attention_norms
        
        return total_reconstruction_loss, view_recons_loss, total_prediction_loss, total_loss, view_attention_norm, per_view_attention_norms
          
      
