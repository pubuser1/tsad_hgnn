
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import random
from sklearn.manifold import TSNE

colors2 = ['purple', 'red', 'green', 'orange', 'blue']
colors1 = [(0.8, 0.4, 0.2), (0.1, 0.5, 0.6), (0.1, 0.1, 0.1), (0.1, 0.6, 0.1), (0.5, 0.5, 0.5)]

def plot_multi_view_data_1(Data):
  
  fig, axs = plt.subplots(len(Data), 1, figsize=(10, 6))
  
  for i, view in enumerate(Data):
  
    # Plot each variable in the view
    for j, variable in enumerate(view):
      axs[i].plot(variable, color=colors1[j])
    axs[i].set_title('View'+str(i+1))
  
  plt.tight_layout()
  plt.show()
  
def plot_multi_view_data(Data):
  
  fig, axs = plt.subplots(len(Data), 1, figsize=(10, 6))
  
  features = Data[0].shape[0]
  #seq_len = Data[0].shape[1]
  #x = np.arange(seq_len)
  
  start = 4000
  end = 6000
  
  for i, view in enumerate(Data):  
    for j in range(features): # getting each variable
       axs[i].plot(view[j, start:end]) #seq_len
    axs[i].set_title('View'+str(i+1))
    
  plt.tight_layout()
  plt.show()
  
def plot_multi_view_data_forecasted(Data, Val_Pred_Data):
  
  fig, axs = plt.subplots(len(Data), 1, figsize=(10, 6))
  
  for i, view in enumerate(Data):
  
    View_Val_Data = Val_Pred_Data[i]
    indices = np.arange(view.shape[1]-View_Val_Data.shape[1], view.shape[1])
    
    # Plot each variable in the view
    for j, variable in enumerate(view):
      axs[i].plot(variable, color=colors1[j])
      axs[i].plot(indices, View_Val_Data[j], color=colors1[j], linestyle='--')
      
    axs[i].set_title('View'+str(i+1))
  
  plt.tight_layout()
  plt.show()
  
def plot_single_view_data_forecasted_1(Data, Val_Pred_Data):
  
    plt.figure(figsize=(10,4))
    indices = np.arange(Data.shape[1]-Val_Pred_Data.shape[1], Data.shape[1])
    #print('Indices : ',indices)
    #for i in range(Data.shape[0]): # getting each variable
        #plt.plot(Data[i])
        #plt.plot(indices,Val_Pred_Data[i],linestyle='--')
        
    plt.plot(Data[0])
    plt.plot(indices,Val_Pred_Data[0],linestyle='--')
    #plt.axvline(x=indices[0], color='r')
    plt.show()
  
def plot_single_view_data_forecasted_2(Data, Val_Pred_Data):
  
    fig, axs = plt.subplots(len(Data), 1, figsize=(10, 6))
    #print('Data.shape : ',Data.shape)
    #print('Val_Pred_Data.shape : ',Val_Pred_Data.shape)
    indices = np.arange(Data.shape[1]-Val_Pred_Data.shape[1], Data.shape[1])
    #print(indices)
    
    for i in range(Data.shape[0]): # getting each variable
        axs[i].plot(Data[i])
        axs[i].plot(indices,Val_Pred_Data[i],linestyle='--')
        axs[i].axvline(x=indices[0], color='r',linestyle=':')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        
    plt.show()
    
def plot_single_view_data_forecasted(Data, Val_Pred_Data, title): #Val_Recons_Data
  
    fig, axs = plt.subplots(len(Data), 1, figsize=(10, 6))
    #print('Data.shape : ',Data.shape)
    #print('Val_Pred_Data.shape : ',Val_Pred_Data.shape)
    #print('Val_Recons_Data.shape : ',Val_Recons_Data.shape)
    val_plot_length = Val_Pred_Data.shape[1]
    indices = np.arange(Data.shape[1]-Val_Pred_Data.shape[1], Data.shape[1])
    #print(indices)
    
    for i in range(Data.shape[0]): # getting each variable
        axs[i].plot(Data[i])
        axs[i].plot(indices,Val_Pred_Data[i],linestyle='--')
        #axs[i].plot(indices,Val_Recons_Data[i][-val_plot_length:],linestyle='--')
        axs[i].axvline(x=indices[0], color='r',linestyle=':')
        #axs[i].set_xticks([])
        #axs[i].set_yticks([])
        
    path = title+'.png'
    plt.savefig(path)
    plt.show()
    
def plot_data(D,title):
 plt.figure(figsize=(10,4))
 seq_len = D.shape[1]
 x = np.arange(seq_len) # getting the time axis labels
 for i in range(D.shape[0]): # getting each variable
    plt.plot(x, D[i, :seq_len], label=f'd{i}')
 plt.title(title)
 plt.legend()
 plt.show()
 #path = title+'.png'
 #plt.savefig(path)
 #plt.clf()

def plot_data_split_features(Data,title,column_names):
   fig, axs = plt.subplots(len(Data), 1, figsize=(15, 6))
 
   for i in range(Data.shape[0]):
         axs[i].plot(Data[i])
         #axs[i].set_title(column_names[i], rotation=90, loc='left')
         axs[i].yaxis.set_label_position('right')
         axs[i].set_ylabel(column_names[i], rotation=90, fontsize=12, labelpad=10)
         #if (i < Data.shape[0]-1):
              #axs[i].set_xticks([])
              #axs[i].set_yticks([])
         
   #plt.title(title)
   #plt.legend()
   plt.xlabel('Timestamp')
   plt.subplots_adjust(hspace=0.4)
   path = title+'.png'
   plt.savefig(path)
   plt.show()
 
def plot_losses_1(num_epochs, train_loss_list): 
 plt.figure(figsize=(10,6))
 plt.plot(range(num_epochs), train_loss_list, label='Training Loss')
 plt.legend()
 #plt.show()
 path = 'Train_Loss.png'
 plt.savefig(path)
 plt.clf()
 
def plot_losses(num_epochs, loss_dict): 
 plt.figure(figsize=(10,6))
 for label, loss_list in loss_dict.items():
  plt.plot(range(num_epochs), loss_list, label=label)
 plt.legend()
 plt.show()
 #path = 'Training_Losses.png'
 #plt.savefig(path)
 #plt.clf()

def heatMap(A, title):
 sns.heatmap(A, cmap='viridis', annot = True)
 path = title+'.png'
 plt.savefig(path)
 plt.clf()
 
def plot_roc_auc(fprs, tprs, auc_score):
  plt.figure(figsize=(8, 6))
  plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0]) #1.05
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.show()
  
def plot_tSNE(infer_embeddings, pred_labels, gt_labels):
    # First objective is to obtain a single tensor comprising of embeddings for all features of the dataset
    # across all instants in inference data
    tensors = []
    
    # infer_embeddings is a list of lists ; Each list consists of embeddings for each sample in the inference data; 
    # Each embedding shape is [curr_instant(#1), #features_per_view, #embed_dim], ex:[1, 13, 10]
    # Hence for each view an embedding tensor of shape #features_per_view x embed_dimension is generated
    # for each inference sample i.e. for each instant of the inference data
    # Hence STEP 1 : for each tensor list corresponding to a view, concatenate it along 
    # the dimension of the instant i.e first dimension
    for tensor_list in infer_embeddings:
      tensors.append(torch.cat(tensor_list, dim=0))
     
    # Output from STEP 1 is a list of tensors corresponding to each view,
    # with shape #instants x #features_per_view x #embed_dim 
    # STEP 2 : concatenate each tensor corresponding to a view along the second dimension
    embed_tensor = torch.cat(tensors, dim=1)
    print(embed_tensor.shape)
    
    # Output from STEP 2 is a tensor of required shape, #instants x #total_features x #embed_dim  
    # Second objective is to choose the embeddings corresponding to a random feature 
    # and seperately for anomalous and non-anomalous samples
    # STEP 3 : Choosing a random feature ID
    num_features = embed_tensor.shape[1]
    #feature_id = random.randint(0, num_features)
    feature_id = 46
    
    # STEP 4 : Choosing indices corresponding to samples where both predicted and ground truth labels are 1
    # i.e. test instants/samples for which model had true positive outcomes 
    # taking a subset now to have better visualizations
    indices_both_1 = [index for index, (val1, val2) in enumerate(zip(pred_labels, gt_labels)) if val1 == val2 == 1] 
    subset_indices_1 = random.sample(indices_both_1, 5)    
    anomalous_embeddings = embed_tensor[subset_indices_1, feature_id]
    #anomalous_embeddings = embed_tensor[indices_both_1, feature_id]
    print(anomalous_embeddings.shape)
    anomalous_embeddings_array = anomalous_embeddings.numpy()
    
    # STEP 5 : Choosing indices corresponding to samples where both predicted and ground truth labels are 0
    # i.e. test instants/samples for which model had true negative outcomes
    # taking a subset now to have better visualizations
    indices_both_0 = [index for index, (val1, val2) in enumerate(zip(pred_labels, gt_labels)) if val1 == val2 == 0] 
    subset_indices_0 = random.sample(indices_both_0, 100)
    normal_embeddings = embed_tensor[subset_indices_0, feature_id]  
    #normal_embeddings = embed_tensor[indices_both_0, feature_id]  
    print(normal_embeddings.shape)
    normal_embeddings_array = normal_embeddings.numpy()
    
    # Third objective is to generate t-SNE embeddings for the chosen embeddings
    # STEP 6 : Generate the t-SNE embeddings
    #tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(n_components=2, random_state=22)
    embeddings_2d_normal = tsne.fit_transform(normal_embeddings_array)
    embeddings_2d_anomalous = tsne.fit_transform(anomalous_embeddings_array)
    
    # STEP 7 : Plot the t-SNE embeddings
    plt.figure(figsize=(8, 6))
    #plt.xticks([])
    #plt.yticks([])
    plt.scatter(embeddings_2d_normal[:, 0], embeddings_2d_normal[:, 1], alpha=0.65, label='Normal Data') #, alpha=0.5
    plt.scatter(embeddings_2d_anomalous[:, 0], embeddings_2d_anomalous[:, 1], label='Anomalous Data') #, color='r', alpha=0.5
    plt.title('Feature '+str(feature_id)+' Embeddings')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    
    path = 'Feature_'+str(feature_id)+'.png'
    plt.savefig(path)
    plt.show()
