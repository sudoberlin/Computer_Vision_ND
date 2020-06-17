import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import math

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer
        # turns words into vector of specific size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #now lstm layer will take the embedded word vector of a specific size 
        # as inputs and outputs hidden ststes (hidden_dim)
        self.lstm = nn.LSTM(input_size = embed_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first = True)
        # Linear_layer--> maps hidden state output dimension to the vocab size
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        # now create embedded word vectors in captions batch for each token
        # [batch_size,caption_lenghth 
        embeds = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim = 1)
        
        # lstm_out.shape--> batch_size, caption_length, hidden_size
        lstm_out, _ = self.lstm(inputs)
        # scores for most likely words
        # outputs.shape-->batch_size, caption_length, vocab_Size
        outputs = self.hidden2vocab(lstm_out)
        return outputs
    
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        
        # initialize the hidden states as inputs
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                 torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        # now to get the caption feed the lstm output and hidden states back to itself
        for i in range(max_len):
            # batch_size = 1, sequence_length = 1-->(1,1,embedsize)
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.hidden2vocab(lstm_out) #1,1,vocab_size
            outputs = outputs.squeeze(1)
            word_id = outputs.argmax(dim = 1)
            caption.append(word_id.item())
            
            # input for next iteratons
            inputs = self.word_embeddings(word_id.unsqueeze(0))
            
        return caption
            
            
            
            
        