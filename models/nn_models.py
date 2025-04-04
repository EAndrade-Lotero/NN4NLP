import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch import Tensor
from prettytable import PrettyTable

PAD_IDX = 1

def summary(self) -> None:
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in self.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f'Total Trainable Params: {total_params}')
    if next(self.parameters()).is_cuda:
        print('Model device: cuda')
    elif next(self.parameters()).is_mps:
        print('Model device: mps')
    else:
        print('Model device: cpu')

setattr(nn.Module, "summary", summary)


class FFNLanguageModeler(nn.Module):
    '''Simple multi-layer perceptron as a causal language model'''

    def __init__(self, vocab_size, embedding_dim, hidden_size, context_size):
        super(FFNLanguageModeler, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = torch.reshape( embeds, (-1,self.context_size * self.embedding_dim))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out


class MLP(nn.Module):
    def __init__(
                self, 
                embedding_dim:int, 
                hidden_dims:List[int], 
                out_dim:int, 
                dropout:float
            ) -> None:
        super(MLP, self).__init__()
        layers = []
        in_dim = embedding_dim
        
        # Iterate over each hidden dimension to construct the layers
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hdim  # update in_dim for the next layer
        
        # Add the final output layer
        layers.append(nn.Linear(in_dim, out_dim))
        
        # Wrap the layers in a Sequential module
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    '''Define the PositionalEncoding class as a PyTorch module for adding positional information to token embeddings'''

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    

class BERTEmbedding (nn.Module):
    '''Include token embedding, segment embedding and positional encoding'''
    def __init__(self, vocab_size, emb_size, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.segment_embedding = nn.Embedding(3, emb_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, bert_inputs, segment_labels=False):
        my_embeddings = self.token_embedding(bert_inputs)
        if self.train:
          x = self.dropout(my_embeddings + self.positional_encoding(my_embeddings) + self.segment_embedding(segment_labels))
        else:
          x = my_embeddings + self.positional_encoding(my_embeddings)
        return x
    

class BERT(torch.nn.Module):
    
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        vocab_size: The size of the vocabulary.
        d_model: The size of the embeddings (hidden size).
        n_layers: The number of Transformer layers.
        heads: The number of attention heads in each Transformer layer.
        dropout: The dropout rate applied to embeddings and Transformer layers.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        # Embedding layer that combines token embeddings and segment embeddings
        self.bert_embedding = BERTEmbedding(vocab_size, d_model, dropout)
        # Transformer Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model*2, dropout=dropout,batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # Linear layer for Next Sentence Prediction
        self.nextsentenceprediction = nn.Linear(d_model, 2)
        # Linear layer for Masked Language Modeling
        self.masked_language = nn.Linear(d_model, vocab_size)

    def forward(self, bert_inputs, segment_labels):
        """
        bert_inputs: Input tokens.
        segment_labels: Segment IDs for distinguishing different segments in the input.
        mask: Attention mask to prevent attention to padding tokens.

        return: Predictions for next sentence task and masked language modeling task.
        """

        try:
            padding_mask = (bert_inputs == PAD_IDX).transpose(0, 1)
        except:
            padding_mask = bert_inputs == PAD_IDX            
            
        # Generate embeddings from input tokens and segment labels
        my_bert_embedding = self.bert_embedding(bert_inputs, segment_labels)
        # Pass embeddings through the Transformer encoder
        transformer_encoder_output = self.transformer_encoder(my_bert_embedding,src_key_padding_mask=padding_mask)
        # Next sentece prediction
        next_sentence_prediction = self.nextsentenceprediction(transformer_encoder_output[ 0,:])      
        # Masked Language Modeling: Predict all tokens in the sequence
        masked_language = self.masked_language(transformer_encoder_output)
        return  next_sentence_prediction, masked_language