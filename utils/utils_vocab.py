import pickle
import json
import torch 
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import (
    List, Iterable, Union, Tuple
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score

class Encoded :
    '''Class to store encoded tokens. Emulates HuggingFace's Encoding'''

    def __init__(self, tokens, ids):
        self.tokens = tokens
        self.ids = ids

    def extend(self, encoded):
        self.tokens += encoded.tokens
        self.ids += encoded.ids


class BasicTokenizer :
    '''Emulates a HuggingFace-like tokenizer'''

    def __init__(self, tokenizer, special_tokens:List[str]) -> None:
        self.tokenizer = tokenizer
        assert(isinstance(special_tokens, list))
        assert(np.all(isinstance(x, str) for x in special_tokens))
        self.special_tokens = special_tokens
        self.unknown_token = special_tokens[0]
        self.has_stoi = False

    def initialize_from_iterable(self, list_sentences:Iterable) -> None:
        assert isinstance(list_sentences, Iterable)
        self.itos = self.special_tokens
        self.stoi = {token:i for i, token in enumerate(self.special_tokens)}
        for sentence in list_sentences:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if token not in self.itos:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)
        self.has_stoi = True

    def encode(self, text:Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        if isinstance(text, str):
            return self._encode_str(text)
        else:
            inicial = True
            for sentence in text:
                if inicial:
                    encoded = self._encode_str(sentence)
                    inicial = False
                else:
                    new_encoded = self._encode_str(sentence)
                    encoded.extend(new_encoded)
            return encoded

    def decode(self, list_ids:Union[List[int], List[List[int]]]) -> Union[List[str], List[List[str]]]:
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        if np.all(isinstance(id, int) for id in list_ids):
            return self._decode_str(list_ids)
        else:
            list_tokens = list()
            for ids in list_ids:
                tokens = self._decode_str(ids)
                list_tokens.append(tokens)
            return list_tokens

    def _encode_str(self, sentence:str) -> List[int]:
        tokens = self.tokenizer(sentence)
        indices = [self.stoi.get(token, 0) for token in tokens]
        tokens = [self.unknown_token if id == 0 else tokens[i] for i, id in enumerate(indices)]
        encoded = Encoded(tokens, indices)
        return encoded
    
    def _decode_str(self, list_ids:str) -> List[int]:
        try:
            tokens = [self.itos[id] for id in list_ids]
        except Exception as e:
            for id in list_ids:
                self.itos[id]
            raise Exception(e)
        return tokens
    
    def get_vocab_size(self):
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        return len(self.itos)

    def save(self, tokenizer_file):
        with open(tokenizer_file, 'wb') as f:  # open a text file
            pickle.dump(self.stoi, f) # serialize the list

    @staticmethod
    def create_using_stoi(tokenizer, special_tokens:List[str], tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            stoi = pickle.load(f) # deserialize using load()
        assert(isinstance(stoi, dict))
        me_tokenizer = BasicTokenizer(tokenizer, special_tokens)
        me_tokenizer.stoi = stoi
        me_tokenizer.itos = list(stoi.keys())
        me_tokenizer.has_stoi = True
        return me_tokenizer


class BasicBertTokenizer(BasicTokenizer):

    def __init__(self, tokenizer) -> None:
        special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        super().__init__(tokenizer, special_tokens)
        self.pad_token = special_tokens[1]
        self.cls_token = special_tokens[2]
        self.sep_token = special_tokens[3]
        self.mask_token = special_tokens[4]
        self.pad_id = 1
        self.cls_id = 2
        self.sep_id = 3
        self.mask_id = 4

    @staticmethod
    def create_using_stoi(tokenizer, tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            stoi = pickle.load(f) # deserialize using load()
        assert(isinstance(stoi, dict))
        me_tokenizer = BasicBertTokenizer(tokenizer)
        me_tokenizer.stoi = stoi
        me_tokenizer.itos = list(stoi.keys())
        me_tokenizer.has_stoi = True
        return me_tokenizer
    

class BertTools:

    def __init__(self, tokenizer, ids=True):
        self.tokenizer = tokenizer
        assert(isinstance(tokenizer, BasicBertTokenizer))
        self.ids = ids
        if self.ids:
            self.mask_ = self.tokenizer.mask_id
            self.pad_ = self.tokenizer.pad_id
            self.cls_ = self.tokenizer.cls_id
            self.sep_ = self.tokenizer.sep_id
        else:
            self.mask_ = self.tokenizer.mask_token
            self.pad_ = self.tokenizer.pad_token
            self.cls_ = self.tokenizer.cls_token
            self.sep_ = self.tokenizer.sep_token

    def bernoulli_true_false(self, p):
        '''Create a Bernoulli distribution with probability p'''
        bernoulli_dist = torch.distributions.Bernoulli(torch.tensor([p]))
        # Sample from this distribution and convert 1 to True and 0 to False
        return bernoulli_dist.sample().item() == 1
    
    def Masking(self, token):

        pad_ = self.pad_
        mask_ = self.mask_

        # Decide whether to mask this token (20% chance)
        mask = self.bernoulli_true_false(0.2)

        # If mask is False, immediately return with '[PAD]' label
        if not mask:
            return token, pad_

        # If mask is True, proceed with further operations
        # Randomly decide on an operation (50% chance each)
        random_opp = self.bernoulli_true_false(0.5)
        random_swich = self.bernoulli_true_false(0.5)

        # Case 1: If mask, random_opp, and random_swich are True
        if mask and random_opp and random_swich:
            # Replace the token with '[MASK]' and set label to a random token
            token_ = mask_
            mask_label = mask_

        # Case 2: If mask and random_opp are True, but random_swich is False
        elif mask and random_opp and not random_swich:
            # Leave the token unchanged and set label to the same token
            token_ = token
            mask_label = token

        # Case 3: If mask is True, but random_opp is False
        else:
            # Replace the token with '[MASK]' and set label to the original token
            token_ = mask_
            mask_label = token

        return token_, mask_label    

    def prepare_for_mlm(self, text):
        """
        Prepares tokenized text for BERT's Masked Language Model (MLM) training.

        """
        bert_input = []  # List to store sentences processed for BERT's MLM
        bert_label = []  # List to store labels for each token (mask, random, or unchanged)
        raw_tokens = []  # List to store raw tokens if needed

        if self.ids:
            token_list = self.tokenizer.encode(text).ids
        else:
            token_list = self.tokenizer.encode(text).tokens
        for token in token_list:
            # Apply BERT's MLM masking strategy to the token
            masked_token, mask_label = self.Masking(token)

            # Append the processed token and its label to the current sentence and label list
            bert_input.append(masked_token)
            bert_label.append(mask_label)

        # Return the prepared lists for BERT's MLM training
        return bert_input, bert_label

    def process_for_nsp(self, input_sentences_pair, input_masked_labels_pair, relations):
        """
        Prepares data for understanding logical relationship.

        Args:
        input_sentences (list): List of tokenized sentences.
        input_masked_labels (list): Corresponding list of masked labels for the sentences.

        Returns:
        bert_input (list): List of sentence pairs for BERT input.
        bert_label (list): List of masked labels for the sentence pairs.
        is_next (list): Binary label list.
        """

        cls_ = self.cls_
        sep_ = self.sep_
        pad_ = self.pad_

        # Verify that both input lists are of the same length and have a sufficient number of sentences
        if len(input_sentences_pair) != len(input_masked_labels_pair):
            raise ValueError("Both lists, input_sentences_pair and input_masked_labels_pair, must have the same number of items.")
        if len(input_sentences_pair) != len(relations):
            raise ValueError("Both lists, input_sentences_pair and relations, must have the same number of items.")

        bert_input = []
        bert_label = []
        is_next = []

        for sentence_pair, masked_pair, relation in zip(input_sentences_pair, input_masked_labels_pair, relations):
            # append list and add  '[CLS]' and  '[SEP]' tokens
            bert_input.append([[cls_] + sentence_pair[0] + [sep_], sentence_pair[1] + [sep_]])
            bert_label.append([[cls_] + masked_pair[0] + [pad_], masked_pair[1]+ [pad_]])
            is_next.append(relation)  # Label 1 indicates these sentences have the required logical relationship

        return bert_input, bert_label, is_next

    def process_for_classification(self, input_sentences, input_masked_labels, classes):
        """
        Prepares data for classification.

        Args:
        input_sentences (list): List of tokenized sentences.
        input_masked_labels (list): Corresponding list of masked labels for the sentences.

        Returns:
        bert_input (list): List of sentences for BERT input.
        bert_label (list): List of masked labels.
        relations (list): Label list.
        """

        cls_ = self.cls_
        sep_ = self.sep_
        pad_ = self.pad_

        # Verify that both input lists are of the same length and have a sufficient number of sentences
        if len(input_sentences) != len(input_masked_labels):
            raise ValueError("Both lists, input_sentences_pair and input_masked_labels_pair, must have the same number of items.")
        if len(input_sentences) != len(classes):
            raise ValueError("Both lists, input_sentences_pair and relations, must have the same number of items.")

        bert_input = []
        bert_label = []
        segment_label = []

        for sentence, masked, relation in zip(input_sentences, input_masked_labels, classes):
            # append list and add  '[CLS]' and  '[SEP]' tokens
            bert_input.append([cls_] + sentence + [sep_])
            bert_label.append([pad_] + masked + [pad_])
            segment_label.append([0] * (len(sentence) + 2))
                             
        return bert_input, bert_label, segment_label, classes

    def zero_pad_list_pair(self, pair_):
        pad_ = self.pad_
        pair = deepcopy(pair_)
        max_len = max(len(pair[0]), len(pair[1]))
        #append [PAD] to each sentence in the pair till the maximum length reaches
        pair[0].extend([pad_] * (max_len - len(pair[0])))
        pair[1].extend([pad_] * (max_len - len(pair[1])))
        return pair[0], pair[1]

    def prepare_bert_final_inputs(self, bert_inputs, bert_labels, is_nexts):
        """
        Prepare the final input lists for BERT training.
        """

        #flatten the tensor
        flatten = lambda l: [item for sublist in l for item in sublist]

        bert_inputs_final, bert_labels_final, segment_labels_final, is_nexts_final = [], [], [], []
        for bert_input, bert_label,is_next in zip(bert_inputs, bert_labels,is_nexts):
            # Create segment labels for each pair of sentences
            segment_label = [[1] * len(bert_input[0]), [2] * len(bert_input[1])]
            # Zero-pad the bert_input and bert_label and segment_label
            bert_input_padded = self.zero_pad_list_pair(bert_input)
            bert_label_padded = self.zero_pad_list_pair(bert_label)
            segment_label_padded = self.zero_pad_list_pair(segment_label,pad=0)
            # Flatten lists
            bert_inputs_final.append(flatten(bert_input_padded))
            bert_labels_final.append(flatten(bert_label_padded))
            segment_labels_final.append(flatten(segment_label_padded))
            is_nexts_final.append(is_next)

        return bert_inputs_final, bert_labels_final, segment_labels_final, is_nexts_final

    @staticmethod
    def evaluate(dataloader, model, loss_fn_mlm, loss_fn_nsp) -> Tuple[float, float, float]:
        '''Evaluate a BERT model'''

        model.eval()  # Turn off dropout and other training-specific behaviors

        total_loss = 0
        total_next_sentence_loss = 0
        total_mask_loss = 0
        total_batches = 0
        total_count = 0
        y_true, y_predict = list(), list()
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            for bert_inputs, bert_labels, segment_labels, is_nexts in dataloader:
                # Forward pass
                #print(f'{bert_inputs.shape=} --- {segment_labels.shape=}')
                next_sentence_prediction, masked_language = model(bert_inputs, segment_labels)

                # Calculate loss for next sentence prediction
                # Ensure is_nexts is of the correct shape for CrossEntropyLoss
                next_loss = loss_fn_nsp(next_sentence_prediction, is_nexts.view(-1))

                # Calculate loss for predicting masked tokens
                # Flatten both masked_language predictions and bert_labels to match CrossEntropyLoss input requirements
                mask_loss = loss_fn_mlm(masked_language.view(-1, masked_language.size(-1)), bert_labels.view(-1))

                # Sum up the two losses
                loss = next_loss + mask_loss
                if torch.isnan(loss):
                    continue
                else:
                    total_loss += loss.item()
                    total_next_sentence_loss += next_loss.item()
                    total_mask_loss += mask_loss.item()
                    total_batches += 1

                #print('next_sentence_pred:', next_sentence_prediction)
                logits = torch.softmax(next_sentence_prediction, dim=1)
                #print('logits: ', logits)
                prediction = torch.argmax(logits, dim=1)         
                total_count += is_nexts.size(0)
                y_true.extend(is_nexts.view(-1).cpu().numpy().tolist())
                y_predict.extend(prediction.cpu().numpy().tolist())

        avg_loss = total_loss / (total_batches + 1)
        avg_next_sentence_loss = total_next_sentence_loss / (total_batches + 1)
        avg_mask_loss = total_mask_loss / (total_batches + 1)

        #print(f"Average Loss: {avg_loss:.4f}, Average Next Sentence Loss: {avg_next_sentence_loss:.4f}, Average Mask Loss: {avg_mask_loss:.4f}")
        acc = accuracy_score(y_true, y_predict)
        f1 = f1_score(y_true, y_predict)    
        #print(f"Accuracy: {acc}")
        #print(f"F1 score: {f1}")
        return avg_loss, acc, f1


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    

# Define a custom dataset with a pandas dataframe
class PandasDataset(Dataset):
    def __init__(
                self, 
                df:pd.DataFrame, 
                x_cols:List[str], 
                y_col:str
            ) -> None:
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.x_cols].to_list()[0]
        y = row[self.y_col]
        return x, y
    

class BERTDataset(Dataset):
    '''
    Bert dataset that assumes four columns: 
        - BERT Input, which consist of two sentences, starting with [CLS] and both ended by [SEP]
        - BERT Label, which contains the target masked token and all other tokens are [PAD]
        - Segment Label, which labels the tokens of the sentences: 1 for tokens in the first sentence and 2 for tokens in the second
        - Is Next, which labels whether the second sentence follows the first
    '''
    def __init__(self, df:pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            bert_input = torch.tensor(json.loads(row['BERT Input']), dtype=torch.long)
            bert_label = torch.tensor(json.loads(row['BERT Label']), dtype=torch.long)
            segment_label = torch.tensor(json.loads(row['Segment Label']), dtype=torch.long)
            # segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
            is_next = torch.tensor(row['Is Next'], dtype=torch.long)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {idx}: {e}")
            print("BERT Input:", row['BERT Input'])
            print("BERT Label:", row['BERT Label'])
            print('El error puede deberse a los apóstrofes y/o comillas. Considere preparar los datos como índices, no como tokens.')
            # Handle the error, e.g., by skipping this row or using default values
            return None  # or some default values
        
        return bert_input, bert_label, segment_label, is_next  # Include original_text if needed
