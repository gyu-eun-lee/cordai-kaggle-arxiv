import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchtext

from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class.
    Args:
        :data: type dict
            - :key 'text': type torch.tensor(dtype=int): vectorized text data
            - :key 'labels': type torch.tensor(dtype=int): multi-hot encoded label data
    Attributes:
        :data: type dict
            - :key 'text': type torch.tensor(dtype=int): vectorized text data
            - :key 'labels': type torch.tensor(dtype=int): multi-hot encoded label data
    """
    def __init__(self, data):
        # store encodings internally
        self.data = data

    def __len__(self):
        # return the number of samples
        return self.data['text'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.data.items()}

class arXivDataModule(pl.LightningDataModule):
    """
    DataModule for the arXiv dataset.
    Args:
        :config: type arXivDataConfig
        :data: type dict
            :key 'train': type tuple
                - training data of form (X,y)
            :key 'val': type tuple
                - validation data of form (X,y)
            :key 'test': type tuple
                - test data of form (X,y)
            :key 'predict': type pd.Series
                - prediction data of form X
            - X == pd.Series of text data, y == pd.DataFrame of class labels
    Attributes:
        :data: type dict
            - :key 'train': training data of form (X,y)
            - :key 'val': validation data of form (X,y)
            - :key 'test': test data of form (X,y)
            - :key 'predict': prediction data of form X
            - X == pd.Series of text data, y == pd.DataFrame of class labels
        :batch_size: type int
        :n_cores: type int
            - number of CPU cores for dataloader
        :vocab: type torchtext.vocab.Vocab
            - vocabulary for word embedding, generated from training data
        :vocab_length: type int
            - vocabulary size
        :max_length: type int
            - dimension of tokenized strings
            - longer strings will be truncated to max_length
            - shorter strings will be padded to max_length
    Methods:
        :get_vocab:
            - creates vocabulary from training data
            - called automatically during setup
        :vectorize_text_data:
            - utility method, convert text data to fixed-length tensors for dataloading
    """
    def __init__(
        self,
        config,
        data
    ):
        super().__init__()
        self.data = data
        self.batch_size = config['batch_size']
        self.n_cores = config['n_cores']
        self.vocab = None
        self.vocab_length = config['vocab_length']
        self.max_length = config['max_length']

    def setup(self, stage: str):
        # Assign datasets for use in dataloaders
        # Validation data is optional in fit stage
        if stage == "fit":
            try:
                self.train_data = self.data['train']
                self.vocab = self.get_vocab(self.data['train'])
            except KeyError:
                raise KeyError('Missing train data')
            try:
                self.val_data = self.data['val']
                self.vocab = self.get_vocab(self.data['train'])
            except KeyError:
                pass
        if stage == "test":
            try:
                self.test_data = self.data['test']
                self.vocab = self.get_vocab(self.data['train'])
            except KeyError:
                raise KeyError('Missing test data')
        if stage == "predict":
            try:
                self.predict_data = self.data['predict']
                self.vocab = self.get_vocab(self.data['train'])
            except KeyError:
                raise KeyError('Missing predict data')
        
    def get_vocab(self, data):
        """
        Create vocabulary from training data.
        """
        X, _ = data
        vectorizer = TfidfVectorizer(
            sublinear_tf = True,
            strip_accents ='unicode',
            stop_words = 'english',
            analyzer ='word',
            token_pattern=r'\w{1,}',
            ngram_range = (1,1),
            max_features = self.vocab_length
        )
        vectorizer.fit(X)
        vocab = vectorizer.vocabulary_
        vocab = torchtext.vocab.vocab(vocab, specials = ['<UNK>', '<PAD>'])
        vocab = torchtext.vocab.Vocab(vocab)
        vocab.set_default_index(vocab['<UNK>'])
        return vocab
        
    
    def pad_truncate(self, tokens):
        if len(tokens) < self.max_length:
            return tokens + (['<PAD>'] * (self.max_length - len(tokens)))
        else:
            return tokens[:self.max_length]

    def vectorize_text_data(self, text_array):
        tokenizer = torchtext.data.get_tokenizer('basic_english')
        text_array = text_array.apply(lambda text: tokenizer(text))
        text_array = text_array.apply(lambda tokens: self.pad_truncate(tokens))
        text_array = text_array.apply(lambda tokens: np.array(self.vocab(tokens), dtype=int))
        text_array = pd.DataFrame(text_array.values.tolist(), index = text_array.index)
        text_array = text_array.to_numpy(dtype=int)
        return torch.tensor(text_array, dtype=torch.int)
        
    def train_dataloader(self):
        X, y = self.train_data
        return DataLoader(
            Dataset({ 
                    'text': self.vectorize_text_data(X),
                    'labels': torch.tensor(y.to_numpy(dtype=int))
                }),
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.n_cores,
            pin_memory = True
        )

    def val_dataloader(self):
        X, y = self.val_data
        return DataLoader(
            Dataset({
                'text': self.vectorize_text_data(X),
                'labels': torch.tensor(y.to_numpy(dtype=int))
            }),
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_cores,
            pin_memory = True
        )
    
    def test_dataloader(self):
        X, y = self.test_data
        return DataLoader(
            Dataset({ 
                'text': self.vectorize_text_data(X),
                'labels': torch.tensor(y.to_numpy(dtype=int))
            }),
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_cores,
            pin_memory = True
        )
    
    def predict_dataloader(self):
        X = self.predict_data
        return DataLoader(
            Dataset({
                'text': self.vectorize_text_data(X)
            }),
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_cores,
            pin_memory = True
        )

class EmbeddingClassifier(nn.Module):
    """
    A simple word embedding + convolutional network.
    Args:
        :config: type dict
            :key 'vocab_length': type int
                - length of vocabulary
            :key 'embedding_dim': type int
                - embedding dimension
            :key 'dropout': type float
                - dropout ratio
            :key 'max_length': type int
                - length of input text sequences
            :key 'hidden_dim': type int
                - linear layer dimension
            :key 'num_labels': type int
                - number of classes
    Attributes:
        :vocab_length: type int
            -length of vocabulary
        :embedding_dim: type int
            - embedding dimension
        :dropout: type float
            - dropout ratio
        :max_length: type int
            - length of input text sequences
        :hidden_dim: type int
            - linear layer dimension
        :num_labels: type int
            - number of classes
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_length = config['vocab_length']
        self.embedding_dim = config['embedding_dim']
        self.dropout = config['dropout']
        self.max_length = config['max_length']
        self.hidden_dim = config['hidden_dim']
        self.num_labels = config['num_labels']

        self.embed = nn.Sequential(
            nn.Embedding(
                self.vocab_length + 2, # +2 for special tokens
                self.embedding_dim,
                padding_idx = 1 # This value may change if you add more special tokens to the vocab
            )
        )
        # tokens with index == padding_idx will embed to zero vector and remain static during training
        # embedding.shape == (max_length,embedding_dim)
        self.conv_1gram = nn.Sequential(
            nn.Conv1d(
                in_channels = self.embedding_dim,
                out_channels = self.embedding_dim,
                kernel_size = 1,
                padding = 0,
                stride = 1,
                bias = False
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.conv_3gram = nn.Sequential(
            nn.Conv1d(
                in_channels = self.embedding_dim,
                out_channels = self.embedding_dim,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                bias = False
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.conv_5gram = nn.Sequential(
            nn.Conv1d(
                in_channels = self.embedding_dim,
                out_channels = self.embedding_dim,
                kernel_size = 5,
                padding = 2,
                stride = 1,
                bias = False
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.conv_pool = nn.Sequential(
            nn.Conv1d(
                in_channels = self.embedding_dim,
                out_channels = self.embedding_dim,
                kernel_size = 2,
                padding = 0,
                stride = 2,
                bias = False
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.flatten = nn.Flatten()
        # shape after self.flatten == (3 * max_length * embedding_dim // 2**(number of pooling layers),)
        self.classifier = nn.Sequential(
            nn.Linear(
                3 * self.max_length * self.embedding_dim // 2,
                self.hidden_dim,
                bias = False
            ),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_labels)
        )
    def forward(self, vectorized_text):
        out = self.embed(vectorized_text)
        out = out.transpose(1,2).contiguous()

        out_1gram = self.conv_1gram(out) + out
        out_1gram = self.conv_pool(out_1gram)

        out_3gram = self.conv_3gram(out) + out
        out_3gram = self.conv_pool(out_3gram)

        out_5gram = self.conv_5gram(out) + out
        out_5gram = self.conv_pool(out_5gram)

        out = torch.cat((out_1gram, out_3gram, out_5gram), dim = 1)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

class PLEmbeddingClassifier(pl.LightningModule):
    """
    LightningModule implementation for EmbeddingClassifier
    Args:
        :config: type dict
            :key 'num_labels': type int
                - number of classes
            :key 'pos_weight': type torch.tensor(dtype=float)
                - class weights for binary cross-entropy loss
            :key 'max_lr': type float
                - maximum learning rate for OneCycleLR
            :key 'pct_start': type float
                - start percentage for OneCycleLR
            :key 'weight_decay': type float
                - weight decay for AdamW
    Attributes:
        :num_labels: type int
            - number of classes
        :pos_weight: type torch.tensor(dtype=float)
            - class weights for binary cross-entropy loss
        :lr: type float
            - AdamW learning rate, for internal logging use
            - overridden by OneCycleLR learning rate
        :max_lr: type float
            - maximum learning rate for OneCycleLR
        :pct_start: type float
            - start percentage for OneCycleLR
        :weight_decay: type float
            - weight decay for AdamW
        :classifier: type EmbeddingClassifier(nn.Module)
            - PyTorch module that runs the forward pass
        :loss_fn
            - loss function
        :MCC, AUROC, AVP, FHalf, FOne, FTwo, Accuracy
            - various accuracy metrics
    Methods:
        :_get_preds_loss_accuracy
            - utility function
            - runs forward pass, computes logits, loss, accuracy metrics
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config['num_labels']
        self.pos_weight = config['pos_weight']
        # Optimizer parameters
        self.lr = 1e-3
        self.max_lr = config['max_lr']
        self.pct_start = config['pct_start']
        self.weight_decay = config['weight_decay']
        # Model layers
        self.classifier = EmbeddingClassifier(config)
        # loss
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)
        # Accuracy metrics
        self.MCC = torchmetrics.MatthewsCorrCoef(
            task = 'multilabel',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.AUROC = torchmetrics.AUROC(
            task = 'multilabel',
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.AVP = torchmetrics.AveragePrecision(
            task = 'multilabel',
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.FHalf = torchmetrics.FBetaScore(
            task = 'multilabel',
            num_beta = 0.5,
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.FOne = torchmetrics.FBetaScore(
            task = 'multilabel',
            num_beta = 1.0,
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.FTwo = torchmetrics.FBetaScore(
            task = 'multilabel',
            num_beta = 2.0,
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )
        self.Accuracy = torchmetrics.FBetaScore(
            task = 'multilabel',
            average = 'weighted',
            num_labels = self.num_labels,
            validate_args = False
        )

        # Save hyperparameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters(config)
        
    def forward(self, vectorized_text):
        logits = self.classifier(vectorized_text)
        return logits
    
    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, mcc, auroc, avp, f_half, f_one, f_two, acc = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_mcc', mcc, sync_dist=True)
        self.log('train_auroc', auroc, sync_dist=True)
        self.log('train_avp', avp, sync_dist=True)
        self.log('train_f_half', f_half, sync_dist=True)
        self.log('train_f_one', f_one, sync_dist=True)
        self.log('train_f_two', f_two, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, mcc, auroc, avp, f_half, f_one, f_two, acc = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_mcc', mcc, sync_dist=True)
        self.log('val_auroc', auroc, sync_dist=True)
        self.log('val_avp', avp, sync_dist=True)
        self.log('val_f_half', f_half, sync_dist=True)
        self.log('val_f_one', f_one, sync_dist=True)
        self.log('val_f_two', f_two, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, mcc, auroc, avp, f_half, f_one, f_two, acc = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_mcc', mcc, sync_dist=True)
        self.log('test_auroc', auroc, sync_dist=True)
        self.log('test_avp', avp, sync_dist=True)
        self.log('test_f_half', f_half, sync_dist=True)
        self.log('test_f_one', f_one, sync_dist=True)
        self.log('test_f_two', f_two, sync_dist=True)
        self.log('test_acc', acc, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        '''used to generate predictions'''
        vectorized_text = batch['text']
        logits = self(vectorized_text)
        preds = torch.sigmoid(logits)
        return preds
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr = self.lr, 
            weight_decay = self.weight_decay
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr = self.max_lr,
                    total_steps = self.trainer.estimated_stepping_batches,
                    anneal_strategy = 'cos',
                    cycle_momentum = False,
                    pct_start = self.pct_start
                )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
        
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        vectorized_text = batch['text']
        labels = batch['labels']
        logits = self(vectorized_text)
        loss = self.loss_fn(logits, labels.float())
        preds = torch.round(torch.sigmoid(logits)).int()
        mcc = self.MCC(logits, labels)
        auroc = self.AUROC(logits, labels)
        avp = self.AVP(logits, labels)
        f_half = self.FHalf(logits, labels)
        f_one = self.FOne(logits, labels)
        f_two = self.FTwo(logits, labels)
        acc = self.Accuracy(logits, labels)
        return preds, loss, mcc, auroc, avp, f_half, f_one, f_two, acc
