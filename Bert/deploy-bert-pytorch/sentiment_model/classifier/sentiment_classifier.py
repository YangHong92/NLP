import json
from torch import nn
from transformers import BertModel

with open("config.json") as f:
    config = json.load(f)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config['PRE_TRAINED_MODEL_NAME']) 
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    return_dict = False)

        # dropout layer for regularization
        output = self.drop(pooled_output)

        # fully-connected layer for our output, return output for following cross-entropy loss function
        # no need to do softmax, as to nn.CrossEntropyLoss, the input is expected to contain raw, unnormalized scores for each class. 
        return self.out(output)