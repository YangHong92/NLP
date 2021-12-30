import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from .sentiment_classifier import SentimentClassifier

with open("config.json") as f:
    config = json.load(f)

class Model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config['PRE_TRAINED_MODEL_NAME'])

        model = SentimentClassifier(len(config['CLASS_NAMES']))
        model.load_state_dict(torch.load(
            config['MODEL_STATE_PATH'], map_location=self.device
        ))
        # set model in evaluation mode as we use it for inference
        model = model.eval()

        self.classifier = model.to(self.device)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(text,
                                              add_special_tokens = True,
                                              max_length = config['MAX_SEQUENCE_LEN'],
                                              return_token_type_ids = False,
                                              truncation = True,
                                              padding = 'max_length',
                                              return_attention_mask=True,
                                              return_tensors = 'pt')
        
        input_ids = encoding['input_ids'].to(self.device) # data is one batch of samples
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            probabilities = probabilities.flatten().cpu().numpy().tolist()
            predicted_class = predicted_class.cpu().item()
            sentiment = config['CLASS_NAMES'][predicted_class]

        return (
            dict(zip(config['CLASS_NAMES'], probabilities)),
            sentiment,
            confidence
        )


model = Model()

# return the single instance of the inference model
def get_model():
    return model