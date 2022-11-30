################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    max_len = config_data['generation']['max_length']
    encoder = Encoder(embedding_size)
    type_mapper = {'LSTM': 0, "Vanilla": 1}
    decoder = Decoder(embedding_size, hidden_size, len(vocab), 2, max_len, type_mapper[model_type])
    model = nn.ModuleList()
    model.append(encoder)
    model.append(decoder)
    return model


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, max_seq_leng=20, model_type=0):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_leng = max_seq_leng
        self.model_type = model_type

        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)

        if self.model_type == 0:
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size,self.num_layers, batch_first=True)
        if self.model_type == 1:
            self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions, lengths):

        caption_embed = self.word_embedding(captions[:, :-1])
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed), 1)
        packed = pack_padded_sequence(caption_embed, lengths, batch_first=True)
        hiddens, _ = self.rnn(packed)

        outputs = self.fc(hiddens[0])
        return outputs

    def sample(self, features, states=None, deterministic=False, temperature=0.1):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_seq_leng):
            hiddens, states = self.rnn(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            if deterministic:
                _, predicted = outputs.max(1)                        # predicted: (batch_size)
            else:
                prob = F.softmax(outputs / temperature, dim=1)
                predicted = torch.tensor([torch.multinomial(prob[x], 1) for x in range(prob.shape[0])]).to(inputs.device)
            sampled_ids.append(predicted)
            inputs = self.word_embedding(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids