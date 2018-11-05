import torch.nn as nn
import torch.nn.functional as F


class LanguageNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, text_length, label_size):
        super(LanguageNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(text_length, 10, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(10, 20, 3)
        self.relu2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(2)
        self.linear1 = nn.Linear(280, 140)
        self.linear2 = nn.Linear(140, label_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        conv1 = self.conv1(embeds)
        conv1_relu = self.relu1(conv1)
        conv2 = self.conv2(conv1_relu)
        conv2_relu = self.relu2(conv2)
        conv2_mp = self.max_pool1(conv2_relu)
        flat = conv2_mp.view(conv2_mp.shape[0], -1)
        l1 = self.linear1(flat)
        l2 = self.linear2(l1)
        l2_relu = self.relu1(l2)
        log_probs = F.log_softmax(l2_relu, dim=1)
        return log_probs