import torch
import torch.nn as nn

# The decoder:
#   >> is a unidirectional LSTM that accepts the above 300-D vector as hidden state. 
#   >> Each cell accepts the GloVe embedding of a word in the caption sentence as input
#   >> Each cell outputs a softmax probability distribution vector of the size of the dataset's vocabulary.

class DecoderRNN(nn.Module):
    """
    generates the caption words
    """

    def __init__(self, vocab_size, embed_size=300, hidden_size=512, num_layers=1):
        """
        instantiates and initializes an object of DecoderRNN
        :param vocab_size: the number of words in the vocabulary
        :param embed_size: the size of embedding for each word in Glove embeddings
        :param hidden_size: size of the hidden layer in the LSTM
        :param num_layers: number of layers in the LSTM
        """
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        :param features: the image's pixels and labels concatenation passed through the CNN. Type: torch Tensor
        :param captions: the captions for the image witch each word encoded as its Glove embedding. Type: torch Tensor
        :return: a torch tensor, representing likelihood of the next word in caption, as a probability distribution
                 over all the words in the vocabulary. Type: torch Tensor
        """
        captions = captions[:, :-1]
        embed = torch.cat((features.unsqueeze(1), captions), dim=1)
        # print("embed:", embed.shape)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)
        # print("out:", out.shape)
        return out

    def generate_caption(self, inputs, states=None, max_len=34):
        """
        computes the predicted caption
        :param inputs:
        :param states:
        :param max_len:
        :return:
        """
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)
        return output_sentence
        