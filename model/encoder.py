import torch
import torch.nn as nn

# The encoder:
#   >> Uses pretrained VGG16 to embed images into 300-d vector
#   >> Uses Glove vectors for embedding words into 300-d vector
#   >> Concatenates image's embedding with corresponding label words' average embedding. This forms a 600-d vector. 
#   >> Above 600-d vector is fed into a single layer nn (600, 450, 300) that outputs a 300-d vector to be fed as initial state to the decoder LSTM. 

class EncoderNN(nn.Module):
    """
    This is the fully connected layer of the encoder in the reference Perison and Tolunay, 2018
    """

    def __init__(self, input_size=4096, hidden_size=300, output_size=300):
        """
        instantiates and initializes an object of type Network
        :param input_size: the size of the vector that contains the image pixels
        :param hidden_size: the size of the hidden layer of the neural network
        :param output_size: the size of the output vector of the neural network
        :return an object of Network class
        """
        super(EncoderNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Inputs to hidden layer linear transformation of size 600
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(2 * hidden_size, output_size)

    def forward(self, img_pixels_tensor, img_label_tensor):
        """
        takes the image's pixels values returned by VGG and the image's label embeddings, concatenates them and uses
        this concatenation to generate one tensor encoding information for both
        :param img_pixels_tensor:
        :param img_label_tensor:
        :return: a torch tensor of size output_size (set to 300 by default)
        """
        # Pass the input tensor through each of our operations
        img_pixels = self.hidden(img_pixels_tensor)  # maps the 4096 sized image pixel tensor to a tensor of 300
        x = torch.cat((img_pixels.float(), img_label_tensor.float()), axis=1)  # concatenate pixel and label
        x = self.output(x)
        return x