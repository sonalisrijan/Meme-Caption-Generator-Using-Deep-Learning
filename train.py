import argparse
from gensim.models import Word2Vec
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import re
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from model.encoder import EncoderNN
from model. decoder import DecoderRNN

np.random.seed(100)
torch.manual_seed(99)

max_sentence_length = 34  # maximum number of words in a caption

def write_dictionaries(captions_file, label2captions_file, word2ind_file):
    """
    creates two dictionaries and writes them to files
    :param captions_file: file containing the captions for all images. Type: String
    :param label2captions_file: file the dictionary labels2captions is supposed to be written to. Type: String
    :param word2ind_file: file the dictionary word2idx is supposed to be written to. Type: String
    :return: None. Just writes the dictionaries to the files passed
    """
    label2captions, word2idx = prepare_data(captions_file)
    ofile = open(label2captions_file, "wb")
    pickle.dump(label2captions, ofile)
    ofile.close()
    ofile = open(word2ind_file, "wb")
    pickle.dump(word2idx, ofile)
    ofile.close()


def read_dictionaries(label2captions_file, word2ind_file):
    """
    reads the labels2captions and word2ind dictionaries from files
    :param label2captions_file: file containing labels2captions dictionary. Type: String
    :param word2ind_file: file containing word2ind dictionary. Type: String
    :return: the labels2captions dictionary, and word2ind dictionary
    """
    ifile = open(label2captions_file, "rb")
    label2captions = pickle.load(ifile)
    ifile.close()
    ifile = open(word2ind_file, "rb")
    word2ind = pickle.load(ifile)
    ifile.close()
    return label2captions, word2ind


####################################################################################################################
# ENCODER FUNCTIONS
####################################################################################################################

def preprocess(image):  # Preprocsessing image data
    """
    takes an image and normalizes, horizontal flip, resized cropping, and converts the results to a torch Tensor
    :param image: the input image. Type: PIL.JpegImagePlugin.JpegImageFile
    :return: a torch Tensor containing the preporcessed image file
    """
    m1, m2, m3 = 0.485, 0.456, 0.406  # these are the three means of each of the three RGB channels for VGG
    std1, std2, std3 = 0.229, 0.224, 0.225  # these are the standard devs of the three RGB channels for VGG
    num_pixels = 224  # the images are stored
    preprocessing = transforms.Compose([
        transforms.RandomResizedCrop(num_pixels),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([m1, m2, m3], [std1, std2, std3])
    ])
    return preprocessing(image)


def image2tensor(image_path, embeddings, output_size=300):
    """
    processes an .jpg file, along with its label to produce one torch Tensor of size 2 * output_size
    :param embeddings: word embeddings. Type: Dictionary. E.g. {'the': [0.1,0.2,...,0.005]}
    :param output_size: this is the output size, of the word embeddings used. Type: int
    :param image_path: the path to the image. Type: String
    :return: [1-by-output_size] torch Tensor
    """
    im = Image.open(image_path)
    prepped_img = preprocess(im)
    prepped_img = torch.unsqueeze(prepped_img, 0)
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(4)])
    # Initially, we freeze all of the model's weights:
    for param in vgg_model.parameters():
        param.requires_grad = False  # this model is pre-trained, and we do not want to train it any further
    img_pixels_tensor = vgg_model(prepped_img)
    # every image_label is of the form 'hello-world-good.jpg'. Need to get the label
    # and remove the .jpg at the end, and get the label words. The image label is included in the image path, of the
    # form /Users/<this_user>/.../hello-world-good.jpg
    image_label = image_path.split('/')[-1]
    num_unnecessary_chars = 4
    label_words = (image_label[0:len(image_label) - num_unnecessary_chars]).split('-')  # individual words in the label
    img_label_vec = np.zeros(output_size)
    for word in label_words:
        img_label_vec += embeddings[word]
    img_label_vec /= len(label_words)
    img_label_vec = torch.from_numpy(img_label_vec)
    img_label_tensor = torch.unsqueeze(img_label_vec, 0)
    return img_pixels_tensor.float(), img_label_tensor.float()


####################################################################################################################
# DECODER FUNCTIONS
####################################################################################################################
start_token = '<START>'
end_token = '<END>'


def prepare_data(filename):
    """
    reads the captions file and returns the data in form of a dictionary
    :param filename: name of the file containing the data. Type: String
    :return: 1. test dataset in a readable format. Type: Dictionary. The keys of the dictionary are the labels of the
            image, while the values are a list of lists, where each small list contains the words for the image. E.g.
            {'y u no': [[<START>, 'meme', 'generator', 'users', 'y', 'u', 'no', 'give', 'me', 'more', 'upvotes', <END>],
            ..]..}
            2. word2id Dictionary of the form {word: int}, where the keys are words, and the values are the index
            where there should be a 1 for one-hot encodings
    """
    max_caption_size = 32  # maximum number of words in the caption
    ifile = open(filename, 'r')
    contents = ifile.readlines()
    ifile.close()
    train_data = {}
    vocabulary = []
    for line in contents:
        line = line.split(' - ')  # line[0] := label, line[1] := caption
        words = re.sub(r'[^\w\s]', '', line[1]).split()
        # label_words = line[0].split()
        label_words = re.sub(r'[^\w\s]', '', line[0]).split()
        vocabulary += words + label_words
        label = '-'.join(label_words)
        if label not in train_data:
            if len(words) <= max_caption_size:
                train_data[label] = [[start_token] + words + [end_token]]
        else:
            if len(words) <= max_caption_size:
                train_data[label].append([start_token] + words + [end_token])
    vocabulary = list(set(vocabulary))

    word2idx = {start_token: 0, end_token: 1}
    for word in vocabulary:
        word2idx[word] = len(word2idx)
    return train_data, word2idx


def read_glove_embeddings(filename):
    """
    reads the Glove embeddings stored in a file, and returns a dictionary of the form {word: <embedding>}
    :param filename: the name of the path containing the embedding data. Type: String
    :return: a Dictionary of the form {'the': <numpy vector of size 300>}
    """
    ifile = open(filename, 'r')
    contents = ifile.readlines()
    ifile.close()
    embeddings = {}
    for line in contents:
        line = line.split()
        embeddings[line[0]] = np.array([float(i) for i in line[1:]])
    return embeddings


def generate_missing_embeddings(vocabulary, embeddings):
    """
    generates embeddings for words not already found in embeddings
    :param vocabulary: all the words in the training dataset. Type: Python list
    :param embeddings: the Glove embeddings. Type: Dictionary, {'the': <300 - long numpy vector>, ..}
    :return: a Python Dictionary containing embeddings for ALL the words in the file pointed to by filename
    """
    test_word = list(embeddings.keys())[0]  # just get a word already present in the embedggings
    embeddings_size = embeddings[test_word].shape[0]  # get the size of the embeddings vector
    for word in vocabulary:
        if word not in embeddings:
            # print(word)
            embeddings[word] = np.random.rand(embeddings_size)  # generate a random embedding vector
    return embeddings


#####################################################################################################

def create_pixel_label_tensors(embeddings, imagefolder, label2captions):
    """
    :param embeddings:
    :param imagefolder:
    :param label2captions:
    :return:
    """
    images_labels = list(label2captions.keys())
    labels2num_caps = {}
    for label in images_labels:
        labels2num_caps[label] = len(label2captions[label])

    img_pixels = []
    img_labels = []
    first = True
    for label in label2captions:
        print(label)
        image_path = imagefolder + '/' + label + '.jpg'
        encodings = image2tensor(image_path, embeddings)
        if first:
            img_pixels = encodings[0].repeat(labels2num_caps[label],1)
            img_labels = encodings[1].repeat(labels2num_caps[label], 1)
            first = False
        else:
            img_pixels = torch.cat((img_pixels, encodings[0].repeat(labels2num_caps[label],1)), axis=0)
            img_labels = torch.cat((img_labels, encodings[1].repeat(labels2num_caps[label], 1)), axis=0)
    return img_pixels, img_labels


def image_labels2tensor(label2captions, imagefolder, embeddings):
    """
    creates a dictionary whose keys are the labels of the images and values are the 600-dimension tensor. 600
    is the input size of the neural network.
    :param label2captions: a dictionary whose keys are image labels, and values are a list of lists. Each list contains
                      all the captions for the image
    :param imagefolder: path the folder containing images. Type: Python String
    :param embeddings: a dictionary whose keys are words, and values are the 300 size embedding for the word
    :return a dictionary with image labels as keys, and its label's embeddings concatenated with its pixel values
            to generate a tensor of size 600*.
    """
    labels2tensors = {}
    for label in label2captions:
        imagepath = imagefolder + '/' + label + '.jpg'
        labels2tensors[label] = image2tensor(imagepath, embeddings)
    return labels2tensors


#####################################################################################################


def create_and_save_tensors(imagesfolder, label2captions, word2ind, embeddings,
                            pixel_file, label_file, captions_file, y_file):
    """
    creates pytorch Tensors for the inputs and outputs used later for training/validation/testing, and
    stores them on disk to be read later directly.
    :param imagesfolder: folder containing all the images (.jpg files). Type: String
    :param label2captions: contains mappings from image labels to image captions. Type: Dictionary of format
                           {String: list of lists}
    :param word2ind: maps each word to an index which is the index of the one in its one-hot encoded vector.
                    Type: Dictionary of format {String: int}
    :param embeddings: maps each word to its embedding. Type: Dictionary of format {String: numpy vector}
    :param pixel_file: name of the file to store image's pixels as torch Tensors. Type: String
    :param label_file: name of the file to store image's labels as torch Tensors. Tyep: String
    :param captions_file: name of the file to store all the captions' words stored as torch Tensors. Type: String
    :param y_file: name of the file to store all words in all captions stored as torch Tensors of indices
                   from word2ind dictionary. Type: String
    :return: None. Just creats and stores the tensors
    """
    # get the images and labels encoding
    img_pixels, img_labels = create_pixel_label_tensors(embeddings, imagesfolder, label2captions)
    torch.save(img_pixels, pixel_file)
    torch.save(img_labels, label_file)
    embeddings_size = 300  # the size of embedding vector for each word
    caps = []  # to store the captions with each word converted to its embeddings
    y = []
    print("######creating y#####")
    for label in label2captions:
        print(label)
        for caption in label2captions[label]:
            emb = []  # to get all the embeddings
            ind_emb = []  # to generate the one-hot vector embeddings for each word
            for word in caption:
                emb.append(embeddings[word])
                ind_emb.append(word2ind[word])
            if len(caption) < max_sentence_length:  # every caption is padded with zeros to make them of same length
                diff = max_sentence_length - len(caption)
                for i in range(diff):
                    emb.append(np.zeros(embeddings_size))
                    ind_emb.append(0)
            caps.append(emb)
            y.append(ind_emb)
    all_captions = torch.Tensor(caps)  # all words in all captions stored as their embeddings
    torch.save(all_captions, captions_file)
    y = torch.Tensor(y)  # all words in all captions stored as their indices. To be used in cross_entropy
    torch.save(y, y_file)


def train(n_vocab, pixel_file, label_file, captions_file, y_file):
    """
    training function for the encoder and decoder modules
    :param n_vocab: number of words in the vocabulary. Type: int
    :param label_file: name of the file which has the image's pixel representation saved as torch Tensors. Type: String
    :param pixel_file: name of the file containing image's label's representation as torch Tensors. Type: String
    :param captions_file: name of the file that has all the captions' words stored as torch Tensors. Type: String
    :param y_file: name of the file containing all words in all captions stored as torch Tensors of indices
                   from word2ind dictionary. Type: String
    :return: trained encoder and decoder modules. Type: EncoderNN and DecoderRNN
    """
    encoder = EncoderNN()
    decoder = DecoderRNN(n_vocab)
    pixels = torch.load(pixel_file)
    labels = torch.load(label_file)
    embeddings_size = 300  # the size of embedding vector for each word
    all_captions = torch.load(captions_file)
    y = torch.load(y_file)
    # there is a one-to-one correspondence between a row of x and a row of image_label_encodings
    batch_size = 512
    num_epochs = 3
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    e_losses = []
    max_iterations = 30
    ofile = open('results_full_lr_0.005.txt', 'w')
    for epoch in range(num_epochs):
        decoder.train()
        encoder.train()
        losses = []
        # num_iter = 0
        for beg_i in range(0, pixels.shape[0], batch_size):
            # num_iter += 1
            pixels_batch = Variable(pixels[beg_i:beg_i + batch_size, :])
            labels_batch = Variable(labels[beg_i:beg_i + batch_size, :])
            y_batch = Variable(all_captions[beg_i:beg_i + batch_size, :])

            one_hot = y[beg_i:beg_i + batch_size, :]
            one_hot = Variable(torch.tensor(one_hot, dtype=torch.long))

            encoder.zero_grad()
            decoder.zero_grad()

            features = encoder(pixels_batch, labels_batch)  # encode the images pixels and labels representation
            y_hat = decoder(features, y_batch)

            loss = criterion(y_hat.view(-1, n_vocab), one_hot.view(-1))
            loss.backward()
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
                epoch, num_epochs, loss.item(), np.exp(loss.item()))
            # Print training statistics (on different line).
            print('\r' + stats)
            ofile.write(stats + '\n')
            # if num_iter == max_iterations:
            #     break
    ofile.close()
    torch.save(encoder.state_dict(), 'encoder.pkl')
    torch.save(decoder.state_dict(), 'decoder.pkl')
    return encoder, decoder


def predict_caption(image, encoder, decoder, word2idx, embeddings):
    """
    takes an image, trained encoder and decoder model, and predicts the caption for the image
    :param image: the image path. Type: String
    :param encoder: the trained encoder. Type: EncoderNN
    :param decoder: the trained decoder. Type: DecoderRNN
    :param word2idx: mapping from words to indices. Type: Dictionary
    :param embeddings: the mapping from words to their (Glove) embeddings. Type: Dictionary
    :return: the computed caption
    """
    encoder.eval()
    decoder.eval()
    pixels, labels = image2tensor(image, embeddings)
    features = (encoder(pixels, labels)).unsqueeze(1)
    output = decoder.generate_caption(features)
    words_seq = []
    words = list(word2idx.keys())
    for i in output:
        if i == word2idx[end_token] or i == word2idx[start_token]:
            continue
        words_seq.append(words[i])
    return ' '.join(words_seq)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_file', type=str, default='CaptionsClean.txt',
                        help='path to file containing the captions')
    parser.add_argument('--memes_folder', type=str, default='memes', help='the folder containing all meme (jpg) files')
    args = parser.parse_args()

    captions_file = args.captions_file
    images_folder = args.memes_folder

    write_dictionaries(captions_file, 'label2captions.pkl', 'word2idx.pkl')
    label2captions, word2ind = read_dictionaries("label2captions.pkl", "word2idx.pkl")
    embeddings = read_glove_embeddings('glove.6B.300d.txt')
    embeddings = generate_missing_embeddings(list(word2ind.keys()), embeddings)
    np.save('all_embeddings', embeddings)
    embeddings = np.load('all_embeddings.npy', allow_pickle=True).item()
    create_and_save_tensors(images_folder, label2captions, word2ind, embeddings, 'pixels.pt',
                            'labels.pt', 'captions.pt', 'y.pt')

    #Train the meme genetor model
    train(len(word2ind), 'pixels.pt', 'labels.pt','captions.pt', 'y.pt')
    



