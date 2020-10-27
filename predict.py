import argparse
import numpy as np
import torch

from model.encoder import EncoderNN
from model. decoder import DecoderRNN
from model.nn import HilarityNN, RelevanceNN
from train import *

def main(args):

    label2captions, word2ind = read_dictionaries("label2captions.pkl", "word2idx.pkl")
    embeddings = np.load('all_embeddings.npy', allow_pickle=True).item()
    input_size = 300
    num_classes = 2

    encoder = EncoderNN()
    decoder = DecoderRNN(len(word2ind))
    encoder.load_state_dict(torch.load('encoder.pkl'))
    decoder.load_state_dict(torch.load('decoder.pkl'))
    hilarity_classifier = HilarityNN(input_size, num_classes)
    meme_relevance = RelevanceNN(2*input_size, num_classes)
    hilarity_classifier.load_state_dict(torch.load('hilarity_classifier_net'))
    meme_relevance.load_state_dict(torch.load('meme_classifier_net'))

    encoder.eval()
    decoder.eval()
    hilarity_classifier.eval()
    meme_relevance.eval()

    # get the image file (jpg)
    image = args.folder + '/' + args.image

    # get the predicted caption for the image
    caption = predict_caption(image, encoder, decoder, word2ind, embeddings)

    # get the average of the embeddings of the label of the image concatenated with its pixels
    image_pixel, image_label = image2tensor(image, embeddings)

    # get the average of all the words in the caption
    caption_vec = np.zeros(input_size)
    for word in caption.split():
        caption_vec += embeddings[word]
    caption_vec /= len(caption.split())

    caption_tensor = torch.from_numpy(caption_vec)
    caption_tensor = (torch.unsqueeze(caption_tensor, 0)).float()
    print("the predicted caption is:")
    print(caption)

    hilarity_measure = hilarity_classifier(caption_tensor)
    is_hilarious = (hilarity_measure.max(1)[1]).item()
    if is_hilarious == 0:
        print("\nthe caption is determined to be hilarious")
    else:
        print("\nthe caption is determined to be not hilarious")

    features = torch.cat((image_label, caption_tensor), dim=1)
    relevance_measure = meme_relevance(features)
    is_relevant = (relevance_measure.max(1)[1]).item()
    if is_relevant == 0:
        print("\nthe caption is predicted to be relevant")
    else:
        print("\nthe caption is predicted to be irrelevant")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='memes', help='path to folder containing the image')
    parser.add_argument('--image', type=str, default='y-u-no.jpg', help='the image (jpg) file')
    args = parser.parse_args()

    predict(args)


