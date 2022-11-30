################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import time
import nltk
import caption_utils
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import torch.optim as optim
import torch.nn as nn
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence
#from vocab import Vocabulary


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.__lr = config_data['experiment']['learning_rate']
        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__deterministic = config_data['generation']['deterministic']
        self.__temperature = config_data['generation']['temperature']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Init Model
        self.__encoder, self.__decoder = get_model(config_data, self.__vocab)

        self.__criterion = nn.CrossEntropyLoss()

        self.__encoder_optimizer = optim.Adam(params=self.__encoder.parameters(),
                                                    lr=self.__lr)
        self.__decoder_optimizer = optim.Adam(params=self.__decoder.parameters(),
                                                    lr=self.__lr)
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            encoder_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_encoder.pt'))
            self.__encoder.load_state_dict(encoder_dict['model'])
            self.__encoder_optimizer.load_state_dict(encoder_dict['optimizer'])
            encoder_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_decoder.pt'))
            self.__decoder.load_state_dict(encoder_dict['model'])
            self.__decoder_optimizer.load_state_dict(encoder_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__encoder = self.__encoder.cuda().float()
            self.__decoder = self.__decoder.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print("start epoch:", epoch)
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        self.__encoder.train()
        self.__decoder.train()
        training_loss = 0

        for i, (_, images, captions, img_ids, lengths) in enumerate(self.__train_loader):

            # Move to GPU, if available
            images = images.to(self.device)
            captions = captions.to(self.device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            self.__decoder_optimizer.zero_grad()
            self.__encoder_optimizer.zero_grad()

            features = self.__encoder(images)

            outputs = self.__decoder(features, captions, lengths)

            loss = self.__criterion(outputs, targets)
            loss.backward()

            self.__decoder_optimizer.step()
            self.__encoder_optimizer.step()

            training_loss += loss.item()

        return training_loss / len(self.__train_loader)


    def __val(self):
        self.__decoder.eval()
        self.__encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (_, images, captions, img_ids, lengths) in enumerate(self.__val_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                features = self.__encoder(images)
                outputs = self.__decoder(features, captions, lengths)
                loss = self.__criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(self.__val_loader)


    def test(self):
        self.__encoder.eval()
        self.__decoder.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0

        with torch.no_grad():
            for iter, (orig_image, images, captions, img_ids, lengths) in enumerate(self.__test_loader):

                images, captions = images.to(self.device), captions.to(self.device)

                features = self.__encoder(images)
                outputs = self.__decoder(features, captions, lengths)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                loss = self.__criterion(outputs, targets)
                test_loss += loss.item()
                
                gt_cat_ids = self.__coco_test.getAnnIds(img_ids[0])
                reference = self.__coco_test.loadAnns(gt_cat_ids)
                reference = [o['caption'] for o in reference]

                #print(sentence)
                bleu_ref = []
                for ref in reference:
                    bleu_ref.append(nltk.tokenize.word_tokenize(ref))

                # Generate a caption from the image
                sampled_ids = self.__decoder.sample(features, deterministic=self.__deterministic, temperature=self.__temperature)
                sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

                # Convert word_ids to words
                sampled_caption = []
                for word_id in sampled_ids:
                    word = self.__test_loader.dataset.vocab.idx2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break
                sampled_caption = sampled_caption[1:-1]
                sentence = ' '.join(sampled_caption)
                predicted = nltk.tokenize.word_tokenize(sentence)
                b1 = caption_utils.bleu1(bleu_ref, predicted)

                bleu1 += caption_utils.bleu1(bleu_ref, predicted)
                bleu4 += caption_utils.bleu4(bleu_ref, predicted)
                
                if b1 > 90:
                    print("With temperature default = 0.4:")
                    self.get_good_prediction(orig_image[0], reference, sentence, b1)
                    
                    sampled_ids_1 = self.__decoder.sample(features, deterministic=self.__deterministic, temperature=0.001)
                    sampled_ids_1 = sampled_ids_1[0].cpu().numpy()
                    sentence1, b1_1 = self.generate_captions(sampled_ids_1, bleu_ref)
                    print("With temperature = 0.001")
                    self.get_good_prediction(orig_image[0], reference, sentence1, b1_1)
                    
                    sampled_ids_2 = self.__decoder.sample(features, deterministic=self.__deterministic, temperature=5)
                    sampled_ids_2 = sampled_ids_2[0].cpu().numpy()
                    sentence2, b1_2 = self.generate_captions(sampled_ids_2, bleu_ref)
                    print("With temperature = 5")
                    self.get_good_prediction(orig_image[0], reference, sentence2, b1_2)
                    
                    sampled_ids_3 = self.__decoder.sample(features, deterministic=True, temperature=self.__temperature)
                    sampled_ids_3 = sampled_ids_3[0].cpu().numpy()
                    sentence3, b1_3 = self.generate_captions(sampled_ids_3, bleu_ref)
                    print("With temperature deterministic")
                    self.get_good_prediction(orig_image[0], reference, sentence3, b1_3)
                    
                if b1 < 40:
                    print("With temperature default = 0.4:")
                    self.get_bad_prediction(orig_image[0], reference, sentence, b1)
                    
                    sampled_ids_1 = self.__decoder.sample(features, deterministic=self.__deterministic, temperature=0.001)
                    sampled_ids_1 = sampled_ids_1[0].cpu().numpy()
                    sentence1, b1_1 = self.generate_captions(sampled_ids_1, bleu_ref)
                    print("With temperature = 0.001")
                    self.get_bad_prediction(orig_image[0], reference, sentence1, b1_1)
                    
                    sampled_ids_2 = self.__decoder.sample(features, deterministic=self.__deterministic, temperature=5)
                    sampled_ids_2 = sampled_ids_2[0].cpu().numpy()
                    sentence2, b1_2 = self.generate_captions(sampled_ids_2, bleu_ref)
                    print("With temperature = 5")
                    self.get_bad_prediction(orig_image[0], reference, sentence2, b1_2)
                    
                    sampled_ids_3 = self.__decoder.sample(features, deterministic=True, temperature=self.__temperature)
                    sampled_ids_3 = sampled_ids_3[0].cpu().numpy()
                    sentence3, b1_3 = self.generate_captions(sampled_ids_3, bleu_ref)
                    print("With temperature deterministic")
                    self.get_bad_prediction(orig_image[0], reference, sentence3, b1_3)

            bleu1 /= iter
            bleu4 /= iter
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss / len(self.__test_loader),
                                                                                               bleu1,
                                                                                               bleu4)
        self.__log(result_str)

        return test_loss / len(self.__test_loader), bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_encoder.pt')
        encoder_dict = self.__encoder.state_dict()
        encoder_dict = {'model': encoder_dict, 'optimizer': self.__encoder_optimizer.state_dict()}
        torch.save(encoder_dict, root_model_path)

        root_model_path = os.path.join(self.__experiment_dir, 'latest_decoder.pt')
        decoder_dict = self.__decoder.state_dict()
        decoder_dict = {'model': decoder_dict, 'optimizer': self.__decoder_optimizer.state_dict()}
        torch.save(decoder_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
        
    def generate_captions(self, sampled_ids, bleu_ref):
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.__test_loader.dataset.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sampled_caption = sampled_caption[1:-1]
        sentence = ' '.join(sampled_caption)
        predicted = nltk.tokenize.word_tokenize(sentence)
        b1 = caption_utils.bleu1(bleu_ref, predicted)
        
        return sentence, b1
        
    def get_good_prediction(self, image, reference, sentence, bleu1):
        plt.imshow(image)
        plt.title('Good Image')
        plt.show()
        print("Actual captions:", reference)
        print("Predicted sentence:", sentence)
        print("bleu1 score:", bleu1)
        
    def get_bad_prediction(self, image, reference, sentence, bleu1):
        plt.imshow(image)
        plt.title('Bad Image')
        plt.show()
        print("Actual captions:", reference)
        print("Predicted sentence:", sentence)
        print("bleu1 score:", bleu1)