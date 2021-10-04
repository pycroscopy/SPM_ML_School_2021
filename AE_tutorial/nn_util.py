import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import tensorflow as tf
import torch
from tqdm import tqdm
import numpy as np

device = 'cuda'


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model
        :param weight_decay: coeifficient of
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        '''
        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model: model
        :return: list of layers needs to be regularized
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'dec' in name and 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        '''
        :param weight_list: list of layers needs to be regularized
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        :param weight_decay: coeifficient
        :return: loss
        '''
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        :param weight_list:
        :return: list of layers' name needs to be regularized
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)


def loss_function(model,
                  encoder,
                  decoder,
                  train_iterator,
                  optimizer,
                  coef=0,
                  coef1=0,
                  ln_parm=1,
                  beta=None,
                  mse = True):
    '''

    :param model:
    :param encoder:
    :param decoder:
    :param train_iterator:
    :param optimizer:
    :param coef:
    :param coef1:
    :param ln_parm:
    :param beta:
    :return:
    '''

    # regularization coefficents
    weight_decay = coef
    weight_decay_1 = coef1

    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0
    #    for i, x in enumerate(train_iterator):
    for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

        # calculates regularization on the entire model
        reg_loss_2 = Regularization(model, weight_decay_1, p=2).to(device)

        x = x.to(device, dtype=torch.float)

        # update the gradients to zero
        optimizer.zero_grad()

        if beta is None:
            embedding = encoder(x)

        else:

            # forward pass
            #        predicted_x = model(x)
            embedding, sd, mn = encoder(x)

        if weight_decay > 0:
            reg_loss_1 = weight_decay * torch.norm(embedding, ln_parm).to(device)
        else:
            reg_loss_1 = 0.0

        predicted_x = decoder(embedding)

        if mse:
            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')
        else:
            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')

            loss = loss + reg_loss_2(model) + reg_loss_1

        # beta VAE
        if beta is not None:
            vae_loss = beta * 0.5 * torch.sum(torch.exp(sd) + (mn) ** 2 - 1.0 - sd).to(device)
            vae_loss /= (sd.shape[0] * sd.shape[1])
        else:
            vae_loss = 0

        loss = loss + vae_loss

        # backward pass
        train_loss += loss.item()

        loss.backward()
        # update the weights
        optimizer.step()

    return train_loss


def Train(model, encoder, decoder, train_iterator, optimizer,
          epochs, coef=0, coef_1=0, ln_parm=1, beta=None, mse = True):
    N_EPOCHS = epochs
    best_train_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train = loss_function(model, encoder, decoder, train_iterator,
                              optimizer, coef, coef_1, ln_parm, beta, mse)

        train_loss = train
        train_loss /= len(train_iterator)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
        print('.............................')

        if best_train_loss > train_loss:
            best_train_loss = train_loss

            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
            if epoch >= 0:
                torch.save(checkpoint, f'/test__Train Loss:{train_loss:.4f}-{epoch}.pkl')


def transform_nn(data, encoder, decoder):
    try:
        encoded_spectra = encoder(torch.tensor(np.atleast_3d(data), dtype=torch.float32).to(device))
    except:
        pass

    try:
        encoded_spectra = encoder(torch.tensor(data, dtype=torch.float32).to(device))
    except:
        pass

    decoded_spectra = decoder(encoded_spectra)

    encoded_spectra = encoded_spectra.to('cpu')
    encoded_spectra = encoded_spectra.detach().numpy()
    decoded_spectra = decoded_spectra.to('cpu')
    decoded_spectra = decoded_spectra.detach().numpy()
    return encoded_spectra, decoded_spectra
