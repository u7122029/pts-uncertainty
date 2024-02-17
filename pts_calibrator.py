import os
import tensorflow as tf
import numpy as np
import torch
import time
from torch.nn.functional import one_hot, relu
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics import MeanMetric


class PTS_calibrator:
    """Class for Parameterized Temperature Scaling (PTS)"""
    def __init__(
        self,
        length_logits,
        epochs=1000,
        lr=0.00005,
        weight_decay=0,
        batch_size=1000,
        nlayers=2,
        n_nodes=5,
        top_k_logits=10
    ):
        """
        Args:
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            length_logits (int): length of logits vector
            top_k_logits (int): top k logits used for tuning
        """

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        #Build model
        input_logits = tf.keras.Input(shape=(self.length_logits))
        l2_reg = tf.keras.regularizers.l2(self.weight_decay)

        #Sort logits in descending order and keep top k logits
        t = tf.reshape(
            tf.sort(input_logits, axis=-1,
                    direction='DESCENDING', name='sort'),
            (-1,self.length_logits)
        )
        t = t[:,:self.top_k_logits]

        for _ in range(nlayers):
            t = tf.keras.layers.Dense(self.n_nodes, activation='relu',
                                      kernel_regularizer=l2_reg)(t)

        t = tf.keras.layers.Dense(1, activation=None, name="temperature")(t)
        temperature = tf.math.abs(t)
        x = input_logits/tf.clip_by_value(temperature,1e-12,1e12)
        x = tf.keras.layers.Softmax()(x)

        self.model = tf.keras.Model(input_logits, x, name="model")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError())
        self.model.summary()

    def tune(self, logits, labels, clip=1e2):
        """
        Tune PTS model
        Args:
            logits (tf.tensor or np.array): logits of shape (N,length_logits)
            labels (tf.tensor or np.array): labels of shape (N,length_logits)
        """

        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits)
        if not tf.is_tensor(labels):
            labels = tf.convert_to_tensor(labels)

        assert logits.get_shape()[1] == self.length_logits, "logits need to have same length as length_logits!"
        assert labels.get_shape()[1] == self.length_logits, "labels need to have same length as length_logits!"

        logits = np.clip(logits, -clip, clip)

        self.model.fit(logits, labels, epochs=self.epochs, batch_size=self.batch_size)

    def calibrate(self, logits, clip=1e2):
        """
        Calibrate logits with PTS model
        Args:
            logits (tf.tensor): logits of shape (N,length_logits)
        Return:
            calibrated softmax probability distribution (np.array)
        """

        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits)

        assert logits.get_shape()[1] == self.length_logits, "logits need to have same length as length_logits!"

        calibrated_probs = self.model.predict(tf.clip_by_value(logits, -clip, clip))

        return calibrated_probs


    def save(self, path = "./"):
        """Save PTS model parameters"""

        if not os.path.exists(path):
            os.makedirs(path)

        print("Save PTS model to: ", os.path.join(path, "pts_model.h5"))
        self.model.save_weights(os.path.join(path, "pts_model.h5"))


    def load(self, path = "./"):
        """Load PTS model parameters"""

        print("Load PTS model from: ", os.path.join(path, "pts_model.h5"))
        self.model.load_weights(os.path.join(path, "pts_model.h5"))


class PTSModel_Torch(nn.Module):
    def __init__(
        self,
        length_logits,
        nlayers=2,
        n_nodes=5,
        top_k_logits=10
    ):
        """
        Args:
            length_logits (int): length of logits vector
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            top_k_logits (int): top k logits used for tuning
        """
        super().__init__()
        assert nlayers >= 0

        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        #Build model
        self.layers = []
        if self.nlayers == 0:
            self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=1))
        else:
            self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=self.n_nodes))

        for _ in range(self.nlayers - 1):
            self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes))

        if self.nlayers > 0:
            self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, inp):
        t = torch.topk(inp, self.top_k_logits, dim=1).values
        t = self.layers[0](t)
        if len(self.layers) > 0:
            t = relu(t)

        for layer_idx in range(1, len(self.layers) - 1):
            t = self.layers[layer_idx](t)
            t = relu(t)

        if len(self.layers) > 0:
            t = self.layers[-1](t)

        t = torch.clip(torch.abs(t),1e-12,1e12)

        x = inp / t
        x = torch.softmax(x, dim=1)
        return x


class PTSCalibrator_Torch(nn.Module):
    """
    Class for Parameterized Temperature Scaling (PTS) using PyTorch.
    This was created because current tensorflow versions do not support gpu on native windows. smh
    """

    def __init__(
        self,
        length_logits,
        epochs=1000,
        lr=5e-5,#0.00005,
        weight_decay=0,
        batch_size=1000,
        nlayers=2,
        n_nodes=5,
        top_k_logits=10
    ):
        """
        Args:
            length_logits (int): length of logits vector
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            top_k_logits (int): top k logits used for tuning
        """
        super().__init__()
        assert nlayers >= 0

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        #Build model
        self.layers = []
        if nlayers == 0:
            self.layers.append(nn.Linear(in_features=self.length_logits, out_features=1))
        else:
            self.layers.append(nn.Linear(in_features=self.length_logits, out_features=self.n_nodes))

        for _ in range(nlayers - 1):
            self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes))

        if nlayers > 0:
            self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=1))

        self.model = PTSModel_Torch(self.length_logits, self.nlayers, self.n_nodes, self.top_k_logits)
        self.criterion = nn.MSELoss()
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def tune(self, logits, labels, device="cpu"):
        """
        Tune PTS model
        Args:
            logits (torch.tensor): logits of shape (N,length_logits)
            labels (torch.tensor): labels of shape (N,length_logits)
        """
        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        dset = TensorDataset(logits, labels)

        self.model = self.model.to(device)
        dataloader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        mean_loss = MeanMetric()

        epoch_progress = tqdm(range(self.epochs), desc="Epoch")
        for _ in epoch_progress:
            epoch_loss = 0
            for logits_batch, labels_batch in dataloader:
                self.optimiser.zero_grad()

                logits_batch = logits_batch.to(device)
                labels_batch = labels_batch.to(device)
                outs = self.model(logits_batch)
                loss = self.criterion(outs, labels_batch)
                epoch_loss += loss.item()

                loss.backward()
                self.optimiser.step()

            mean_loss.update(epoch_loss)
            epoch_progress.set_postfix({"mean_loss": mean_loss.compute().item()})

        #logits = np.clip(logits, -clip, clip)

    def calibrate(self, logits, clip=1e2):
        """
        Calibrate logits with PTS model
        Args:
            logits (tf.tensor): logits of shape (N,length_logits)
        Return:
            calibrated softmax probability distribution (np.array)
        """

        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits)

        assert logits.get_shape()[1] == self.length_logits, "logits need to have same length as length_logits!"

        calibrated_probs = self.model.predict(tf.clip_by_value(logits, -clip, clip))

        return calibrated_probs

    def save(self, path = "./"):
        """Save PTS model parameters"""

        if not os.path.exists(path):
            os.makedirs(path)

        print("Save PTS model to: ", os.path.join(path, "pts_model.h5"))
        self.model.save_weights(os.path.join(path, "pts_model.h5"))

    def load(self, path = "./"):
        """Load PTS model parameters"""

        print("Load PTS model from: ", os.path.join(path, "pts_model.h5"))
        self.model.load_weights(os.path.join(path, "pts_model.h5"))


if __name__ == "__main__":
    d = torch.load("openai_clip-vit-large-patch14_imagenet_val.pt")
    logits = d["logits"]
    labels = one_hot(d["labels"], 1000).to(torch.float)
    c = PTSCalibrator_Torch(1000, batch_size=1500)
    c.tune(logits, labels, "cuda")