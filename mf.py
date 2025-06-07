#!/usr/bin/env python
# coding: utf-8

# ## Matrix Factorization
# 
# Let $\mathbf{R} \in \mathbb{R}^{m \times n}$ denote the interaction matrix with $m$ users and $n$ items, and the testues of $\mathbf{R}$ represent explicit ratings. The user-item interaction will be factorized into a user latent matrix $\mathbf{P} \in \mathbb{R}^{m \times k}$ and an item latent matrix $\mathbf{Q} \in \mathbb{R}^{n \times k}$, where $k \ll m, n$, is the latent factor size. 
# 
# Let $\mathbf{p}_u$ denote the $u^{th}$ row of $\mathbf{P}$ and $\mathbf{q}_i$ denote the $i^{th}$ row of $\mathbf{Q}$. For a given item $i$, the elements of $\mathbf{q}_i$ measure the extent to which the item possesses those characteristics such as the genres and languages of a movie. For a given user $u$, the elements of $\mathbf{p}_u$ measure the extent of interest the user has in items' corresponding characteristics. These latent factors might measure obvious dimensions as mentioned in those examples or are completely uninterpretable. The predicted ratings can be estimated by
# $$ \hat{\mathbf{R}} = \mathbf{P} \mathbf{Q}^{\top} $$
# where $\hat{\mathbf{R}} \in \mathbb{R}^{m \times n}$ is the predicted rating matrix which has the same shape as $\mathbf{R}$. One major problem of this prediction rule is that users/items biases can not be modeled. For example, some users tend to give higher ratings or some items always get lower ratings due to poorer quality. These biases are commonplace in real-world applications. To capture these biases, user specific and item specific bias terms are introduced. Specifically, the predicted rating user $u$ gives to item $i$ is calculated by
# $$ \hat{R}_{ui} = \mathbf{p}_u \mathbf{q}_i^{\top} + b_u + b_i $$
# Then, we train the matrix factorization model by minimizing the mean squared error between predicted rating scores and real rating scores. The objective function is defined as follows:
# $$ \underset{\mathbf{P}, \mathbf{Q}, b}{\text{argmin}} \sum_{(u,i) \in \mathcal{K}} \| R_{ui} - \hat{R}_{ui} \|^2 + \lambda (\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2 + b_u^2 + b_i^2) $$
# where $\lambda$ denotes the regularization rate. The regularizing term $\lambda (\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2 + b_u^2 + b_i^2)$ is used to avoid over-fitting by penalizing the magnitude of the parameters. The $(u,i)$ pairs for which $R_{ui}$ is known are stored in the set $\mathcal{K} = \{(u,i) \mid R_{ui} \text{ is known}\}$. The model parameters can be learned with an optimization algorithm, such as Stochastic Gradient Descent and Adam.
# 
# 

# In[ ]:


from d2l import torch as d2l
import os
import pandas as pd
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
import data

devices = d2l.try_all_gpus()
print(f"Detected devices: {devices}")


# In[ ]:


num_users, num_movies, train_iter, test_iter = data.split_and_load_data(test_ratio=0.1, batch_size=256)
print("Finished loading data. Number of users:", num_users, "Number of movies:", num_movies)


# In[ ]:


class MF(d2l.Module):
    def __init__(self, num_latent, num_users, num_movies, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_latent)
        self.Q = nn.Embedding(num_movies, num_latent)
        self.num_latent = num_latent
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_id, movie_id):
        """
        user_id and movie_id should be tensors of shape (batch_size,).
        This function computes a pairwise dot product between each user and movie pair in the batch.
        Returns a tensor of shape (batch_size, 1) containing the predicted ratings for each pair.
        """
        if user_id.shape != movie_id.shape:
            raise testueError("user_id and movie_id must have the same shape.")        

        # if this is a single scalar tensor, add a dimension to make it a batch of size 1
        if user_id.ndim == 0: 
            user_id = user_id.unsqueeze(0)
        if movie_id.ndim == 0:
            movie_id = movie_id.unsqueeze(0)

        P_u = self.P(user_id)
        Q_m = self.Q(movie_id)
        user_bias = self.user_bias(user_id)
        movie_bias = self.movie_bias(movie_id)
        outputs = torch.sum(P_u * Q_m, dim = 1, keepdim=True) + user_bias + movie_bias
        return outputs


# In[4]:


mf = MF(32, num_users, num_movies)
mf(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])).shape


# ### Evaluator - difference between predicted and real rating scores

# In[5]:


def evaluator(net, test_iter, devices=None):
    """
    Compute the RMSE of the model `net` on the test set `test_iter`
    `test_iter` generally consists of batches of (batch_size, 1, 1, 1) tuples. Each tuple is (user_id, movie_id, rating), where `rating` is the ground truth rating.
    """
    mse = nn.MSELoss() # torch doesn't have a built-in RMSE loss, so we use MSE and compute RMSE from it
    rmse = lambda y_hat, y: torch.sqrt(mse(y_hat, y))
    rmse_list = []

    # Wrap the model with DataParallel if multiple GPUs are available
    # Then, move the base model (or DataParallel wrapper) to the primary device as the parallelized module must 
    # have its parameters and buffers on device_ids[0] before running this DataParallel module.
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    if devices:
        net.to(devices[0])
        if len(devices) > 1:
            net = nn.DataParallel(net, device_ids=devices)
            print(f"Model wrapped with DataParallel on {len(devices)} GPUs.")    

    for idx, (users, movies, ratings) in enumerate(test_iter):
        users = users.to(devices[0] if devices else torch.device("cpu"))
        movies = movies.to(devices[0] if devices else torch.device("cpu"))
        ratings = ratings.to(devices[0] if devices else torch.device("cpu"))    
        outputs = net(users, movies).squeeze(1)
        rmse_list.append(rmse(outputs, ratings).item())
    return np.mean(np.array(rmse_list))


# In[6]:


mf = MF(32, num_users, num_movies)
evaluator(mf, test_iter, devices)


# ### Training and Etestuating the model

# In[10]:


class Trainer(d2l.Trainer):
    def __init__(self, max_epochs, optimizer, loss, lr = 0.002, wd = 1e-5, num_gpus=torch.cuda.device_count(), gradient_clip_test=0):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.test_losses = []
        self.optim = optimizer
        self.loss = loss
        self.devices = d2l.try_all_gpus() if num_gpus else [torch.device('cpu')]

    def prepare_model(self, model):
        """
        Configure the model for training, including wrapping it with DataParallel if multiple GPUs are available.
        This method is called implicitly by fit() before the training loop starts.

        Wrap the model with DataParallel if multiple GPUs are available.
        Then, move the base model (or DataParallel wrapper) to the primary device as the parallelized module must 
        have its parameters and buffers on device_ids[0] before running this DataParallel module.
        The data can be on any device, but the model must be on the primary device.
        https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html  
        """
        self.model = model

        if self.devices:
            self.model.to(self.devices[0])
            if len(self.devices) > 1: 
                self.model = nn.DataParallel(self.model, device_ids=self.devices)
                print(f"Model wrapped with DataParallel on {len(self.devices)} GPUs.")
        self.model.to(self.devices[0])

    def prepare_data(self, train_iter, test_iter):
        self.train_dataloader = train_iter
        self.test_dataloader = test_iter
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = (len(self.test_dataloader)
                                if self.test_dataloader is not None else 0)

    def fit(self, model, train_iter, test_iter):
        self.animator = d2l.Animator(xlabel='epoch', xlim=[1, self.max_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])

        self.prepare_data(train_iter, test_iter)
        self.prepare_model(model)
        self.epoch = 0
        self.train_batch_idx = 0
        self.test_batch_idx = 0
        self.train_losses = []
        self.test_losses = []
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()    


    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()

            self.train_batch_idx += 1
            num_batches = len(self.train_dataloader)
            self.train_losses.append(loss.item())


        if self.test_dataloader is None:
            return

        self.model.eval()
        with torch.no_grad():
            loss = self.test_step()
            self.test_losses.append(loss)
            self.test_batch_idx += 1

        self.animator.add(self.epoch + 1, (self.train_losses[-1], self.test_losses[-1]))
        plt.savefig("mf_loss.png")


    def training_step(self, batch):
        users, movies, ratings = batch
        outputs = self.model(users, movies).squeeze(1)
        loss = self.loss(outputs, ratings.to(self.devices[0]))
        return loss

    def test_step(self):
        return evaluator(self.model, self.test_dataloader, self.devices)

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')

        if self.test_losses:
            plt.plot(range(1, len(self.test_losses) + 1), self.test_losses, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)

        # Clear the output to avoid multiple plots
        from IPython.display import clear_output
        clear_output(wait=True)
        plt.show()




# In[ ]:


class TrainerDDP(d2l.Trainer):
    def __init__(self, max_epochs, optimizer, loss, lr = 0.002, wd = 1e-5, ddp_rank = None, gradient_clip_test=0):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.test_losses = []
        self.optim = optimizer
        self.loss = loss
        self.ddp_rank = ddp_rank

    def prepare_model(self, model):
        """
        The model should already be wrapped around DDP() on the correct device (self.device).
        """
        self.model = model


    def prepare_data(self, train_iter, test_iter):
        self.train_dataloader = train_iter
        self.test_dataloader = test_iter
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = (len(self.test_dataloader)
                                if self.test_dataloader is not None else 0)

    def fit(self, model, train_iter, test_iter):
        self.animator = d2l.Animator(xlabel='epoch', xlim=[1, self.max_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])

        self.prepare_data(train_iter, test_iter)
        self.prepare_model(model)
        self.epoch = 0
        self.train_batch_idx = 0
        self.test_batch_idx = 0
        self.train_losses = []
        self.test_losses = []
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()    


    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()

            self.train_batch_idx += 1
            num_batches = len(self.train_dataloader)
            self.train_losses.append(loss.item())


        if self.test_dataloader is None:
            return

        self.model.eval()
        with torch.no_grad():
            loss = self.test_step()
            self.test_losses.append(loss)
            self.test_batch_idx += 1

        self.animator.add(self.epoch + 1, (self.train_losses[-1].cpu().numpy(), self.test_losses[-1].cpu().numpy()))
        plt.savefig("mf_loss.png")


    def prepare_batch(self, batch):
        """
        Prepare the batch for training or testing.
        This method is called implicitly by fit() before each batch is passed to the forward call of self.model
        It moves the batch to the correct device (self.device).
        """
        users, movies, ratings = batch
        return (users.to(self.ddp_rank), movies.to(self.ddp_rank), ratings.to(self.ddp_rank))

    def training_step(self, batch):
        users, movies, ratings = batch
        outputs = self.model(users, movies).squeeze(1)
        loss = self.loss(outputs)
        return loss

    def test_step(self):
        return evaluator(self.model, self.test_dataloader, None)

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')

        if self.test_losses:
            plt.plot(range(1, len(self.test_losses) + 1), self.test_losses, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)

        # Clear the output to avoid multiple plots
        from IPython.display import clear_output
        clear_output(wait=True)
        plt.show()


# In[ ]:


def worker(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    net = MF(32, num_users, num_movies).to(rank)

    ddp_net = DDP(net, device_ids=[rank])

    lr = 0.002
    wd = 1e-5
    num_epochs = 10
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    trainer = TrainerDDP(max_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, lr=lr, wd=wd, ddp_rank = rank)

    trainer.fit(ddp_net, train_iter, test_iter)


    if rank == 0:
        CHECKPOINT_PATH = "./mf_checkpoint.pth"
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    dist.destroy_process_group()




def main():
    world_size = torch.cuda.device_count()

    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

    print(f"Using {world_size} for training.")
    mp.spawn()



# In[ ]:


devices = d2l.try_all_gpus()
net = MF(32, num_users, num_movies)
lr = 0.002
wd = 1e-5
num_epochs = 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
loss_fn = nn.MSELoss()
trainer = Trainer(max_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, lr=lr, wd=wd, num_gpus=len(devices))


# In[ ]:


trainer.fit(net, train_iter, test_iter)


# In[9]:


scores = net(torch.tensor([20], dtype=torch.int32, device=devices[0] if devices else torch.device('cpu')),
             torch.tensor([30], dtype=torch.int32, device=devices[0] if devices else torch.device('cpu')) 
             )
scores


# In[ ]:




