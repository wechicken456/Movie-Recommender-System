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

# In[5]:


from d2l import torch as d2l
import os
import pandas as pd
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
import data_DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import itertools

from common import MemoryMonitor
from dataset import TorchSharedTensorDataset


devices = d2l.try_all_gpus()
print(f"Detected devices: {devices}")


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
            raise "user_id and movie_id must have the same shape."  

        # if this is a single scalar tensor, add a dimension to make it a batch of size 1
        if user_id.ndim == 0: 
            user_id = user_id.unsqueeze(0)
        if movie_id.ndim == 0:
            movie_id = movie_id.unsqueeze(0)

        # convert user_id and movie_id to long tensors if they are not already
        if user_id.dtype != torch.long:
            user_id = user_id.long()
        if movie_id.dtype != torch.long:
            movie_id = movie_id.long()

        # ensure user_id and movie_id are on the same device as the model parameters
        user_id = user_id.to(self.P.weight.device)
        movie_id = movie_id.to(self.Q.weight.device)

        P_u = self.P(user_id)
        Q_m = self.Q(movie_id)
        user_bias = self.user_bias(user_id)
        movie_bias = self.movie_bias(movie_id)
        outputs = torch.sum(P_u * Q_m, dim = 1, keepdim=True) + user_bias + movie_bias
        return outputs


# ### Evaluator - difference between predicted and real rating scores

# In[ ]:


def evaluator(net, test_iter, device=None):
    """
    Compute the RMSE of the model `net` on the test set `test_iter`
    `test_iter` generally consists of batches of (batch_size, 1, 1, 1) tuples. Each tuple is (user_id, movie_id, rating), where `rating` is the ground truth rating.
    """
    mse = nn.MSELoss() # torch doesn't have a built-in RMSE loss, so we use MSE and compute RMSE from it
    rmse = lambda y_hat, y: torch.sqrt(mse(y_hat, y))
    rmse_list = []

    for batch in test_iter:
        if type(batch) is tuple:
            users, movies, ratings = batch
        else:
            users, movies, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        outputs = net(users, movies).squeeze(1)
        rmse_list.append(rmse(outputs, ratings.to(device)))
    rmse_list = torch.tensor(rmse_list, device=device)
    return rmse_list.mean()


# ### Training and Etestuating the model

# In[ ]:


class TrainerDDP(d2l.Trainer):
    def __init__(self, max_epochs, optimizer, loss, ddp_rank, world_size, lr = 0.002, wd = 1e-5, gradient_clip_test=0, save_every_n_epochs=2):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.test_losses = []
        self.optim = optimizer
        self.loss = loss
        self.world_size = world_size
        self.ddp_rank = ddp_rank
        self.save_every_n_epochs = save_every_n_epochs
        # To plot the training and test losses
        self.board = d2l.ProgressBoard()


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

        self.prepare_data(train_iter, test_iter)
        self.prepare_model(model)
        self.epoch = 0
        self.train_batch_idx = 0
        self.test_batch_idx = 0
        self.train_losses = []
        self.test_losses = []
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()    

            # save model checkpoint
            if self.ddp_rank == 0 and (self.epoch  % self.save_every_n_epochs == 0 or self.epoch == self.max_epochs - 1):
                checkpoint_path = "./mf_checkpoint.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")


    def fit_epoch(self):
        self.model.train()
        dist.barrier()
        self.train_dataloader.sampler.set_epoch(self.epoch)  # set seed for shuffling in DDP

        total_loss = torch.tensor(0.0, device=self.ddp_rank)
        batch_size = 0
        print(f"Rank {self.ddp_rank}, Epoch {self.epoch + 1}/{self.max_epochs}, Dataloader size: {len(self.train_dataloader)}. START.", flush = True)
        for batch in self.train_dataloader:
            batch_size = len(batch)
            loss = self.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                total_loss += loss

            self.train_batch_idx += 1
        print(f"Rank {self.ddp_rank}, Epoch {self.epoch + 1}/{self.max_epochs}, Dataloader size: {len(self.train_dataloader)}. DONE.", flush = True)

        # Average the loss across all batches
        total_loss = total_loss / self.train_batch_idx  
        local_train_loss = total_loss.clone()  # Keep a local copy of the loss for logging

        # Synchronize and average the total loss across all processes
        dist.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM) 
        total_loss = total_loss / self.world_size  
        self.train_losses.append(total_loss)

        if self.test_dataloader is None:
            return

        self.model.eval()
        with torch.no_grad():
            loss = self.test_step()
            local_test_loss = loss.clone()  # Keep a local copy of the loss for logging
            dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM) # Add up the losses from all processes
            self.test_losses.append(loss / self.world_size)  # Average the loss across all processes
            self.test_batch_idx += 1

        if self.ddp_rank == 0:       
            self.plot('loss', self.train_losses[-1], train=True)
            self.plot('loss', self.test_losses[-1], train=False)
            plt.savefig("mf_loss.png")

        print(f"Rank {self.ddp_rank}, Epoch {self.epoch + 1}/{self.max_epochs}. \nAvg Train Loss: {self.train_losses[-1]:.4f}, Avg Test RMSE: {self.test_losses[-1]:.4f}, Local Train Loss: {local_train_loss:.4f}, Local Test Loss: {local_test_loss:.4f}.\nTrain Batch: {self.train_batch_idx}, Test Batch: {self.test_batch_idx}, # of samples in train batch: {batch_size}", flush = True)


    def prepare_batch(self, batch):
        """
        Prepare the batch for training or testing.
        This method is called implicitly by fit() before each batch is passed to the forward call of self.model
        Note that due to the nature of memory-mapped file in distributed dataloading (`dataset.py`), 
        the dataloader will use tensors instead of TensorDataset. 
        Hence, each batch is a tensor of [[u1, m1, r1], [u2, m2, r2]...,] where each [u, m, r] is a tensor of (user_id, movie_id, rating).
        So we need to convert the batch to [users, movies, ratings] before passing it to the model.
        batch is a tensor of [[u1, m1, r1], [u2, m2, r2]...,] where each [u, m, r] is a tensor of (user_id, movie_id, rating).
        """
        users, movies, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        return users, movies, ratings

    def training_step(self, batch):
        users, movies, ratings = batch
        outputs = self.model(users, movies).squeeze(1)
        loss = self.loss(outputs, ratings.to(self.ddp_rank))
        return loss

    def test_step(self):
        loss = evaluator(self.model, self.test_dataloader, self.ddp_rank)
        return loss


    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        self.board.draw(self.epoch, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=self.save_every_n_epochs)


# In[ ]:


def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12555"
    os.environ["NCCL_P2P_DISABLE"] = "1"  

def worker(rank, world_size):    
    torch.cuda.set_device(rank)
    print(f"Process {rank} started. Device: {torch.cuda.current_device()}. World size: {world_size}.", flush=True)
    ddp_setup()

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.empty_cache()

    batch_size = 64

    monitor = MemoryMonitor()

    print(f"Process {rank} akdalkfslakflaksdlfklak", flush=True)

    # Only rank 0 will load the datasets and create the dataloaders.
    if rank == 0:
        num_users, num_movies, train_set, test_set = data_DDP.get_datasets(test_ratio=0.1, batch_size=batch_size, world_size=world_size, is_big_computer=False)
        train_set = TorchSharedTensorDataset(data=train_set, is_rank0=True, world_size=world_size, metadata={'num_users': num_users, 'num_movies': num_movies})
        test_set = TorchSharedTensorDataset(data=test_set, is_rank0=True, world_size =world_size, metadata={'num_users': num_users, 'num_movies': num_movies})
    else:
        train_set = TorchSharedTensorDataset(data=None, is_rank0=False, world_size=world_size)
        test_set = TorchSharedTensorDataset(data=None, is_rank0=False, world_size=world_size)

    if rank == 0:
        print(monitor.table())

    dist.barrier()
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True,
                                            shuffle=False, sampler=DistributedSampler(train_set, rank=rank, num_replicas = world_size, shuffle = True, drop_last=True)) 
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True,
                                            shuffle=False, sampler=DistributedSampler(test_set, rank=rank, num_replicas = world_size, shuffle = False, drop_last=True))



    # Setup memory monitor
    pids = [os.getpid()]
    all_pids = [None for _ in range(world_size)]
    dist.all_gather_object(all_pids, pids)
    all_pids = list(itertools.chain.from_iterable(all_pids))
    monitor = MemoryMonitor(all_pids)
    if rank == 0:
        print(monitor.table())

    num_users = train_set.metadata['num_users']
    num_movies = train_set.metadata['num_movies']
    print(f"Rank: {rank}, finished loading data. Number of users: {num_users}. Number of movies: {num_movies}. Train set size: {len(train_set)}. Test set size: {len(test_set)}", flush=True)


    # Create the model and wrap it in DDP
    net = MF(32, num_users, num_movies).to(rank)
    ddp_net = DDP(net, device_ids=[rank])


    lr = 0.002  
    wd = 1e-5
    num_epochs = 20
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    trainer = TrainerDDP(max_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, lr=lr, 
                        wd=wd, ddp_rank = rank, world_size = world_size, save_every_n_epochs=1)

    trainer.fit(ddp_net, train_iter, test_iter)


    if rank == 0:
        CHECKPOINT_PATH = "./mf_checkpoint.pth"
        torch.save(ddp_net.state_dict(), CHECKPOINT_PATH)

    dist.destroy_process_group()


# In[ ]:





# In[ ]:


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} for training.")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)    
    print("Training complete.")


# In[ ]:




