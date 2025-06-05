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

# In[12]:


from d2l import torch as d2l
import os
import pandas as pd
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
import data_DDP as data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, all_reduce
import torch.multiprocessing as mp


devices = d2l.try_all_gpus()
print(f"Detected devices: {devices}")


# In[2]:


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

    for users, movies, ratings in test_iter:
        outputs = net(users, movies).squeeze(1)
        rmse_list.append(rmse(outputs, ratings.to(device)))
    rmse_list = torch.tensor(rmse_list, device=device)
    return rmse_list.mean()


# ### Training and Etestuating the model

# In[ ]:


class TrainerDDP(d2l.Trainer):
    def __init__(self, max_epochs, optimizer, loss, ddp_rank, lr = 0.002, wd = 1e-5, gradient_clip_test=0, save_every_n_epochs=5):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.test_losses = []
        self.optim = optimizer
        self.loss = loss
        self.ddp_rank = ddp_rank
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
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        barrier()
        self.train_dataloader.sampler.set_epoch(self.epoch)  # set epoch for shuffling in DDP
        total_loss = torch.tensor(0.0, device=self.ddp_rank)
        batch_size = 0
        print(f"Process {self.ddp_rank}, Epoch {self.epoch + 1}/{self.max_epochs}, Dataloader size: {len(self.train_dataloader)}", flush = True)
        for batch in self.train_dataloader:
            batch_size = len(batch[0])
            loss = self.training_step(batch)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()
                
            self.train_batch_idx += 1
            num_batches = len(self.train_dataloader)
            total_loss += loss

        self.train_losses.append(total_loss / len(self.train_dataloader.dataset))

        if self.test_dataloader is None:
            return

        self.model.eval()
        with torch.no_grad():
            loss = self.test_step()
            all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            self.test_losses.append(loss / len(self.test_dataloader.dataset))


            self.test_batch_idx += 1

        if self.ddp_rank == 0:       
            self.plot('loss', self.train_losses[-1], train=True)
            self.plot('loss', self.test_losses[-1], train=False)
            plt.savefig("mf_loss.png")
        
        print(f"Process {self.ddp_rank}, Epoch {self.epoch + 1}/{self.max_epochs}, Train Loss: {self.train_losses[-1]:.4f}, Test RMSE: {self.test_losses[-1]:.4f}, Train Batch: {self.train_batch_idx}, Test Batch: {self.test_batch_idx}, # of samples in train batch: {batch_size}", flush = True)

    
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
        loss = self.loss(outputs, ratings.to(self.ddp_rank))
        return loss

    def test_step(self):
        loss = evaluator(self.model, self.test_dataloader, self.ddp_rank)
        return loss


    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        if train:
            x = self.train_batch_idx / \
                self.num_train_batches
            n = self.num_train_batches / \
                2
        else:
            x = self.epoch + 1
            n = self.num_test_batches / \
                2
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))


# In[ ]:


def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

def worker(rank, world_size):    
    init_process_group("nccl", rank=rank, world_size=world_size)
    barrier()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    global num_users, num_movies, train_set, test_set
    train_iter, test_iter = data.get_dataloaders(train_set, test_set, batch_size=32)
    net = MF(32, num_users, num_movies).to(rank)

    ddp_net = DDP(net, device_ids=[rank])

    lr = 0.002
    wd = 1e-5
    num_epochs = 5
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    trainer = TrainerDDP(max_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, lr=lr, wd=wd, ddp_rank = rank)

    trainer.fit(ddp_net, train_iter, test_iter)


    if rank == 0:
        CHECKPOINT_PATH = "./mf_checkpoint.pth"
        torch.save(ddp_net.state_dict(), CHECKPOINT_PATH)

    destroy_process_group()


# In[ ]:


num_users, num_movies, train_set, test_set = data.get_datasets() 


# In[ ]:


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Finished loading data. Number of users:", num_users, "Number of movies:", num_movies)
    print(f"Using {world_size} for training.")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)    
    print("Training complete.")

