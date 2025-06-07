# Movie Recommendation Engine using Matrix Factorization

Trained using the [MovieLens](https://grouplens.org/datasets/movielens/) 32M dataset on 4 NVIDIA RTX A600 GPUs using Pytorch's DDP. 

See [run.sh](./run.sh) for the command used to train.

Trained model weights are stored in [mf_checkpoint.pth](./mf_checkpoint.pth).



## NVIDIA GPU communication issues

https://github.com/NVIDIA/nccl/issues/631

Bascially, process 0 will finish a training epoch before the other processes and it will terminate and leave other processes hanging. Just set the environment variable `export NCCL_P2P_DISABLE=1`.


## Pytorch's Distributed Data Parallelism (DDP) for Training 

How DDP works: https://docs.pytorch.org/docs/stable/notes/ddp.html

Using DDP: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun


Note that during **training** `DistributedSampler`, the batch size is set for only 1 GPU. The gradients are averaged over ALL GPUs, so `N * batch_size` samples. 


## Big Dataloading

Read [RAM-Usage in Multiprocess Dataloader  ](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/).

Linux has a *copy-on-write* mechanism: when a process forks, the child process will share its entire memory space with the parent, and only copy the relevant pages when necessary, i.e. when the child process needs to write to the page. This mechanism allows read-only pages to be shared to reduce total memory usage.

However, this mechanism did not help us when we read our dataset. The problem is that our dataset is a large nested data structure that contains many small Python objects. Even though the dataset is "read-only" in theory, accessing any Python object will increment its refcount - causing a lot of memory writes. With these writes, memory can no longer be shared among parent and child processes. In other words, objects are not only copy-on-write, but also copy-on-read.

The end game is that each child process has to replicate all the pages that contain object refcounts in the dataset. For a dataset with many objects, this is almost the size of the dataset itself. 

Therefore, if we only use DistributedSampler and DDP, then this problem will occur.

### Solution

The essence of the solution is to let all processes share memory through a single `torch.Tensor` object, which needs to be moved to Linux shared memory by PyTorch's custom pickling routine. The TLDR on how to achieve sharing is:

- Don't let dataloader workers access many Python objects in their parent. Serialize all objects into a single `torch.Tensor` (but not numpy array) for workers to access.
- Don't let all GPU workers load data independently. Load in one GPU worker, and share with others through a `torch.Tensor`.


The reason why it works, is that multiprocessing uses a customizable pickle implementation called ForkingPickler, and PyTorch customizes how torch.Tensor should be pickled by it: the tensor data will not be serialized to bytes. Instead, during pickling the tensor will be moved to shared memory files (typically under /dev/shm) to be accessed by other processes directly.
