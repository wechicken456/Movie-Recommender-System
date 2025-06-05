## NVIDIA GPU communication issues

https://github.com/NVIDIA/nccl/issues/631

Bascially, process 0 will finish a training epoch before the other processes and it will terminate and leave other processes hanging. Just set the environment variable `export NCCL_P2P_DISABLE=1`.


## Pytorch's Distributed Data Parallelism (DDP) for Training 

How DDP works: https://docs.pytorch.org/docs/stable/notes/ddp.html

Using DDP: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun


Note that during **training** `DistributedSampler`, the batch size is set for only 1 GPU. The gradients are averaged over ALL GPUs, so `N * batch_size` samples. 

During **testing**, we only use 