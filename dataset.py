from typing import Any, Optional
import tempfile
import torch
import torch.distributed as dist
from tensordict import MemoryMappedTensor

import os

class TorchTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TorchSharedTensorDataset(TorchTensorDataset):
    def __init__(self, data : torch.Tensor, is_rank0 : bool, world_size : int, metadata : dict  = None):
        """

        Args:
            is_rank0 (bool): Whether this is the rank 0 process.
            data (torch.Tensor): The tensor to be shared.
            metadata (dict, optional): Additional metadata to be shared with the dataset.
        """ 

        if is_rank0:
            super().__init__(data)
            self.tmp_file = tempfile.NamedTemporaryFile(prefix='shared-tensor-', dir='/dev/shm')
            print(f"Rank 0 created file {self.tmp_file.name}")
            filename = self.tmp_file.name
            # immediately unlink the file; the processes should still have a reference
            os.unlink(filename)
            meta_information = (filename, self.data.shape, self.data.dtype, metadata)
        else:
            meta_information = None

        filename, data_shape, data_type, _metadata = local_scatter_torch(meta_information, is_rank0, world_size)
        self.metadata = _metadata 

        if is_rank0:
            self.data = MemoryMappedTensor.from_tensor(self.data, filename=filename, existsok=True)
        else:
            self.data = MemoryMappedTensor.from_filename(filename=filename, dtype=data_type, shape=data_shape)

        dist.barrier()
        


def local_scatter_torch(obj: Optional[Any], is_rank0: bool, world_size: int) -> Any:
    """
    Scatter an object across all processes in a distributed setting. In this case, the object is the meta information needed to find the memory-mapped file, along with its user-defined metadata.
    If `is_rank0` is True, the object is scattered from rank 0 to all other ranks. If `is_rank0` is False, the object is received from rank 0.
    """
    if world_size == 1:
        # Just one worker. Do nothing.
        return obj

    array = [obj] * world_size
    target_array = [None]
    if is_rank0:
        dist.scatter_object_list(target_array, scatter_object_input_list=array, src=0)
    else:
        dist.scatter_object_list(target_array, scatter_object_input_list=None, src=0)
    return target_array[0]
