import sys
sys.path.append("../Diffusion-Benchmark/")

from datetime import datetime as dt
from glob import glob
import time

import numpy as np
import torch
import tensorrt as trt
from cuda import cudart


class TRTModel:
    def __init__(self, path="./checkpoints/model.plan"):
        self.path = path

        cudart.cudaDeviceSynchronize()

        #logger = trt.Logger(trt.Logger.VERBOSE)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.path, "rb") as f:
            engineString = f.read()

        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

        # lTensorName: ['sample', 'timestep', 'cond', 'action']
        # nIO: 4
        # nInput: 3
        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)

        self.context = self.engine.create_execution_context()

        for i in range(self.nIO):
            print("[%2d]%s->" % (i, "Input " if i < self.nInput else "Output"), 
                  self.engine.get_tensor_dtype(self.lTensorName[i]), 
                  self.engine.get_tensor_shape(self.lTensorName[i]), 
                  self.context.get_tensor_shape(self.lTensorName[i]), self.lTensorName[i])


    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        bufferH = []
        bufferH.append(np.ascontiguousarray(sample.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(timesteps.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(cond.cpu().numpy()))
        for i in range(self.nInput, self.nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
        
        bufferD = []
        for i in range(self.nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(self.nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))

        res = self.context.execute_async_v3(0)

        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        
        result = bufferH[3]
        
        for b in bufferD:
            cudart.cudaFree(b)

        result = torch.Tensor(result)

        return result


