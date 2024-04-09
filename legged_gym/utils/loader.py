

import torch

from legged_gym.scripts.trt_model import TRTModel


def loadModel(checkpoint_name: str):
    """
    Load a model from a file

    Args:
        checkpoint_name: str
            The name of the file to load the model from

    Returns:
        model: TRTModel or torch.Model
            The loaded model
    """
    
    # check file extension name
    if checkpoint_name.endswith(".plan"):
        model = TRTModel(checkpoint_name)

    elif checkpoint_name.endswith(".pt"):
        model = torch.load(checkpoint_name)
        model.eval()
    
    return model