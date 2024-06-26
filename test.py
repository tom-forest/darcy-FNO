from neuralop.models import TFNO
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training.trainer import Trainer
from load_dataset import load_samples
import torch
import matplotlib.pyplot as plt
import model_tools

model_tools.model_test('tata.pt', 'training_samples_64_simple_mc.pt', 10, 10)