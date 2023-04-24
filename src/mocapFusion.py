import numpy as np
import matplotlib.pyplot as plt
from dataloader import *

class OnlineLearingFusion:
    def __init__(self):
        pass
    def propogateStep(self):
        raise NotImplementedError
    def measurmentStep(self):
        raise NotImplementedError
    
    def runPipeline(self):
        raise NotImplementedError