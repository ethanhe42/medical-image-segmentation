import caffe
import numpy as np

class MyLossLayer(caffe.Layer):
    """
    negative sampling, since too few positive examples
    """
    def setup(self,bottom,top):
        if len(bottom) !=2:
            raise Exception("Need two inputs to compute loss")

    def reshape(self,bottom,top):
        pass

    def forward(self,bottom,top):
        pass

    def backward(self,top,propagate_down,bottom):
        pass

