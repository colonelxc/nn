import numpy as np


class Network(object):

    def __init__(self, layers):
        self.layers = layers

    def query(self, inputs, alpha=0.01):
        self.alpha = alpha
        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = outputs # outputs are the inputs to the next layer
        return outputs

    def batchFit(self, inputs, expected_outputs):
        data = [inputs]
        for i in range(len(self.layers)):
            data.append(self.layers[i].forward(data[i]))

        gradient = data[-1] - expected_outputs
        gradient /= expected_outputs.shape[0] #scale down the gradient by the number of changes we're tracking. Otherwise we would murder the layers.
        for i in range(len(self.layers)):
            layer = self.layers[-1-i]
#            print "grad", gradient
#            print "last in", data[-2-i]
#            print "last in shape", data[-2-i].shape
#            print "last out", data[-1-i]
            dw, db, gradient = layer.backward(gradient, data[-2-i], data[-1-i])
            # update (TODO, regularization?)
#            print "i", i
#            print "dw", dw
#            print "db", db
            layer.W = layer.W - (self.alpha * dw)
            layer.B = layer.B - (self.alpha * db)

class Layer(object):
    def __init__(self, size_in, size_out, activation=None):
        #TODO make into float16's
        self.W = np.random.randn(size_in, size_out) * np.sqrt(2.0/size_in)
        self.B = np.zeros((1, size_out))
        self.activation = activation

    def forward(self, in_act):
        # remember for backprop
        self.last_in_act = in_act

        # Feed forward against weights
        out_act = np.dot(in_act, self.W) + self.B

        if self.activation is not None:
            self.activation.forward(out_act)

        # remember for backprop
        self.last_out_act = out_act

        return out_act

    def backward(self, out_grad, last_in_act, last_out_act):
        if self.activation is not None:
            self.activation.backward(out_grad, last_out_act)

        #print "Grad", out_grad.shape
        #print "last_in_act.T", last_in_act.T.shape
        dW = np.dot(last_in_act.T, out_grad)
        dB = np.sum(out_grad, axis=0, keepdims=True)

        #print "dw", dW.shape
        #print "db", dB.shape

        # compute the gradient for the next layer
        higher_out_grad = np.dot(out_grad, self.W.T)

        return dW, dB, higher_out_grad

class Activation(object):

    def forward(self, weighted):
        """Takes the weighted values, and applies the activation function (mutates)"""
        pass

    def backward(self, backprop_gradient, last_out_activation):
        """mutates the backprop_gradient by applying the derivative of the Activation"""
        pass

class Linear(Activation):
    pass


class ReLU(Activation):
    def forward(self, weighted): 
        weighted[weighted < 0] = 0

    def backward(self, backprop_gradient, last_out_activation):
        # strictly speaking, there shouldn't be anything less than zero
        # in last_out_activation, so we're really matching things that are zero.
        # But floating point, so whatev.
        backprop_gradient[last_out_activation <= 0] = 0
