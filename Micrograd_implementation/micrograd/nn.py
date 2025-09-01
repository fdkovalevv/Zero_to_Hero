import random
from .engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
        
    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
    
    def __call__(self, x):
        activation = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = activation.tanh() if self.nonlin else activation
        return out

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f'{'Relu' if self.nonlin else 'Linear'} Neuron({len(self.w)})'
    

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.layer = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.layer]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.layer for p in neuron.parameters()]
    
    def __repr__(self):
        return f'Layer of [{', '.join(str(n) for n in self.layer)}]'


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # Все скрытые слои нелинейные, последний слой — линейный
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i < len(nouts) - 1))
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f'MLP of [{', '.join(str(layer) for layer in self.layers)}]'
