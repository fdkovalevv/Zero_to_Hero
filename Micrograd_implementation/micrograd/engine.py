import math

class Value:
    # Для того, чтобы наши градиентые текли через наши переменные, 
    # необходимо создать указатели на детей каждой из переменных
    # Также необходимо понимать, какое действие было совершено над детьми
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label


    # Добавляем в класс сложение
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad # "+" добавлен для накопления градиента
            other.grad += 1.0 * out.grad # "+" добавлен для накопления градиента

        out._backward = _backward
        return out 
    

    # Добавялем умножение
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # "+" добавлен для накопления градиента
            other.grad += self.data * out.grad # "+" добавлен для накопления градиента
        out._backward = _backward
        return out 

    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad # "+" добавлен для накопления градиента
        
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supported int/float'
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    
    # Обрвтное распространение ошибки
    def backward(self):

        # Далее реализована Топологическая сортировка
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        # other / self => other * (self ** -1)
        return Value(other) * (self ** -1)
        
    def __rmul__(self, other): # int * Value
        return self * other
    
    def __repr__(self):
        return f'Value(data={self.data})'
