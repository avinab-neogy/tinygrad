import math

class Value:
    
    def __init__(self,data,_children=(),_op='',label =''):
        self.data = data
        self.grad = 0
        self._backward = lambda : None 
        # Setting _backward to lambda: None allows every Value node to 
        # safely have a backward function by default and lets you swap in the correct logic when you 
        # know what operation created the node
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self,other):
        other = other  if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += 1 * out.grad #accumultaing the gradients in case inputs repeats itself
            other.grad += 1 * out.grad
        out._backward = _backward
        # we are only assigning the function here and NOT calling hence no brackets
        return out

    def __radd__(self,other): #other + self
        return self + other
    
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __mul__(self,other):
        other = other  if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self,other): #fallback inbuilt in py that reverses the other to do 2*a since mul cant do that
        return self*other
        
    def __truediv__(self,other):
        other = other  if isinstance(other, Value) else Value(other)
        out = self*other**(-1.0)
        return out

    def __rtruediv__(self,other):
        return other * self**(-1.0)
    
    def tanh(self):
        n= self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)) #only supporting int/float powers for now
        out = Value(self.data**other, (self,), f'**{other}')
    
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        n= self.data
        out = Value(math.exp(n),(self,),'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for node in reversed(topo):
            node._backward()