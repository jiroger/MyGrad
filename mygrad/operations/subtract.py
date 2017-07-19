from .operation_base import BroadcastableOp

## STUDENT CODE HERE ####
__all__ = []
## FILL IN  __all__ ###

class Subtract(BroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'subtract' forward-pass: f(a,b) -> a - b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.a = a
        self.b = b
        out = a.data - b.data
        self.broadcast_check(a,b,out.shape)
        return np.asarray(out)
    
        # STUDENT CODE HERE
        # 1. Bind a and b as attributes of `self`
        # 2. Compute out = a - b, **using the underlying numpy-arrays stored by these tensors**
        # 3. Because this is a BroadcastableOP, call: self.broadcast_check(a, b, out.shape)
        # 4. return the result of the forward pass, which should **not** be a Tensor instance, but a numpy-array
        

    def backward_a(self, grad):
        bcg = self.a.backward(super(Subtract, self).backward_a(-b))
        self.a.backward(bcg)
        # STUDENT CODE HERE
        # 1. Given, dL/df (a.k.a `grad`), compute dL/da
        # 2. Because this is a broadcastable op, get the broadcasted gradient
        #    by calling super(Subtract, self).backward_a( dL/da )
        # 3. Pass the broadcasted gradient to self.a.backward

    def backward_b(self, grad):
        # STUDENT CODE HERE
        bcg = self.b.backward(super(Subtract, self).backward_b(-a))
        self.b.backward(bcg)
