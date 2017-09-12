from __future__ import print_function
import theano
import numpy as np
from theano import pp
from theano import function
from theano.scalar.basic import same_out_nocomplex
from theano.scalar.basic import UnaryScalarOp
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from theano.tensor.elemwise import Elemwise
#
#Every .eval() call initializes graph again,
#So random function will make error checking hard. 
#Be careful.
a = RandomStreams(lasagne.random.get_rng().randint(2, 5)) #Initializes a random stream, dont know what randint does here. """lasagne.random.get_rng() ==> np.random
b = T.cast(a.binomial(size=(4,5)), theano.config.floatX) #If you remove T.switch, some bullshit hapens
d = T.cast(T.switch(b, 1, -1), theano.config.floatX)
ab = T.cast(a.binomial(size=(4,3)), dtype=theano.config.floatX)
print(b.eval())
#print(b.eval())
#print(d.eval()) #If you initialize without random.seed(1), you'll get different results.
#                #because d and b are evaluated in different computation graphs
#
#
#def returner(inp, out):
#
#    return "%(inp)s = round(%(out)s);" % locals()
#print(returner(555,3))
#
#
#class Round3(UnaryScalarOp): #This class extends UnaryScalarOpto support custom formatting
#    def c_code(self, node, name, inputs, outputs, sub):
#        (x,) = inputs
#        (z,) = outputs
#        return "%(z)s = round(%(x)s);" % locals() # Run returner to find out its function.
#
#    def grad(self, inputs, gout):
#
#        (gz,) = gout
#        return gz,
#
#round3_scalar = Round3(same_out_nocomplex, name='round3')
#round3 = Elemwise(round3_scalar)
#        
#ceee = np.array([1], dtype=np.complex128)
#
#
#def hard_sigmoid(x):
#    return T.clip((x+1.)/2., 0,1)
#
#    
#def binary_tanh_unit(x):
#    return 2.*round3(x) - 1.
##print(ab.eval())
#ced = T.cast(binary_tanh_unit(ab), theano.config.floatX)
#print(ced.eval())
#
#
#def binarization(W, H, binary = True, deterministic = False, stochastic = False, srng = None):
#
#    if not binary or (deterministic and stochastic):
#        Wb = W #No change
#    
#    else:
#        Wb = hard_sigmoid(W/H)  #Prepare for rounding
#        if stochastic:
#            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX) #srng 
#                                                                                          #INITIALIZES RANDOM STREAM
#        else:
#            Wb = T.round(Wb)
#        Wb = T.cast(T.switch(Wb, H, -H), theano.config.floatX)
#    return Wb
    
k =4*np.random.rand(4,4)
print(np.intc(k))
#class Conv2DLayer(lasagne.layers.Conv2DLayer):
#
#        
#        
#        
#        
#        
#        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    