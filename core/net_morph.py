"""

Network Morphism
Author: Kyunghyun Paeng

"""
import numpy as np
import itertools

class NetMorph(object):
    def __init__(self):
        print 'NetMorp module initialize...'

    def morph(self, G, channel):
        """ Network Morph
        
        Args:
            G: Target weight matrix (kH, kW, InChannel, OutChannel) 
            channel: the number of channels for morphing
            k1: the kernel size of first layer
            k2: the kernel size of second layer
      
        Returns:
            Two weights matrices for target matrix (G)
            - F1 = (k1H, k1W, InChannel, channel)
            - F2 = (k2H, k2W, channel, OutChannel)
      
        """
        if G.ndim == 2:
            return self._fc_morph(G, channel)
        elif G.ndim == 4:
            print 'Not Support conv mode'

    def _conv_morph(self, G, channel, k1, k2):
        print 'conv morph'

    def _fc_morph(self, G, channel):
        c0 = G.shape[0]
        c1 = channel
        c2 = G.shape[1]

        f1 = np.random.rand(c0, c1)
        f2 = np.random.rand(c1, c2)
       
        new_f2 = np.dot( np.linalg.pinv(f1), G)
        new_f1 = np.dot( G, np.linalg.pinv(new_f2) )
        # normalization
        m1 = new_f1.mean()
        new_f1 = (new_f1-m1)/(np.sqrt(new_f1.var())*10.)
        m2 = new_f2.mean()
        new_f2 = (new_f2-m2)/(np.sqrt(new_f2.var())*10.)
        #if True: #verification:
        #    loss = G - np.matmul(new_f1, new_f2)
        #    print np.sum(loss**2)

        return new_f1, new_f2

if __name__ == '__main__':
    """
    obj = NetMorph()
    # test test test
    c0 = 768
    c1 = 50
    c2 = 10
    
    g = np.random.rand(c0,c2)
    f1, f2 = obj.morph(g, c1)
    """
    aaa = np.random.rand(3,3)
    bbb = np.random.rand(3,3)
    import scipy.signal as ss
    ccc = ss.convolve2d(aaa,bbb)

    import ipdb
    ipdb.set_trace()
    print 'test done'

