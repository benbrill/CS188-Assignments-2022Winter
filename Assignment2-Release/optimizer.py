import torch

class SGD():

    def __init__(self, params, lr=0.01, momentum=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.buf = dict()

        for j in range(len(self.params)):
            self.buf[j] = None

    def zero_grad(self):
        for (j,p) in enumerate(self.params):
            p.grad = None

    def step(self):
        for (j,p) in enumerate(self.params):
            if p.grad is None:
                continue

            g_t = p.grad.data
            
            """
            Implement SGD optimizer 
            The goal is to modify p.data, which are the parameters (theta) of the neural network.
            buf saves the buffer at each time step

            """
            #############################################################Your code here################################

            #############################################################################################

class Adam():

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.buf_m = dict()
        self.buf_v = dict()
        self.counter = 1
        for j in range(len(self.params)):
            self.buf_m[j] = None
            self.buf_v[j] = None

    def zero_grad(self):
        for (j,p) in enumerate(self.params):
            p.grad = None

    def step(self):
        for (j,p) in enumerate(self.params):
            if p.grad is None:
                continue

            d_p = p.grad.data

            if self.counter == 1:
                self.buf_m[j] = torch.zeros_like(d_p)
                self.buf_v[j] = torch.zeros_like(d_p)
            
            """
            Implement the Adam optimizer below
            buf_m is m_t in the algorithm
            buf_v is v_t in the algorithm
            self.counter keeps track of the time
            """
            #############################################################Your code here################################            

            #############################################################################################
            
        self.counter += 1



            
        