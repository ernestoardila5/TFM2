import torch
import torch.nn as nn
import numpy as np
import sympy
from utils import *



class Symbolic_KANLayer(nn.Module):

    def __init__(self, in_dim=3, out_dim=2, device='cpu'):

        super(Symbolic_KANLayer, self).__init__() #se construye una capa simbólica
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False) #conocer qué conexiones están activas
        # torch
        self.funs = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        self.funs_avoid_singularity = [[lambda x, y_th: ((), x*0.) for i in range(self.in_dim)] for j in range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?
        
        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
        # c*f(a*x+b)+d
        
        self.device = device
        self.to(device)
        
    def to(self, device):
        '''
        move to device
        '''
        super(Symbolic_KANLayer, self).to(device)
        self.device = device    
        return self
    
    def forward(self, x, singularity_avoiding=False, y_th=10.): #calcula la salida simbólica de cada capa para un input x

        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j,i,2]*self.funs_avoid_singularity[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1], torch.tensor(y_th))[1]+self.affine[j,i,3]
                else:
                    xij = self.affine[j,i,2]*self.funs[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1])+self.affine[j,i,3]
                postacts_.append(self.mask[j][i]*xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2,1,0,3)[:,:,:,0]
        y = torch.sum(postacts, dim=2)
        
        return y, postacts #devuelve la salida total y las activaciones por conexión para análisis posterior
        
        
    def get_subset(self, in_id, out_id): #se crea una copia de la capa pero conservando solo un subset, para el pruning. 

        sbb = Symbolic_KANLayer(self.in_dim, self.out_dim, device=self.device) #mismos que la capa original 
        sbb.in_dim = len(in_id) #luego ajustamos a los índices que queremos
        sbb.out_dim = len(out_id)
        sbb.mask.data = self.mask.data[out_id][:,in_id]
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id] #por cada nueorna de salida j, copia las funciones que vienen desde las entradas i seleccionadas
        sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:,in_id]
        return sbb
    
    
    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10,10), b_range=(-10,10), verbose=True):
        #se asgina una función simbólica concreta a la conexión entre la neurona i y la j
        if isinstance(fun_name,str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            
            if x == None or y == None:
                #initialzie from just fun
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                if random == False:
                    self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
                else:
                    self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
                return None
            else:
                #initialize from x & y and fun
                params, r2 = fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=self.device)
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
            else:
                self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
            return None
        
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        '''
        with torch.no_grad():
            def swap_list_(data, i1, i2, mode='in'):

                if mode == 'in':
                    for j in range(self.out_dim):
                        data[j][i1], data[j][i2] = data[j][i2], data[j][i1]

                elif mode == 'out':
                    data[i1], data[i2] = data[i2], data[i1] 

            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

                elif mode == 'out':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()

            swap_list_(self.funs_name,i1,i2,mode)
            swap_list_(self.funs_sympy,i1,i2,mode)
            swap_list_(self.funs_avoid_singularity,i1,i2,mode)
            swap_(self.affine.data,i1,i2,mode)
            swap_(self.mask.data,i1,i2,mode)
