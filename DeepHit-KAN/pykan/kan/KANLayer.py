import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask


class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):
        ''''
        El constructor lo ponemos para definir los atributos que va a tener una capa y preparar las funciones internas, es decir, definir una capa. 
        Cuando llamen a KANLayer(...) entonces se invoca al constructor
        '''
        super(KANLayer, self).__init__() #estamos llamando al constructor de la clase padre de nn.Module, para personalizar nuestra KANLayer y la llamamos self
        # size 
        self.out_dim = out_dim #número de características de salida 
        self.in_dim = in_dim # número de características de entrada 
        self.num = num #número de intervalos = grid
        self.k = k #orden del spline 
        #se van a crear tantas splines como conexiones input-output 

        #para crear la spline necesitaremos: un rango de entrada, la cuadrícula (para saber cuántos puntos tendremos), la extensión (para manejar bien los bordes) 
        # y la función inicial para convertirla luego en coefs spline

        #Un spline es una función suave por tramos y en cada tramo la función es un polinomio de k grado. Luego, en los bordes son derivables para que sean suaves


        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1) #se crea la cuadrícula de puntos donde definiremos el spline 
        grid = extend_grid(grid, k_extend=k) 
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k)) #sacamos los coeficientes que, al aplicar la spline, obtendremos el noise como resultado
        #lo de nn.Parameter define que se puede entrenar y cambiar durante el entrenamiento

        if sparse_init: #definimos las máscaras para ver qué conexiones entre entrada-salida están activas
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False) #el requires_grad hace que no se entrene
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable) #esto es lo que se le multiplica al SiLU
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # lo que se multiplica al spline 
        self.base_fun = base_fun #la función base de toda la vida, SiLU por ejemplo 

        #teoría:
        #yj​=i∑​[scale_baseij​⋅b(xi​)+scale_spij​⋅sij​(xi​)]
        
        self.grid_eps = grid_eps #esto explica el grado de adaptación de las cuadrículas a las funciones
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x): #con esta función vemos cómo se transforma una entrada x en una salida y -> ejecutamos output=model(x) se ejecuta el forward

        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim) #esto lo necesitamos para interpretar, ya que se guardan valores intermedios e iniciales de xi
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k) #aplica el spline al pasar los coefs (QUE ACTUALIZAMOS EN LA SIGUIENTE FUNCION) 
        #a una curva con k orden y grid espacios a los x. 
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y #aplica la formula teórica: yj​=i∑​[scale_baseij​⋅b(xi​)+scale_spij​⋅sij​(xi​)]
        y = self.mask[None,:,:] * y #aplicamos la máscara para ver qué conexiones están apagadas y cuáles no 
        
        postacts = y.clone().permute(0,2,1) #guardamos el resultado completo (base+spline) por cada conexión xi->yj
            
        y = torch.sum(y, dim=1) #yj es la sima de todas las entradas y sus combinaciones de base y spline
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'): #aquí lo que vamso a hacer es ordenar un poco los nodos para calcular las funciones spline donde estén los datos, 
        #y no en el rango que le hemos dicho 

        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0] #ordenamos los datos -> dónde están???
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k) #evaluamos el spline actual con los datos ordenados 
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            margin = 0.00
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        
        self.grid.data = extend_grid(grid, k_extend=self.k)
        #print('x_pos 2', x_pos.shape)
        #print('y_eval 2', y_eval.shape)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):

        batch = x.shape[0]
        
        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k

        def get_grid(num_interval):
            x_pos = parent.grid[:,parent.k:-parent.k]
            #print('x_pos', x_pos)
            sp2 = KANLayer(in_dim=1, out_dim=self.in_dim,k=1,num=x_pos.shape[1]-1,scale_base_mu=0.0, scale_base_sigma=0.0).to(x.device)

            #print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            #print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim), x_pos.permute(1,0).unsqueeze(dim=2), sp2.grid[:,:], k=1).permute(1,0,2)
            shp = sp2_coef.shape
            #sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            #print('sp2_coef',sp2_coef)
            #print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1,1,self.num+1).to(self.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1,0)
            #print('c', grid)
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id): #esto es útil para el pruning, ya que nos extrae una subred de la red normal, quedándonos solo con las entradas y salidas relevantes 
        #in_id son los in seleccionados, lo mismo los outputs, y con ellos creamos la spb
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id] #solo las filas del grid correspondientes a esos inputs seleccionados
        spb.coef.data = self.coef[in_id][:,out_id] #de la matriz de coeficientes solo los que corresponden a input-output conexiones
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'): #cambia filas o columnas de orden, para poder nse, agrupar salidas o entradas y visualizarlo mejor, pero vaya, que no tiene na 

        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()
            #cambiamos tb todos los elementos, el grid, coef, todo para que sea coherente 
            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

