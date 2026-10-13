import torch
import torch.nn as nn
import numpy as np
from spline import *
from utils import sparse_mask


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
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        
        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    # def update_grid_from_samples(self, x, mode='sample'):
    #     '''
    #     update grid from samples
        
    #     Args:
    #     -----
    #         x : 2D torch.float
    #             inputs, shape (number of samples, input dimension)
            
    #     Returns:
    #     --------
    #         None
        
    #     Example
    #     -------
    #     >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
    #     >>> print(model.grid.data)
    #     >>> x = torch.linspace(-3,3,steps=100)[:,None]
    #     >>> model.update_grid_from_samples(x)
    #     >>> print(model.grid.data)
    #     '''
        
    #     batch = x.shape[0]
    #     #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
    #     x_pos = torch.sort(x, dim=0)[0]

    #     print("\n=== GRID UPDATE DIAGNOSTIC ===")
    #     print("x shape:", x.shape)
    #     print("x finite:", torch.isfinite(x).all().item())
    #     print("grid finite:", torch.isfinite(self.grid).all().item())
    #     print("coef finite:", torch.isfinite(self.coef).all().item())
    #     print("coef NaN:", torch.isnan(self.coef).sum().item())
    #     print("coef Inf:", torch.isinf(self.coef).sum().item())

    #     y_eval = coef2curve(
    #         x_pos,
    #         self.grid,
    #         self.coef,
    #         self.k
    #     )

    #     print("y_eval finite:", torch.isfinite(y_eval).all().item())
    #     print("y_eval NaN:", torch.isnan(y_eval).sum().item())
    #     print("y_eval Inf:", torch.isinf(y_eval).sum().item())

    #     if not torch.isfinite(y_eval).all():
    #         raise RuntimeError(
    #             "y_eval became non-finite BEFORE curve2coef/lstsq."
    #         )
    #     # y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
    #     num_interval = self.grid.shape[1] - 1 - 2*self.k
        
    #     def get_grid(num_interval):
    #         ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
    #         grid_adaptive = x_pos[ids, :].permute(1,0)
    #         margin = 0.00
    #         h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
    #         grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
    #         grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
    #         return grid
        
        
    #     grid = get_grid(num_interval)
        
    #     if mode == 'grid':
    #         sample_grid = get_grid(2*num_interval)
    #         x_pos = sample_grid.permute(1,0)
    #         y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        
    #     self.grid.data = extend_grid(grid, k_extend=self.k)
    #     #print('x_pos 2', x_pos.shape)
    #     #print('y_eval 2', y_eval.shape)
    #     self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    # def update_grid_from_samples(self, x, mode='sample'):
    #     '''
    #     Update grid from samples.

    #     Además incluye comprobaciones diagnósticas para detectar
    #     exactamente dónde aparecen valores NaN/Inf.
    #     '''

    #     batch = x.shape[0]

    #     x_pos = torch.sort(x, dim=0)[0]

    #     # ========================================================
    #     # DIAGNÓSTICO 1:
    #     # estado ANTES de actualizar el grid
    #     # ========================================================

    #     print("\n=== GRID UPDATE DIAGNOSTIC ===")

    #     print("x shape:", x.shape)

    #     print("x finite:",
    #         torch.isfinite(x).all().item())

    #     print("grid finite:",
    #         torch.isfinite(self.grid).all().item())

    #     print("coef finite:",
    #         torch.isfinite(self.coef).all().item())

    #     print("coef NaN:",
    #         torch.isnan(self.coef).sum().item())

    #     print("coef Inf:",
    #         torch.isinf(self.coef).sum().item())

    #     # Si coef YA está corrupto, paramos aquí.
    #     if not torch.isfinite(self.coef).all():

    #         raise RuntimeError(
    #             "KAN coefficients are already non-finite "
    #             "BEFORE grid update."
    #         )


    #     # ========================================================
    #     # Evaluar la spline ACTUAL antes de mover el grid
    #     # ========================================================

    #     y_eval = coef2curve(
    #         x_pos,
    #         self.grid,
    #         self.coef,
    #         self.k
    #     )

    #     print("y_eval finite:",
    #         torch.isfinite(y_eval).all().item())

    #     print("y_eval NaN:",
    #         torch.isnan(y_eval).sum().item())

    #     print("y_eval Inf:",
    #         torch.isinf(y_eval).sum().item())

    #     # Si coef era correcto pero y_eval no lo es,
    #     # el problema está en coef2curve/B_batch.
    #     if not torch.isfinite(y_eval).all():

    #         raise RuntimeError(
    #             "y_eval became non-finite BEFORE "
    #             "curve2coef/lstsq."
    #         )


    #     # ========================================================
    #     # Construcción del nuevo grid
    #     # ========================================================

    #     num_interval = (
    #         self.grid.shape[1]
    #         - 1
    #         - 2 * self.k
    #     )

    #     def get_grid(num_interval):

    #         ids = [
    #             int(batch / num_interval * i)
    #             for i in range(num_interval)
    #         ] + [-1]

    #         grid_adaptive = (
    #             x_pos[ids, :]
    #             .permute(1, 0)
    #         )

    #         margin = 0.00

    #         h = (
    #             grid_adaptive[:, [-1]]
    #             - grid_adaptive[:, [0]]
    #             + 2 * margin
    #         ) / num_interval

    #         grid_uniform = (
    #             grid_adaptive[:, [0]]
    #             - margin
    #             + h
    #             * torch.arange(
    #                 num_interval + 1
    #             )[None, :].to(x.device)
    #         )

    #         grid = (
    #             self.grid_eps * grid_uniform
    #             + (1 - self.grid_eps)
    #             * grid_adaptive
    #         )

    #         return grid


    #     grid = get_grid(num_interval)


    #     # ========================================================
    #     # Modo grid
    #     # ========================================================

    #     if mode == 'grid':

    #         sample_grid = get_grid(
    #             2 * num_interval
    #         )

    #         x_pos = sample_grid.permute(1, 0)

    #         y_eval = coef2curve(
    #             x_pos,
    #             self.grid,
    #             self.coef,
    #             self.k
    #         )

    #         if not torch.isfinite(y_eval).all():

    #             raise RuntimeError(
    #                 "y_eval became non-finite "
    #                 "in mode='grid'."
    #             )


    #     # ========================================================
    #     # Extender y sustituir grid
    #     # ========================================================

    #     new_grid = extend_grid(
    #         grid,
    #         k_extend=self.k
    #     )

    #     print("new grid finite:",
    #         torch.isfinite(new_grid).all().item())

    #     print("new grid shape:",
    #         new_grid.shape)

    #     grid_diff = (
    #         new_grid[:, 1:]
    #         - new_grid[:, :-1]
    #     )

    #     print("new grid min diff:",
    #         grid_diff.min().item())

    #     print("new grid zero diffs:",
    #         (grid_diff == 0).sum().item())

    #     print("new grid negative diffs:",
    #         (grid_diff < 0).sum().item())

    #     if not torch.isfinite(new_grid).all():

    #         raise RuntimeError(
    #             "New KAN grid contains NaN/Inf."
    #         )


    #     # ========================================================
    #     # Calcular coeficientes correspondientes al NUEVO grid
    #     # ========================================================

    #     new_coef = curve2coef(
    #         x_pos,
    #         y_eval,
    #         new_grid,
    #         self.k
    #     )


    #     # ========================================================
    #     # DIAGNÓSTICO 2:
    #     # comprobar resultado de curve2coef
    #     # ========================================================

    #     print("new coef finite:",
    #         torch.isfinite(new_coef).all().item())

    #     print("new coef NaN:",
    #         torch.isnan(new_coef).sum().item())

    #     print("new coef Inf:",
    #         torch.isinf(new_coef).sum().item())


    #     if not torch.isfinite(new_coef).all():

    #         raise RuntimeError(
    #             "curve2coef produced non-finite "
    #             "spline coefficients."
    #         )


    #     # ========================================================
    #     # SOLO si todo es correcto actualizamos el layer
    #     # ========================================================

    #     self.grid.data = new_grid

    #     self.coef.data = new_coef

    #     print("GRID UPDATE OK")
# ahora a las 10:54
    # def update_grid_from_samples(self, x, mode='sample'):
    #     batch = x.shape[0]
    #     x_pos = torch.sort(x, dim=0)[0]

    #     # --------------------------------------------------
    #     # Comprobaciones antes de actualizar
    #     # --------------------------------------------------
    #     if not torch.isfinite(x_pos).all():
    #         raise RuntimeError(
    #             "Non-finite activations received by KAN grid update."
    #         )

    #     if not torch.isfinite(self.grid).all():
    #         raise RuntimeError(
    #             "KAN grid is non-finite before grid update."
    #         )

    #     if not torch.isfinite(self.coef).all():
    #         raise RuntimeError(
    #             "KAN coefficients are non-finite before grid update."
    #         )

    #     old_grid = self.grid.data.clone()
    #     old_coef = self.coef.data.clone()

    #     # Evaluamos la función aprendida ANTES de mover el grid
    #     y_eval = coef2curve(
    #         x_pos,
    #         old_grid,
    #         old_coef,
    #         self.k
    #     )

    #     if not torch.isfinite(y_eval).all():
    #         raise RuntimeError(
    #             "KAN spline evaluation became non-finite before grid update."
    #         )

    #     num_interval = (
    #         self.grid.shape[1]
    #         - 1
    #         - 2 * self.k
    #     )

    #     # --------------------------------------------------
    #     # Detectar dimensiones degeneradas
    #     # --------------------------------------------------
    #     x_min = x_pos[0, :]
    #     x_max = x_pos[-1, :]
    #     x_range = x_max - x_min

    #     # Tolerancia relativa al tamaño de cada activación.
    #     # Evita intentar adaptar un grid a una variable
    #     # esencialmente constante.
    #     scale = torch.maximum(
    #         torch.maximum(torch.abs(x_min), torch.abs(x_max)),
    #         torch.ones_like(x_range)
    #     )

    #     range_tol = (
    #         100.0
    #         * torch.finfo(x.dtype).eps
    #         * scale
    #     )

    #     valid_dim = x_range > range_tol

    #     # --------------------------------------------------
    #     # Construir nuevo grid
    #     # --------------------------------------------------
    #     ids = [
    #         int(batch / num_interval * i)
    #         for i in range(num_interval)
    #     ] + [-1]

    #     grid_adaptive = (
    #         x_pos[ids, :]
    #         .permute(1, 0)
    #     )

    #     h = (
    #         grid_adaptive[:, [-1]]
    #         - grid_adaptive[:, [0]]
    #     ) / num_interval

    #     grid_uniform = (
    #         grid_adaptive[:, [0]]
    #         + h
    #         * torch.arange(
    #             num_interval + 1,
    #             device=x.device,
    #             dtype=x.dtype
    #         )[None, :]
    #     )

    #     grid = (
    #         self.grid_eps * grid_uniform
    #         + (1.0 - self.grid_eps) * grid_adaptive
    #     )

    #     new_grid = extend_grid(
    #         grid,
    #         k_extend=self.k
    #     )

    #     # --------------------------------------------------
    #     # Las dimensiones degeneradas NO se actualizan
    #     # --------------------------------------------------
    #     new_grid[~valid_dim] = old_grid[~valid_dim]

    #     # --------------------------------------------------
    #     # Comprobar separación de knots
    #     # --------------------------------------------------
    #     grid_diff = new_grid[:, 1:] - new_grid[:, :-1]

    #     bad_grid_dim = (
    #         (~torch.isfinite(new_grid).all(dim=1))
    #         | (grid_diff <= 0).any(dim=1)
    #     )

    #     # Si una dimensión que parecía válida acaba generando
    #     # un grid degenerado, también conservamos el anterior.
    #     if bad_grid_dim.any():
    #         new_grid[bad_grid_dim] = old_grid[bad_grid_dim]

    #     # Dimensiones que realmente vamos a actualizar
    #     update_dim = valid_dim & (~bad_grid_dim)

    #     # --------------------------------------------------
    #     # Recalcular coeficientes
    #     # --------------------------------------------------
    #     candidate_coef = old_coef.clone()

    #     if update_dim.any():
    #         coef_updated = curve2coef(
    #             x_pos[:, update_dim],
    #             y_eval[:, update_dim, :],
    #             new_grid[update_dim],
    #             self.k
    #         )

    #         if not torch.isfinite(coef_updated).all():
    #             raise RuntimeError(
    #                 "curve2coef produced non-finite coefficients "
    #                 "during KAN grid update."
    #             )

    #         candidate_coef[update_dim] = coef_updated

    #     # Commit solo después de comprobar todo
    #     self.grid.data.copy_(new_grid)
    #     self.coef.data.copy_(candidate_coef)

    def update_grid_from_samples(self, x, mode='sample'):
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]

        print("\n=== GRID UPDATE DIAGNOSTIC ===")
        print("x shape:", x.shape)
        print("x finite:", torch.isfinite(x_pos).all().item())
        print("grid finite:", torch.isfinite(self.grid).all().item())
        print("coef finite:", torch.isfinite(self.coef).all().item())
        print("coef NaN:", torch.isnan(self.coef).sum().item())
        print("coef Inf:", torch.isinf(self.coef).sum().item())

        # ========================================================
        # 1. Comprobar estado ANTES de actualizar
        # ========================================================

        if not torch.isfinite(x_pos).all():
            raise RuntimeError(
                "Non-finite activations received by KAN grid update."
            )

        if not torch.isfinite(self.grid).all():
            raise RuntimeError(
                "KAN grid is non-finite before grid update."
            )

        if not torch.isfinite(self.coef).all():
            raise RuntimeError(
                "KAN coefficients are non-finite before grid update."
            )

        old_grid = self.grid.data.clone()
        old_coef = self.coef.data.clone()

        # ========================================================
        # 2. Evaluar spline ACTUAL antes de cambiar el grid
        # ========================================================

        y_eval = coef2curve(
            x_pos,
            old_grid,
            old_coef,
            self.k
        )

        print("y_eval finite:", torch.isfinite(y_eval).all().item())
        print("y_eval NaN:", torch.isnan(y_eval).sum().item())
        print("y_eval Inf:", torch.isinf(y_eval).sum().item())

        if not torch.isfinite(y_eval).all():
            raise RuntimeError(
                "KAN spline evaluation became non-finite "
                "before grid update."
            )

        num_interval = (
            self.grid.shape[1]
            - 1
            - 2 * self.k
        )

        # ========================================================
        # 3. Rango de las activaciones
        # ========================================================

        x_min = x_pos[0, :]
        x_max = x_pos[-1, :]
        x_range = x_max - x_min

        scale = torch.maximum(
            torch.maximum(
                torch.abs(x_min),
                torch.abs(x_max)
            ),
            torch.ones_like(x_range)
        )

        range_tol = (
            100.0
            * torch.finfo(x.dtype).eps
            * scale
        )

        valid_dim = x_range > range_tol

        print("x range min:", x_range.min().item())
        print("x range max:", x_range.max().item())
        print("zero-range dims:", (x_range == 0).sum().item())
        print(
            "tiny-range dims:",
            (~valid_dim).sum().item()
        )

        # ========================================================
        # 4. Construir nuevo grid
        # ========================================================

        ids = [
            int(batch / num_interval * i)
            for i in range(num_interval)
        ] + [-1]

        grid_adaptive = (
            x_pos[ids, :]
            .permute(1, 0)
        )

        h = (
            grid_adaptive[:, [-1]]
            - grid_adaptive[:, [0]]
        ) / num_interval

        grid_uniform = (
            grid_adaptive[:, [0]]
            + h
            * torch.arange(
                num_interval + 1,
                device=x.device,
                dtype=x.dtype
            )[None, :]
        )

        grid = (
            self.grid_eps * grid_uniform
            + (1.0 - self.grid_eps)
            * grid_adaptive
        )

        # Mantener compatibilidad con mode='grid'
        if mode == 'grid':
            sample_num_interval = 2 * num_interval

            sample_ids = [
                int(batch / sample_num_interval * i)
                for i in range(sample_num_interval)
            ] + [-1]

            sample_grid_adaptive = (
                x_pos[sample_ids, :]
                .permute(1, 0)
            )

            sample_h = (
                sample_grid_adaptive[:, [-1]]
                - sample_grid_adaptive[:, [0]]
            ) / sample_num_interval

            sample_grid_uniform = (
                sample_grid_adaptive[:, [0]]
                + sample_h
                * torch.arange(
                    sample_num_interval + 1,
                    device=x.device,
                    dtype=x.dtype
                )[None, :]
            )

            sample_grid = (
                self.grid_eps * sample_grid_uniform
                + (1.0 - self.grid_eps)
                * sample_grid_adaptive
            )

            x_pos = sample_grid.permute(1, 0)

            y_eval = coef2curve(
                x_pos,
                old_grid,
                old_coef,
                self.k
            )

            if not torch.isfinite(y_eval).all():
                raise RuntimeError(
                    "y_eval became non-finite in mode='grid'."
                )

        new_grid = extend_grid(
            grid,
            k_extend=self.k
        )

        # ========================================================
        # 5. Las dimensiones de rango degenerado NO se actualizan
        # ========================================================

        new_grid[~valid_dim] = old_grid[~valid_dim]

        # ========================================================
        # 6. Comprobar separación REAL de los knots
        # ========================================================

        grid_diff = (
            new_grid[:, 1:]
            - new_grid[:, :-1]
        )

        # IMPORTANTE:
        # no buscamos solamente diff <= 0.
        # En el fallo real vimos ~1e-41, que es > 0 pero
        # numéricamente degenerado.
        grid_scale = torch.maximum(
            new_grid.abs().amax(dim=1),
            torch.ones(
                new_grid.shape[0],
                device=new_grid.device,
                dtype=new_grid.dtype
            )
        )

        min_allowed_diff = (
            100.0
            * torch.finfo(new_grid.dtype).eps
            * grid_scale
        )

        tiny_diff_mask = (
            grid_diff
            <= min_allowed_diff[:, None]
        )

        bad_grid_dim = (
            (~torch.isfinite(new_grid).all(dim=1))
            | tiny_diff_mask.any(dim=1)
        )

        print("new grid finite:",
            torch.isfinite(new_grid).all().item())

        print("new grid shape:", new_grid.shape)

        print("new grid min diff:",
            grid_diff.min().item())

        print("new grid zero diffs:",
            (grid_diff == 0).sum().item())

        print("new grid negative diffs:",
            (grid_diff < 0).sum().item())

        print("new grid tiny diffs:",
            tiny_diff_mask.sum().item())

        rejected_range = ~valid_dim
        rejected_grid = valid_dim & bad_grid_dim
        update_dim = valid_dim & (~bad_grid_dim)

        print("dimensions rejected by range:",
            rejected_range.sum().item(), "/", new_grid.shape[0])

        print("dimensions rejected by grid:",
            rejected_grid.sum().item(), "/", new_grid.shape[0])

        print("dimensions updated:",
            update_dim.sum().item(), "/", new_grid.shape[0])

        print("total:",
            (rejected_range | rejected_grid | update_dim).sum().item(),
            "/", new_grid.shape[0])

        # ========================================================
        # 7. Rechazar grids degenerados
        # ========================================================

        if bad_grid_dim.any():
            new_grid[bad_grid_dim] = old_grid[bad_grid_dim]

        update_dim = (
            valid_dim
            & (~bad_grid_dim)
        )

        print(
            "dimensions updated:",
            update_dim.sum().item(),
            "/",
            new_grid.shape[0]
        )

        # ========================================================
        # 8. Recalcular coef SOLO para dimensiones seguras
        # ========================================================

        candidate_coef = old_coef.clone()

        if update_dim.any():

            coef_updated = curve2coef(
                x_pos[:, update_dim],
                y_eval[:, update_dim, :],
                new_grid[update_dim],
                self.k
            )

            print(
                "updated coef finite:",
                torch.isfinite(coef_updated).all().item()
            )

            print(
                "updated coef NaN:",
                torch.isnan(coef_updated).sum().item()
            )

            print(
                "updated coef Inf:",
                torch.isinf(coef_updated).sum().item()
            )

            if not torch.isfinite(coef_updated).all():
                raise RuntimeError(
                    "curve2coef produced non-finite coefficients "
                    "during KAN grid update."
                )

            candidate_coef[update_dim] = coef_updated

        # ========================================================
        # 9. Comprobación FINAL antes del commit
        # ========================================================

        print(
            "final grid finite:",
            torch.isfinite(new_grid).all().item()
        )

        print(
            "final coef finite:",
            torch.isfinite(candidate_coef).all().item()
        )

        if not torch.isfinite(new_grid).all():
            raise RuntimeError(
                "Final KAN grid is non-finite."
            )

        if not torch.isfinite(candidate_coef).all():
            raise RuntimeError(
                "Final KAN coefficients are non-finite."
            )

        # ========================================================
        # 10. Commit SOLO cuando todo está correcto
        # ========================================================

        self.grid.data.copy_(new_grid)
        self.coef.data.copy_(candidate_coef)

        print("GRID UPDATE OK")
        print("==============================\n")

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        
        '''
        # based on samples
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid'''
        
        #print('p', parent.grid)
        # based on interpolating parent grid
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

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

