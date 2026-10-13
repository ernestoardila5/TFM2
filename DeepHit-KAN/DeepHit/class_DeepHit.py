
import torch
import torch.nn as nn
import torch.nn.functional as F
from MultKAN import MultKAN



class Model_DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(Model_DeepHit, self).__init__()

        # INPUT DIMENSIONS
        self.x_dim        = input_dims['x_dim'] #numero de features
        self.num_Event    = input_dims['num_Event'] #nº de causas (estamos en eventos competitivos)
        self.num_Category = input_dims['num_Category'] #tiempo discretizado

        # NETWORK HYPER-PARAMETERS
        # self.h_dim_shared      = network_settings['h_dim_shared'] #tamaño del tronco común
        # self.h_dim_CS          = network_settings['h_dim_CS'] #nº neuronas del tronco específico por causa
        # self.num_layers_shared = network_settings['num_layers_shared'] #número de layers comunes
        # self.num_layers_CS     = network_settings['num_layers_CS'] #número de nodos específicos

        # NETWORK HYPER-PARAMETERS
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']

        self.sum_nodes_shared = network_settings['sum_nodes_shared']
        self.mult_nodes_shared = network_settings['mult_nodes_shared']

        self.sum_nodes_CS = network_settings['sum_nodes_CS']
        self.mult_nodes_CS = network_settings['mult_nodes_CS']

        # Hiperparámetros KAN
        self.grid = network_settings['grid']
        self.k = network_settings['k']
        self.mult_arity = network_settings['mult_arity']
        self.noise_scale = network_settings['noise_scale']
        self.grid_eps = network_settings['grid_eps']
        self.grid_range = network_settings['grid_range']
        self.sparse_init = network_settings['sparse_init']
        self.keep_prob = network_settings['keep_prob']
        self.dropout = nn.Dropout(p=1.0 - self.keep_prob)

        self.seed = network_settings.get('seed',1)

        # ----------------------------
        #  SHARED KAN (igual que Shared FCNet original)
        # ----------------------------
        # width_shared = [self.x_dim] + [self.h_dim_shared] * self.num_layers_shared #nº features por número de nodos por el nº de hidden layers
        width_shared = (
            [[self.x_dim, 0]]
            +
            [[self.sum_nodes_shared, self.mult_nodes_shared]
            for _ in range(self.num_layers_shared)]
        )

        self.shared_kan = MultKAN(
            width=width_shared,

            grid=self.grid,
            k=self.k,
            mult_arity=self.mult_arity,

            noise_scale=self.noise_scale,

            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            base_fun='silu',

            symbolic_enabled=False,
            affine_trainable=False,

            grid_eps=self.grid_eps,
            grid_range=self.grid_range,

            sp_trainable=True,
            sb_trainable=True,

            seed=self.seed,

            save_act=True,
            sparse_init=self.sparse_init,

            auto_save=False,
            first_init=True,
            ckpt_path='./model',
            state_id=0,
            round=0,
            device='cpu'
        )

        # Tamaño real de h = x + shared_out
        shared_out_dim = (
            self.sum_nodes_shared
            + self.mult_nodes_shared
        )

        h_dim_concat = (
            self.x_dim
            + shared_out_dim
        )

        # ----------------------------
        #  CAUSE-SPECIFIC KANs (igual que FCNet con concat original)
        # ----------------------------
        width_CS = (
            [[h_dim_concat, 0]]
            +
            [[self.sum_nodes_CS, self.mult_nodes_CS]
            for _ in range(self.num_layers_CS)]
        ) #cada nodo al final es una coordenada del vector que define a ese sample

        self.cs_kans = nn.ModuleList([
            MultKAN(
                width=width_CS,

                grid=self.grid,
                k=self.k,
                mult_arity=self.mult_arity,

                noise_scale=self.noise_scale,

                scale_base_mu=0.0,
                scale_base_sigma=1.0,
                base_fun='silu',

                symbolic_enabled=False,
                affine_trainable=False,

                grid_eps=self.grid_eps,
                grid_range=self.grid_range,

                sp_trainable=True,
                sb_trainable=True,

                seed=self.seed + e +1,

                save_act=True,
                sparse_init=self.sparse_init,

                auto_save=False,
                first_init=True,
                ckpt_path='./model',
                state_id=0,
                round=0,
                device='cpu'
            )
            for e in range(self.num_Event)
        ])

        # Final fully connected → igual que en DeepHit original
        self.cs_out_dim = (
            self.sum_nodes_CS
            +
            self.mult_nodes_CS
        )

        self.output_layer = nn.Linear(
            self.num_Event * self.cs_out_dim,
            self.num_Event * self.num_Category
        )

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)


    # ----------------------------
    #  FORWARD (fiel al código TensorFlow)
    # ----------------------------
    def forward(self, x):

        shared_out = self.shared_kan(x)

        h = torch.cat(
            [x, shared_out],
            dim=1
        )

        out = []

        for cs_kan in self.cs_kans:
            out.append(cs_kan(h))

        out = torch.stack(out, dim=1)

        out = out.view(
            out.size(0),
            -1
        )

        out = self.dropout(out)

        out = self.output_layer(out)

        # DeepHit original:
        # softmax conjunto sobre evento × tiempo
        out = torch.softmax(
            out,
            dim=1
        )

        out = out.view(
            -1,
            self.num_Event,
            self.num_Category
        )

        return out

    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    @staticmethod
    def loss_log_likelihood(y_pred, t, k, fc_mask1, fc_mask2):
        eps = 1e-8
        I_1 = (k > 0).float()

        # Evento observado
        tmp1 = torch.sum(fc_mask1 * y_pred, dim=2)
        tmp1 = torch.sum(tmp1, dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + eps)

        # Censura
        tmp2 = torch.sum(fc_mask1 * y_pred, dim=2)
        tmp2 = torch.sum(tmp2, dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + eps)

        loss = -torch.mean(tmp1 + tmp2)
        return loss



    ### LOSS-FUNCTION 2 -- Ranking loss
    @staticmethod
    def loss_ranking(y_pred, t, k, fc_mask2, num_Event):
        #esto fuerza a que quien falla antes, tenga mayor riesgo acumulado antes CIF
        sigma1 = 0.1
        eps = 1e-8
        B = t.size(0)

        eta_all = []

        for e in range(num_Event):
            # Indicador de evento e (k == e+1)
            I_2 = (k.squeeze() == (e + 1)).float()  # [B]
            I_2_diag = torch.diag(I_2)              # [B, B]

            # Probabilidad conjunta evento e
            tmp_e = y_pred[:, e, :]                 # [B, T]

            # Riesgo acumulado: R_ij = r_i(T_j)
            R = tmp_e @ fc_mask2.T                  # [B, B]

            # diag_R[i] = r_i(T_i)
            diag_R = torch.diag(R).view(B, 1)       # [B, 1]

            # R_ij = r_i(T_i) - r_j(T_i)
            R = diag_R.T - R                          # [B, B]
            R = R.T                                 # igual que TF

            # T_ij = 1 si t_i < t_j
            T = (t < t.T).float()
            T = torch.relu(T)

            # Solo sujetos con evento e
            T = I_2_diag @ T

            eta_e = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta_all.append(eta_e)

        eta = torch.cat(eta_all, dim=1)             # [B, E]
        eta = torch.mean(eta, dim=1, keepdim=True)  # [B, 1]

        return torch.sum(eta)

    @torch.no_grad()
    def update_grids(self, x):

        was_training = self.training

        self.eval()

        # --------------------------------------------------
        # 1. Actualizar grid de la KAN compartida
        # --------------------------------------------------
        self.shared_kan.update_grid_from_samples(x)

        # --------------------------------------------------
        # 2. Recalcular representación shared
        #    usando el nuevo grid
        # --------------------------------------------------
        shared_out = self.shared_kan(x)

        h = torch.cat(
            [x, shared_out],
            dim=1
        )

        # --------------------------------------------------
        # 3. Actualizar grids de cada KAN cause-specific
        #    utilizando SU input real
        # --------------------------------------------------
        for cs_kan in self.cs_kans:
            cs_kan.update_grid_from_samples(h)

        if was_training:
            self.train()


    ### LOSS-FUNCTION 3 -- Calibration Loss
    @staticmethod
    def loss_calibration(y_pred, k, fc_mask2, num_Event):
        B = y_pred.size(0)
        eta_all = []

        for e in range(num_Event):
            # Indicador de evento
            I_2 = (k.squeeze() == (e + 1)).float()   # [B]

            tmp_e = y_pred[:, e, :]                  # [B, T]

            # r(t) agregado por tiempo (axis=0 en TF)
            r = torch.sum(tmp_e * fc_mask2, dim=0)  # [T]

            # Broadcast contra individuos
            tmp_eta = torch.mean((r - I_2.unsqueeze(1)) ** 2, dim=1, keepdim=True)
            eta_all.append(tmp_eta)

        eta = torch.cat(eta_all, dim=1)              # [B, E]
        eta = torch.mean(eta, dim=1, keepdim=True)   # [B, 1]

        return torch.sum(eta)

    def regularization_loss(self, l1=1e-4, l2=1e-4):
        """
        DeepHit-KAN regularization:

        - L1 on the final DeepHit output weight matrix.
        - L2 on the trainable numerical KAN parameters.

        The KAN L2 term is an adaptation of DeepHit's internal-network
        weight regularization, since a KAN has no direct FC-weight equivalent.
        """

        # --------------------------------------------------
        # DeepHit output layer:
        # L1 ONLY on the weight matrix, not on the bias
        # --------------------------------------------------
        reg = l1 * torch.sum(
            torch.abs(self.output_layer.weight)
        )

        # --------------------------------------------------
        # Numerical KAN parameters
        # --------------------------------------------------
        kan_modules = [
            self.shared_kan,
            *list(self.cs_kans)
        ]

        for kan in kan_modules:
            for layer in kan.act_fun:
                for param in layer.parameters():
                    if param.requires_grad:
                        reg = reg + 0.5* l2 * torch.sum(param ** 2)

        return reg

    def compute_loss(self, y_pred, t, k, fc_mask1, fc_mask2, alpha, beta, gamma):
        #combinamos los 3 modelos de pérdidas 
        L1 = self.loss_log_likelihood(y_pred, t, k, fc_mask1, fc_mask2)
        L2 = self.loss_ranking(y_pred, t, k, fc_mask2, self.num_Event)
        L3 = self.loss_calibration(y_pred, k, fc_mask2, self.num_Event)
        return alpha * L1 + beta * L2 + gamma * L3

    
    def get_cost(self, x_mb, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma):
        self.eval()  # eval mode (no dropout)´
        #para evaluar el modelo sin entrenar -> medimos el error sin aprender 
        with torch.no_grad():
            y_pred = self.forward(x_mb)
            loss = self.compute_loss(y_pred, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma)
            loss = loss + self.regularization_loss()
        return loss.item()


    # def train_step(self, x_mb, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma, optimizer):
    #     self.train()  # training mode (dropout activado si lo hubiera)
    #     #aquí ya sí entrenamos 
    #     optimizer.zero_grad()
    #     y_pred = self.forward(x_mb) 
    #     #se calculan predicciones y pér5didas
    #     loss = self.compute_loss(y_pred, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma)
    #     loss = loss + self.regularization_loss() #añade regularización (evitando pesos irrelevantes)
    #     loss.backward() #aquí sí se calculan los gradientes porque sí queremos aprender, en el de predicción no pq ya solo queremos usarlo
    #     optimizer.step() #cambiando pesos
    #     return loss.item() #devuelve el valor del error

    def train_step(self, x_mb, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma, optimizer):
        self.train()

        optimizer.zero_grad()

        y_pred = self.forward(x_mb)

        loss = self.compute_loss(
            y_pred,
            t_mb,
            k_mb,
            m1_mb,
            m2_mb,
            alpha,
            beta,
            gamma
        )

        loss = loss + self.regularization_loss()

        # Protección: parar inmediatamente si la loss se corrompe
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected: {loss.item()}"
            )

        loss.backward()

        # Protección: comprobar que ningún gradiente tenga NaN/Inf
        for name, param in self.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient detected in {name}"
                )

        optimizer.step()

        return loss.item()

    
    def predict(self, x): #usamos el modelo ya entrenado como modelo y nos da las predicciones
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return out


    # def predict(self, x_test, MASK, keep_prob=1.0):
    #     (m1_test, m2_test) = MASK
    #     return self.sess.run(self.out, 
    #                          feed_dict={self.x: x_test, self.rnn_mask1:m1_test, self.rnn_mask2:m2_test, self.keep_prob: keep_prob})
