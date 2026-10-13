import torch
import torch.nn as nn


# ============================================================
# Fully-connected subnetwork used by the original DeepHit MLP
# ============================================================
class FCNet(nn.Module):
    """
    PyTorch equivalent of utils_network.create_FCNet from the
    original DeepHit repository.

    Important details preserved:
    - Xavier initialization
    - same hidden width in all layers
    - activation also on the final layer of each subnetwork
    - dropout only between layers (not after the final subnetwork layer)
    - when num_layers == 1, there is no intermediate dropout
    """

    def __init__(self, input_dim, num_layers, h_dim, output_dim,
                 active_fn, keep_prob):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.dropout = nn.Dropout(p=1.0 - keep_prob)
        self.activation = self._get_activation(active_fn)

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, h_dim))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(h_dim, h_dim))

            layers.append(nn.Linear(h_dim, output_dim))

        self.layers = nn.ModuleList(layers)

        self._initialize_weights()

    @staticmethod
    def _get_activation(active_fn):
        if active_fn == "relu":
            return nn.ReLU()
        elif active_fn == "elu":
            return nn.ELU()
        elif active_fn == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                f"Unsupported activation function: {active_fn}. "
                "Use 'relu', 'elu' or 'tanh'."
            )

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Original create_FCNet:
        # final layer also uses o_fn=active_fn in DeepHit.
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)

            # Dropout only after non-final layers.
            if i < len(self.layers) - 1:
                x = self.dropout(x)

        return x


# ============================================================
# DeepHit-MLP
# ============================================================
class Model_DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(Model_DeepHit, self).__init__()

        # ----------------------------
        # Input dimensions
        # ----------------------------
        self.x_dim = input_dims["x_dim"]
        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        # ----------------------------
        # MLP hyperparameters
        # ----------------------------
        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]

        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]

        self.active_fn = network_settings["active_fn"]
        self.keep_prob = network_settings["keep_prob"]

        # Kept for reproducibility / compatibility with KAN pipeline.
        self.seed = network_settings.get("seed", 1)

        # ----------------------------
        # Shared MLP
        # Equivalent to original:
        # create_FCNet(x, num_layers_shared,
        #              h_dim_shared, active_fn,
        #              h_dim_shared, active_fn, ...)
        # ----------------------------
        self.shared_mlp = FCNet(
            input_dim=self.x_dim,
            num_layers=self.num_layers_shared,
            h_dim=self.h_dim_shared,
            output_dim=self.h_dim_shared,
            active_fn=self.active_fn,
            keep_prob=self.keep_prob,
        )

        # Residual/skip concatenation from original DeepHit:
        # h = concat([x, shared_out])
        h_dim_concat = self.x_dim + self.h_dim_shared

        # ----------------------------
        # Cause-specific MLPs
        # One independent MLP per competing event
        # ----------------------------
        self.cs_mlps = nn.ModuleList([
            FCNet(
                input_dim=h_dim_concat,
                num_layers=self.num_layers_CS,
                h_dim=self.h_dim_CS,
                output_dim=self.h_dim_CS,
                active_fn=self.active_fn,
                keep_prob=self.keep_prob,
            )
            for _ in range(self.num_Event)
        ])

        # Original DeepHit applies one more dropout after concatenating
        # the cause-specific representations.
        self.output_dropout = nn.Dropout(p=1.0 - self.keep_prob)

        # Final fully-connected layer:
        # E * h_dim_CS -> E * num_Category
        self.output_layer = nn.Linear(
            self.num_Event * self.h_dim_CS,
            self.num_Event * self.num_Category
        )

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # ========================================================
    # Forward
    # ========================================================
    def forward(self, x):
        # Shared representation
        shared_out = self.shared_mlp(x)

        # Same residual concatenation as original DeepHit
        h = torch.cat([x, shared_out], dim=1)

        # Cause-specific representations
        out = []

        for cs_mlp in self.cs_mlps:
            out.append(cs_mlp(h))

        # [B, E, h_dim_CS]
        out = torch.stack(out, dim=1)

        # [B, E * h_dim_CS]
        out = out.reshape(out.size(0), -1)

        # Dropout before final output layer, as in original DeepHit
        out = self.output_dropout(out)

        # [B, E * T]
        out = self.output_layer(out)

        # Joint probability distribution P(T=t, K=k | x)
        # One common softmax over event x time.
        out = torch.softmax(out, dim=1)

        # [B, E, T]
        out = out.reshape(
            -1,
            self.num_Event,
            self.num_Category
        )

        return out

    # ========================================================
    # Loss 1 -- Log-likelihood
    # SAME DeepHit loss used in the final KAN pipeline
    # ========================================================
    @staticmethod
    def loss_log_likelihood(y_pred, t, k, fc_mask1, fc_mask2):
        eps = 1e-8

        # 1 = observed event, 0 = censored
        I_1 = (k > 0).float()

        # Observed event:
        # probability mass at the observed (event, time)
        tmp1 = torch.sum(fc_mask1 * y_pred, dim=2)
        tmp1 = torch.sum(tmp1, dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + eps)

        # Censored:
        # fc_mask1 already contains the valid probability mass
        # after the censoring time for censored subjects.
        tmp2 = torch.sum(fc_mask1 * y_pred, dim=2)
        tmp2 = torch.sum(tmp2, dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + eps)

        return -torch.mean(tmp1 + tmp2)

    # ========================================================
    # Loss 2 -- Ranking loss
    # SAME implementation/protocol as DeepHit-KAN
    # ========================================================
    @staticmethod
    def loss_ranking(y_pred, t, k, fc_mask2, num_Event):
        sigma1 = 0.1
        B = t.size(0)

        eta_all = []

        for e in range(num_Event):
            # Indicator: subject experienced event e
            I_2 = (k.squeeze() == (e + 1)).float()
            I_2_diag = torch.diag(I_2)

            # Event-specific joint probabilities [B, T]
            tmp_e = y_pred[:, e, :]

            # R_ij = risk of subject i evaluated at time of subject j
            R = tmp_e @ fc_mask2.T

            diag_R = torch.diag(R).view(B, 1)

            # Match the TensorFlow DeepHit formulation
            R = diag_R.T - R
            R = R.T

            # T_ij = 1 when t_i < t_j
            T = (t < t.T).float()

            # Keep only comparable rows whose subject i
            # experienced event e
            T = I_2_diag @ T

            eta_e = torch.mean(
                T * torch.exp(-R / sigma1),
                dim=1,
                keepdim=True
            )

            eta_all.append(eta_e)

        eta = torch.cat(eta_all, dim=1)
        eta = torch.mean(eta, dim=1, keepdim=True)

        return torch.sum(eta)

    # ========================================================
    # Loss 3 -- Calibration loss
    # gamma = 0 in our experiment, but kept for compatibility
    # ========================================================
    @staticmethod
    def loss_calibration(y_pred, k, fc_mask2, num_Event):
        eta_all = []

        for e in range(num_Event):
            I_2 = (k.squeeze() == (e + 1)).float()

            tmp_e = y_pred[:, e, :]

            r = torch.sum(
                tmp_e * fc_mask2,
                dim=0
            )

            tmp_eta = torch.mean(
                (r - I_2.unsqueeze(1)) ** 2,
                dim=1,
                keepdim=True
            )

            eta_all.append(tmp_eta)

        eta = torch.cat(eta_all, dim=1)
        eta = torch.mean(eta, dim=1, keepdim=True)

        return torch.sum(eta)

    # ========================================================
    # Regularization
    # Original DeepHit:
    # - TensorFlow-style L2 = 0.5 * 1e-4 * sum(W^2) on internal FC weights
    # - L1 = 1e-4 on final output weights
    # Biases are not regularized.
    # ========================================================
    def regularization_loss(self, l1=1e-4, l2=1e-4):
        reg = torch.zeros(
            (),
            device=self.output_layer.weight.device,
            dtype=self.output_layer.weight.dtype
        )

        # Shared MLP weights: L2
        for layer in self.shared_mlp.layers:
            reg = reg + 0.5 * l2 * torch.sum(layer.weight ** 2)

        # Cause-specific MLP weights: L2
        for cs_mlp in self.cs_mlps:
            for layer in cs_mlp.layers:
                reg = reg + 0.5 * l2 * torch.sum(layer.weight ** 2)

        # Output weights: L1
        reg = reg + l1 * torch.sum(
            torch.abs(self.output_layer.weight)
        )

        return reg

    # ========================================================
    # Total DeepHit loss
    # ========================================================
    def compute_loss(
        self,
        y_pred,
        t,
        k,
        fc_mask1,
        fc_mask2,
        alpha,
        beta,
        gamma
    ):
        L1 = self.loss_log_likelihood(
            y_pred,
            t,
            k,
            fc_mask1,
            fc_mask2
        )

        L2 = self.loss_ranking(
            y_pred,
            t,
            k,
            fc_mask2,
            self.num_Event
        )

        # gamma = 0 in our protocol, but retain the term
        # to keep exactly the same interface as DeepHit-KAN.
        L3 = self.loss_calibration(
            y_pred,
            k,
            fc_mask2,
            self.num_Event
        )

        return alpha * L1 + beta * L2 + gamma * L3

    # ========================================================
    # Cost without parameter update
    # ========================================================
    def get_cost(
        self,
        x_mb,
        t_mb,
        k_mb,
        m1_mb,
        m2_mb,
        alpha,
        beta,
        gamma
    ):
        self.eval()

        with torch.no_grad():
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

        return loss.item()

    # ========================================================
    # One optimization step
    # ========================================================
    def train_step(
        self,
        x_mb,
        t_mb,
        k_mb,
        m1_mb,
        m2_mb,
        alpha,
        beta,
        gamma,
        optimizer
    ):
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

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected: {loss.item()}"
            )

        loss.backward()

        for name, param in self.named_parameters():
            if (
                param.grad is not None
                and not torch.isfinite(param.grad).all()
            ):
                raise RuntimeError(
                    f"Non-finite gradient detected in {name}"
                )

        optimizer.step()

        return loss.item()

    # ========================================================
    # Prediction
    # ========================================================
    def predict(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)

        return out
