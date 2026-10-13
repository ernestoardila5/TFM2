# deephit_kan_tf2.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== 1) Capa KAN (placeholder) =====
# Sustituye el interior por tu implementación real de KAN
class KANLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # Ejemplo mínimo: comportamiento "dense" (luego lo cambias por tu KAN real)
        self.W = self.add_weight(
            name="W", shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="b", shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        z = tf.linalg.matmul(inputs, self.W) + self.b
        return self.activation(z) if self.activation else z

# ===== 2) Bloques constructores MLP/KAN =====
def dense_block(x, num_layers, hidden_dim, activation, dropout_rate, use_kan=False, name_prefix="blk"):
    for i in range(num_layers):
        if use_kan:
            x = KANLayer(hidden_dim, activation=activation, name=f"{name_prefix}_kan_{i}")(x)
        else:
            x = layers.Dense(hidden_dim, activation=activation, name=f"{name_prefix}_dense_{i}")(x)
        if dropout_rate and dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop_{i}")(x)
    return x

# ===== 3) Modelo DeepHit+ (single o competing risks) =====
class DeepHitKAN(keras.Model):
    def __init__(self, x_dim_list, num_events, num_categories,
                 num_layers_shared, h_dim_shared,
                 num_layers_cs, h_dim_cs,
                 dropout_shared=0.0, dropout_cs=0.0,
                 activation="relu",
                 use_kan=False,
                 **kwargs):
        """
        x_dim_list: lista con las particiones de covariables: [x_common, x_e1, x_e2, ...]
        num_events: E
        num_categories: T (tiempos discretos)
        num_layers_shared, h_dim_shared: profundidad/ancho del bloque compartido
        num_layers_cs, h_dim_cs: listas de longitud E (por evento)
        use_kan: True → usar KANLayer; False → Dense
        """
        super().__init__(**kwargs)
        self.x_dim_list = x_dim_list
        self.num_events = num_events
        self.num_categories = num_categories
        self.num_layers_shared = num_layers_shared
        self.h_dim_shared = h_dim_shared
        self.num_layers_cs = num_layers_cs
        self.h_dim_cs = h_dim_cs
        self.dropout_shared = dropout_shared
        self.dropout_cs = dropout_cs
        self.activation = activation
        self.use_kan = use_kan

        # Definimos sub-módulos como Layers/Sequential para registro de pesos
        self.shared_net = keras.Sequential(name="shared_block")
        # shared: sobre las x comunes
        for i in range(num_layers_shared):
            if use_kan:
                self.shared_net.add(KANLayer(h_dim_shared, activation=activation, name=f"shared_kan_{i}"))
            else:
                self.shared_net.add(layers.Dense(h_dim_shared, activation=activation, name=f"shared_dense_{i}"))
            if dropout_shared and dropout_shared > 0.0:
                self.shared_net.add(layers.Dropout(dropout_shared, name=f"shared_drop_{i}"))

        # cause-specific nets
        self.cs_nets = []
        for e in range(num_events):
            net = keras.Sequential(name=f"cs_block_e{e+1}")
            for i in range(num_layers_cs[e]):
                if use_kan:
                    net.add(KANLayer(h_dim_cs[e], activation=activation, name=f"cs{e+1}_kan_{i}"))
                else:
                    net.add(layers.Dense(h_dim_cs[e], activation=activation, name=f"cs{e+1}_dense_{i}"))
                if dropout_cs and dropout_cs > 0.0:
                    net.add(layers.Dropout(dropout_cs, name=f"cs{e+1}_drop_{i}"))
            self.cs_nets.append(net)

        # salida final E*T con softmax
        self.out_dense = layers.Dense(num_events * num_categories, activation=None, name="out_linear")
        self.out_softmax = layers.Activation("softmax", name="out_softmax")

    def call(self, x, training=False):
        # Split según x_dim_list: x = [X_common | X_e1 | X_e2 | ...]
        splits = tf.split(x, self.x_dim_list, axis=-1)
        x_common = splits[0]                    # shape [B, x_dim_common]
        x_event_specific = splits[1:]           # lista E de tensores

        # Shared features
        shared_out = self.shared_net(x_common, training=training)

        # Por evento: concat([x_e, shared_out]) → cs_net → cs_out
        cs_outputs = []
        for e in range(self.num_events):
            inp_e = tf.concat([x_event_specific[e], shared_out], axis=-1)
            cs_out = self.cs_nets[e](inp_e, training=training)
            cs_outputs.append(cs_out)

        # Concat de todos los eventos
        h = tf.concat(cs_outputs, axis=-1)

        # Capa final + softmax + reshape a [B, E, T]
        logits = self.out_dense(h)                             # [B, E*T]
        probs = self.out_softmax(logits)                       # softmax en E*T
        probs = tf.reshape(probs, [-1, self.num_events, self.num_categories])
        return probs  # == self.out en 1.x
