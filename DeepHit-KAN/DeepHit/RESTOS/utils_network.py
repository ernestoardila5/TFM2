import tensorflow as tf

class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units, grid_size=5, activation=None, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Inicialización de los pesos y sesgos
        self.W = self.add_weight(
            name="W", shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="b", shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs, training=False):
        z = tf.linalg.matmul(inputs, self.W) + self.b
        
        # Aplicar la activación si está definida
        if self.activation:
            z = self.activation(z)

        # Si estamos en entrenamiento, aplicar dropout
        if training:
            z = tf.nn.dropout(z, rate=0.5)  # Aquí puedes cambiar la tasa de dropout según lo necesites.

        return z



### FEEDFORWARD NETWORK
def create_KAN(inputs, num_layers, h_dim, h_fn, o_dim, o_fn, w_init, keep_prob=1.0, w_reg=None):
    '''
        GOAL             : Create a KAN with different specifications 
        inputs (tensor)  : input tensor
        num_layers       : number of layers in FCNet
        h_dim  (int)     : number of hidden units
        h_fn             : activation function for hidden layers (default: tf.nn.relu)
        o_dim  (int)     : number of output units
        o_fn             : activation function for output layers (defalut: None)
        w_init           : initialization for weight matrix (defalut: Xavier)
        keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
    '''
    # default active functions (hidden: relu, out: None)
    if h_fn is None:
        h_fn = tf.nn.relu
    if o_fn is None:
        o_fn = None


    # default initialization functions (weight: Xavier, bias: None)
    if w_init is None:
        w_init = tf.contrib.layers.xavier_initializer() # Xavier initialization

    for layer in range(num_layers):
        if num_layers == 1:
            out = KANLayer(inputs, o_dim, activation_fn=o_fn, weights_initializer=w_init, weights_regularizer=w_reg)
        else:
            if layer == 0:
                h = KANLayer(inputs, h_dim, activation=h_fn, weights_initializer=w_init, weights_regularizer=w_reg)
                if not keep_prob is None:
                    h = tf.nn.dropout(h, rate=1-keep_prob)  # Aquí aplicamos dropout

            elif layer > 0 and layer != (num_layers - 1):
                h = KANLayer(h, h_dim, activation=h_fn, weights_initializer=w_init, weights_regularizer=w_reg)
                if not keep_prob is None:
                    h = tf.nn.dropout(h, rate=1-keep_prob)

            else: # layer == num_layers-1 (the last layer)
                out = KANLayer(h, o_dim, activation_fn=o_fn, weights_initializer=w_init, weights_regularizer=w_reg)

    return out