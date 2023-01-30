import tensorflow as tf
tf.keras.backend.set_floatx('float64') # for numerical stability
from tensorflow.keras import layers

class DAIN_Layer(layers.Layer):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=2, **kwargs):
        super(DAIN_Layer, self).__init__(**kwargs)
        
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr
        
        self.mean_layer = layers.Dense(input_dim, bias_initializer='zeros', kernel_initializer='identity')
        self.scaling_layer = layers.Dense(input_dim, bias_initializer='zeros', kernel_initializer='identity')
        self.gating_layer = layers.Dense(input_dim, bias_initializer='zeros', kernel_initializer='glorot_uniform')
        
        self.eps = 1e-8
        
    def call(self, x):
        # Expecting (n_samples, dim, n_feature_vectors)

        def adaptive_avg(x):
            avg = tf.reduce_mean(x, axis=2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = tf.expand_dims(adaptive_avg, axis=-1)
            x = x - adaptive_avg
            return x

        def adaptive_std(x):
            std = tf.reduce_mean(x ** 2, axis=2)
            std = tf.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = tf.where(tf.less_equal(adaptive_std, self.eps), tf.ones_like(adaptive_std), adaptive_std)
            adaptive_std = tf.expand_dims(adaptive_std, axis=-1)
            x = x / adaptive_std
            return x

        def gating(x):
            avg = tf.reduce_mean(x, axis=2)
            avg = self.gating_layer(avg)
            gate = tf.sigmoid(avg)
            gate = tf.expand_dims(gate, axis=-1)
            x = x * gate
            return x
        
        if self.mode == None:
            pass
        
        elif self.mode == 'avg':
            avg = tf.reduce_mean(x, axis=2)
            avg = tf.expand_dims(avg, axis=-1)
            x = x - avg
            
        elif self.mode == 'adaptive_avg':
            x = adaptive_avg(x)

        elif self.mode == 'adaptive_std':
            x = adaptive_avg(x)
            x = adaptive_std(x)
            
        elif self.mode == 'full':
            x = adaptive_avg(x)
            x = adaptive_std(x)
            x = gating(x)
        
        else:
            assert False
            
        return x
