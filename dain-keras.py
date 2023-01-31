import tensorflow as tf
from tensorflow.keras import layers

class DAIN_Layer(layers.Layer):
    def __init__(self, mode='full', input_dim=5, **kwargs):
        super(DAIN_Layer, self).__init__(**kwargs)
        
        self.mode = mode
        # self.mean_lr = mean_lr #0.00001
        # self.gate_lr = gate_lr #0.00001
        # self.scale_lr = scale_lr #0.00001
        
        self.mean_layer = layers.Dense(input_dim, kernel_initializer='identity', use_bias=False)
        self.scaling_layer = layers.Dense(input_dim, kernel_initializer='identity', use_bias=False)
        self.gating_layer = layers.Dense(input_dim, bias_initializer='zeros', kernel_initializer='glorot_uniform')
        
        self.eps = 1e-8
        
    def call(self, x):
        # Expecting (batch_size, window_len, num_features)
        x = tf.transpose(x, perm=[0, 2, 1])
        def adaptive_avg(x):
            # Expecting (batch_size, num_features, window_len)
            avg = tf.reduce_mean(x, axis=2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = tf.expand_dims(adaptive_avg, axis=-1)
            x = x - adaptive_avg
            return x

        def adaptive_std(x):
            # Expecting (batch_size, num_features, window_len)
            std = tf.reduce_mean(x ** 2, axis=2)
            std = tf.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = tf.where(tf.less_equal(adaptive_std, self.eps), tf.ones_like(adaptive_std), adaptive_std)
            adaptive_std = tf.expand_dims(adaptive_std, axis=-1)
            x = x / adaptive_std
            return x

        def gating(x):
            # Expecting (batch_size, num_features, window_len)
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
            
        return tf.transpose(x, perm=[0, 2, 1])


if __name__ == '__main__':
    # Test
    import numpy as np
    # Create an example input batch of size (32, 10, 5)
    # The batch should have varying ranges of values for each batch
    example_batch = np.random.random((32, 10, 5))
    example_batch = example_batch * np.random.randint(1, 10, size=(32, 1, 1))
    example_batch = example_batch.astype(np.float32)
    
    dain_layer = DAIN_Layer(5)
    out_batch = dain_layer(example_batch)
    print(out_batch.shape)

    
