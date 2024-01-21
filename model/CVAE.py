from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
import os

# サンプリング層
class Sampling(layers.Layer):

    # 順伝播処理
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # 正規分布に従う乱数を生成（デフォルトは標準正規分布）
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.sqrt(z_var) * epsilon

# エンコーダ
class Encoder(layers.Layer):

    def __init__(
            self, 
            encoder_conv_filters, 
            encoder_conv_kernel_size,
            encoder_conv_strides,
            latent_dim=32, 
            use_batch_norm = False,
            use_dropout= False,
            name="encoder", 
            **kwargs):
        """
        params
        -------------------
        latent_dim:潜在変数の次元数
        """
        super().__init__(name=name, **kwargs)

        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.n_layers_encoder = len(encoder_conv_filters)
        self.shape_beforeFlatten = None

        self.list_convLayer = []
        self.list_batchNorm = []
        for i in range(self.n_layers_encoder):
            conv_layer = layers.Conv2D(
            filters = self.encoder_conv_filters[i]
            , kernel_size = self.encoder_conv_kernel_size[i]
            , strides = self.encoder_conv_strides[i]
            , padding = 'same'
            , name = 'encoder_conv_' + str(i)
            )
            self.list_convLayer.append(conv_layer)
            self.list_batchNorm.append(layers.BatchNormalization(name='encoder_batchNorm_' + str(i)))

        # self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(rate = 0.25)

        self.flatten = layers.Flatten()

        self.dense_mu = layers.Dense(self.latent_dim, name='mu')
        self.dense_var = layers.Dense(self.latent_dim, name='var')

        self.sampling = Sampling(name='encoder_output')

    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape, name='layer_in')
        # buildメソッドを実行させる
        self(x, training=False)
        functionalModel = tf.keras.Model(
            inputs=[x],
            outputs=self.call(x, training=False),
            name="functionalModel"
        )
        
        return functionalModel
    
    # 畳み込み後のサイズを計算し、デコーダで使用する
    def build(self, input_shape):
        # input_shape[0]がNoneになっているため、1にする
        # print("build", (1, *input_shape[1:]))
        x = tf.zeros((1, *input_shape[1:]))
        for conv, BatchNorm in zip(self.list_convLayer, self.list_batchNorm):
            x = conv(x)
            if self.use_batch_norm:
                x = BatchNorm(x, training=True)
            x = self.leaky_relu(x)
            if self.use_dropout:
                x = self.dropout(x, training=True)

        # デコーダで使用
        self.shape_beforeFlatten = tf.shape(x)[1:].numpy()


    # 順伝播処理
    def call(self, inputs, training=False):
        # print('encoder call')
        # print("trainnig:", training)
         
        x = inputs    
        # 訓練時にはbatch_normおよびdropoutを通す
        for conv, BatchNorm in zip(self.list_convLayer, self.list_batchNorm):
            x = conv(x)
            # tf.print("conv", x)
            if self.use_batch_norm:
                x = BatchNorm(x, training=training)
            x = self.leaky_relu(x)
            if self.use_dropout:
                x = self.dropout(x, training=training)

        x = self.flatten(x)
        # tf.print(x.shape)
        z_mean = self.dense_mu(x)
        z_var = tf.math.softplus(self.dense_var(x)) # 分散が非負になるようにsoftplusに通す
        z = self.sampling((z_mean, z_var))

        return z_mean, z_var, z
    

# デコーダ
class Decoder(layers.Layer):

    def __init__(
            self, 
            encoder, 
            decoder_conv_t_filters,
            decoder_conv_t_kernel_size,
            decoder_conv_t_strides,
            # original_dim, 
            use_batch_norm = False,
            use_dropout= False,
            name="decoder", 
            **kwargs):
        """
        parameters
        -------------
        encoder:エンコーダオブジェクト
        """

        super().__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_decoder = len(decoder_conv_t_filters)

        # _difine_inputShape()で定義する
        self.dense_input = None
        self.reshape = None

        self.list_convTransLayer = []
        self.list_batchNorm = []
        for i in range(self.n_layers_decoder):
            convTrans_layer = layers.Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
            )
            self.list_convTransLayer.append(convTrans_layer)
            self.list_batchNorm.append(layers.BatchNormalization(name='Decoder_batchNorm_' + str(i)))

        # self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(rate = 0.25)
        self.sigmoid = layers.Activation("sigmoid")

    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape, name='layer_in')
        # buildメソッドを実行させる
        self(x, training=False)
        functionalModel = tf.keras.Model(
            inputs=[x],
            outputs=self.call(x, training=False),
            name="functionalModel"
        )
        
        return functionalModel

    # call()時に一度だけ呼び出される
    def build(self, input_shape):
        """
        エンコーダがflattenを呼び出す前のサイズに応じてconv2DTransに渡すshpaeを決定する
        """
        # tf.print("build", self.encoder.shape_beforeFlatten)
        shape_beforeFlatten = self.encoder.shape_beforeFlatten

        self.dense_input = layers.Dense(np.prod(shape_beforeFlatten))
        # tf.print("reshape", shape_beforeFlatten)
        self.reshape = layers.Reshape(shape_beforeFlatten) # output shape => (batch_size) + target_shape


    # 順伝播処理
    def call(self, inputs, training=False):
        x = inputs
        x = self.dense_input(x)
        x = self.reshape(x)

        for idx, (conv_t, BatchNorm) in enumerate(zip(self.list_convTransLayer, self.list_batchNorm)):
            x = conv_t(x)
            # tf.print("con_vt", x.shape)
            if idx < self.n_layers_decoder -1 :
                if self.use_batch_norm:
                    x = BatchNorm(x, training=training)
                x = self.leaky_relu(x)
                
                if self.use_dropout:
                    x = self.dropout(x, training=training)
            else:
                x = self.sigmoid(x)

        return x


class ConvolutionalVariationalAutoEncoder(tf.keras.Model):
    """エンコーダとデコーダを結合する"""

    def __init__(
        self,
        # original_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        latent_dim=32,
        use_batch_norm = False,
        use_dropout= False,
        name="autoencoder",
        **kwargs
    ):
        """
        params
        -------------------
        original_dim:入力データの次元
        latent_dim:潜在変数の次元数
        """
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        # エンコーダ用の初期値
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides

        # デコーダ用の初期値
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

        self.encoder = Encoder(
            encoder_conv_filters, 
            encoder_conv_kernel_size,
            encoder_conv_strides,
            latent_dim=self.latent_dim, 
            use_batch_norm = self.use_batch_norm ,
            use_dropout= self.use_dropout
            )
        self.decoder = Decoder(
            self.encoder,
            decoder_conv_t_filters,
            decoder_conv_t_kernel_size,
            decoder_conv_t_strides,
            use_batch_norm = self.use_batch_norm ,
            use_dropout= self.use_dropout
            )

    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape, name='layer_in')
        functionalModel = tf.keras.Model(
            inputs=[x],
            outputs=self.call(x, training=False),
            name="functionalModel"
        )
        
        return functionalModel

 
    def call(self, inputs, training=False):
        z_mean, z_var, z = self.encoder(inputs, training)
        reconstructed = self.decoder(z, training)

        # KL divergence regularization loss
        tf.debugging.assert_all_finite(tf.math.log(z_var+1e-7) , 'tf.math.log(z_var) must be finite') # expがNaNまたはInfになった場合エラーを投げる
        kl_loss = 1 + tf.math.log(z_var+1e-7) - tf.square(z_mean) - z_var # => (batch_size, latent_dim)    

        # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1)) # => [] axis=1方向に加算してからバッチ方向に平均 
        kl_loss = tf.reduce_mean(kl_loss) #=> [] 上のやり方よりもこちらのやり方(一度にkl_lossの平均を算出する)のほうが再構成誤差が小さくなった。

        kl_loss *= -0.5
        self.add_loss(kl_loss)

        return reconstructed