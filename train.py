from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import tensorflow as tf

class Train_Pipeline:
    def __init__(self, summary_writer):
        """
        summary_writer:Tensor BoardのSummaryWriterオブジェクト
        """
        self.summary_writer = summary_writer

        # optimizer
        self.optimizer = None

        # loss
        self.criterion = tf.keras.losses.MeanSquaredError()

        # metrics
        self.total_loss_metric = tf.keras.metrics.Mean()
        self.reconstructed_loss_metric = tf.keras.metrics.Mean()
        self.val_reconstructed_loss_metric = tf.keras.metrics.Mean()

        # 損失関数における再構成誤差項をKL_lossの何倍にするか
        self.r_loss_factor = None

        # モデル 
        self.VAE = None
        
    def train(self, VAE, optimizer, r_loss_factor, EPOCHS, train_loader, val_loader, ckpt, manager):
        """
        VAE           :変分オートエンコーダ
        optimizer     :optimizer
        r_loss_factor :損失関数における再構成誤差項をKL_lossの何倍にするか
        train_loader    :訓練データセット->tf.data
        val_loader      :バリデーションデータセット->tf.data
        ckpt            :チェックポイントのオブジェクト, ckptはepoch回数を保持するstep属性を持つ必要がある ->tf.train.Checkpoint
        manager         :チェックポイントの管理 ->tf.train.CheckpointManager 
        ----------------------------------------
        return
        lossとmetricをそれぞれ各エポックごとに平均したリストをvalueとする辞書
        """
        # モデル
        self.VAE = VAE

        # optimizer
        self.optimizer = optimizer

        self.r_loss_factor = r_loss_factor

        best_val = float('inf')
        list_loss, list_mae, list_val_mae = [], [], []
        tf.summary.trace_on(graph=True)
        for epoch in range(EPOCHS):
            # プログレスバー 
            num_iteration = int(train_loader.cardinality())
            with tqdm(total=num_iteration, unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch+1}/{EPOCHS}]")

                for step, image_batch in enumerate(train_loader):
                    """ 訓練 """
                    total_loss, reconstructed_loss = self.train_step(image_batch)
                    
                    if epoch==0 and step==0:  
                        with self.summary_writer.as_default():
                            tf.summary.trace_export(
                            name="my_func_trace",
                            step=0)  

                    self.total_loss_metric.update_state(total_loss) # lossの平均値を算出
                    self.reconstructed_loss_metric.update_state(reconstructed_loss)# 再構成誤差の平均を算出

                    """ 生成画像の保存 """
                    if step == 0:                        
                        ## 再構成画像（D(E(x))） ##
                        save_image_size_for_recon = 4
                        images = image_batch[:save_image_size_for_recon]
                        D_E_x = self.VAE(images, training=False)
                        diff_images = tf.abs(images - D_E_x)
                        comparison = tf.concat([images , D_E_x, diff_images], axis=0) # バッチ方向に連結
                        # 各行がimages, G(E(x)), diff_imagesになるように描画・保存
                        figure_D_E_x = Train_Pipeline.image_grid(comparison, nrow=3, ncol=save_image_size_for_recon)   
                        image_D_E_x = Train_Pipeline.plot_to_image(figure_D_E_x)

                     # プログレスバーの更新
                    pbar.set_postfix({"total_loss":self.total_loss_metric.result().numpy(), "reconstructed_loss":self.reconstructed_loss_metric.result().numpy()})
                    pbar.update(1)

            """ エポック終了時の処理 """
            # バリデーションデータの異常度算出および再構成画像の保存
            for val_step, val_image_batch in enumerate(val_loader):
                # バリデーションデータに対する再構成誤差算出
                reconstructed = self.VAE(val_image_batch, training=False)
                reconstructed_loss = self.criterion(val_image_batch, reconstructed)
                self.val_reconstructed_loss_metric.update_state(reconstructed_loss)

            # lossとmetricsの算出・表示
            loss_mean = self.total_loss_metric.result()
            reconstructed_loss_mean = self.reconstructed_loss_metric.result()
            val_reconstructed_loss_mean = self.val_reconstructed_loss_metric.result()
            print(f"{epoch}/{EPOCHS}  Epoch total_loss: {loss_mean:.3f}  reconstructed_loss: {reconstructed_loss_mean:.3f} val_reconstructed_loss: {val_reconstructed_loss_mean:.3f}")       

            # チェックポイントの更新
            ckpt.step.assign_add(1)
            if val_reconstructed_loss_mean < best_val:
                best_val = val_reconstructed_loss_mean
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            self.total_loss_metric.reset_states()
            self.reconstructed_loss_metric.reset_states()
            self.val_reconstructed_loss_metric.reset_states()
            
            list_loss.append(loss_mean.numpy())
            list_mae.append(reconstructed_loss_mean.numpy()) 
            list_val_mae.append(val_reconstructed_loss_mean.numpy())    
            with self.summary_writer.as_default():
                tf.summary.scalar('total_loss', loss_mean, step=epoch)
                tf.summary.scalar('reconstructed_loss', reconstructed_loss_mean, step=epoch)
                tf.summary.scalar('val_reconstructed_loss', val_reconstructed_loss_mean, step=epoch)
                
                for weight in self.VAE.trainable_variables:
                    tf.summary.histogram(name=weight.name, data=weight, step=epoch)

                tf.summary.image("D(E(x))", image_D_E_x, step=epoch)
            
        history = {"loss":list_loss, "metrics":list_mae, "val_metrics":list_val_mae}

        return history

    @tf.function
    def train_step(self, image_batch):
        with tf.GradientTape() as tape:
            training=True
            with tf.name_scope('FP'):
                reconstructed = self.VAE(image_batch, training=training)

            with tf.name_scope('Loss'):
                reconstructed_loss = self.criterion(image_batch, reconstructed) 
                total_loss = self.r_loss_factor*reconstructed_loss# 再構成誤差
                total_loss += sum(self.VAE.losses) # call時にadd_loss()によって計算したKL_lossを加算 

        with tf.name_scope('BP'):
            
            grads = tape.gradient(total_loss, self.VAE.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.VAE.trainable_weights))

        # tf.print("total_loss", tf.shape(total_loss))

        return total_loss, reconstructed_loss
    
    # 画像保存用のヘルパー
    @staticmethod
    def plot_to_image(figure):
        """matplotlibのプロットをPNGに変換する。
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # notebookに表示されているグラフを消すと同時にリソースを開放する.
        plt.close(figure)
        
        # buf内の「ファイルポインタ」（現在の読み書き位置）を指定した位置に移動しておく(0=先頭)
        buf.seek(0) 
        # バフに保存したPNG画像をtensor imageに変換
        image = tf.io.decode_png(buf.getvalue(), channels=3) # channels: カラーチャネルの数。4の場合、RGBA画像を出力
        image = tf.expand_dims(image, axis=0)# バッチの次元を追加
        
        return image

    @staticmethod
    def image_grid(images, nrow=4, ncol=4):
        """
        グリッド画像のfigを返す
        images:->[batch:h:w:c]
        """
        fig = plt.figure(figsize=(4, 4))

        for i in range(images.shape[0]):
            plt.subplot(nrow,  ncol, i+1)
            plt.axis('off')
            plt.grid(False)
            plt.imshow(images[i, :, :, 0] * 255)# -1～1に正規化されている画像を0～255に変換して表示
            
        return fig
