import tensorflow as tf
import numpy as np
from tqdm import tqdm 

# 学習時のMetricとして使用する異常度。各ミニバッチの異常度の合計を返す。
@tf.function
def Anomaly_score(x, model_x):    
    anomaly_score = tf.reduce_mean(tf.abs(x - model_x), axis=[1, 2, 3])

    return anomaly_score


def test(model, data_loader):
    """
    model         :モデル
    data_loader     :データセット->tf.data
    =========================================================
    return:
    各データごとの異常度を要素とする一次元配列(ndarray)
    """
    all_anomaly_score  =[]
    for image_batch in tqdm(data_loader):
        ## 異常度の計算 ##
        training=False
        model_x = model(image_batch, training=training)
        anomaly_score = Anomaly_score(image_batch, model_x) # -> array
        all_anomaly_score += anomaly_score.numpy().tolist()

    return np.array(all_anomaly_score)

