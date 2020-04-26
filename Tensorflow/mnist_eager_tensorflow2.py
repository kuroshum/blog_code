import tensorflow as tf
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


#--------------------------------------------------------------------------------
# mnistデータをロードし、説明変数と目的変数を返す
def load_mnist_data():
    # mnistデータをロード
    mnist = fetch_openml('mnist_784', version=1,)

    # 画像データ　784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...], ... ]
    xData = mnist.data.astype(np.float32)
    
    # 0-1に正規化する
    xData /= 255 

    # ラベルデータ70000
    yData = mnist.target.astype(np.int32) 

    return xData, yData
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# 分類モデル
class ClassifierModel(tf.keras.Model):
    
    # モデルのレイヤー定義
    # args : 
    #   output_dim : 出力結果の次元  
    def __init__(self, output_dim, **kwargs):
        super(ClassifierModel, self).__init__(**kwargs)
        # wx + bを出力
        # 活性化関数はsoftmax
        self.fc1 = tf.keras.layers.Dense(output_dim, activation='softmax')
    
    # モデルの実行
    def call(self, x_t):
        pred = self.fc1(x_t)
    
        return pred
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# クロスエントロピー損失関数
# args : 
#   model : 定義したモデル(Classifier model)
#   x_t : 学習 or テスト データ
#   y_t : 教師データ
def cross_entropy_loss(model, x_t, y_t):
    # categorical_crossentropyの入力データはone-hot表現なので
    # y_t, model(x_t)はどちらも(データ数, 10)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.cast(y_t,tf.float32), tf.cast(model(x_t),tf.float32)))
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# 1エポックで行う学習(ミニバッチ)
# args : 
#   model : 定義したモデル(Classifier model)
#   dataset : 学習データと教師データをtf.data.Datasetのコレクションに変換したやつ
#   loss_list : lossを記録する
#   optimizer : 最適化関数
@tf.function
def fit(model, dataset, loss_list, optimizer=tf.keras.optimizers.Adam(), training=False):
    # 損失関数から勾配を計算する
    #loss_and_grads = tfe.implicit_value_and_gradients(cross_entropy_loss)
    
    # ミニバッチ
    for _, (x, y) in enumerate(dataset):
        
        # 学習時は損失と勾配を計算, テスト時は損失のみ計算
        if training:
            
            #　自動微分のAPI
            with tf.GradientTape() as tape:
                # cross-entropy-lossの計算
                loss = cross_entropy_loss(model, x, y)
            
            # 損失から勾配を計算
            grad = tape.gradient(loss, sources=model.trainable_variables)
            
            # 損失と勾配を計算
            #loss, grads = loss_and_grads(model, x, y)

            # 勾配を更新
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        else:
            loss = cross_entropy_loss(model, x, y)

        # 損失を記録
        loss_list.update_state(loss)
    
    # ミニバッチで記録した損失の平均を返す
    return loss_list.result()
#--------------------------------------------------------------------------------



if __name__ == "__main__":

    # mnistデータをロード
    xData, yData = load_mnist_data()

    # 目的変数のカテゴリー数(次元)を設定
    label_num = 10

    # ラベルデータをone-hot表現に変換
    yData = np.squeeze(np.identity(label_num)[yData])

    # 目的変数のカテゴリー数(次元)を取得
    yDim = yData.shape[1]

    # 学習データとテストデータに分割
    xData_train, xData_test, yData_train, yData_test =  train_test_split(xData, yData, test_size=0.2, random_state=42)
    
    # モデルの定義
    classifier_model = ClassifierModel(yDim)

    # イテレーションの反復回数
    nIte = 100

    # テスト実行の割合(test_rate回につき1回)
    test_rate = 10

    # バッチサイズ
    BATCH_SIZE = 64


    # 学習データ・テストデータの数
    num_data_train = xData_train.shape[0]
    num_data_test = xData_test.shape[0]

    # numpyデータをtensorに
    xData_train = tf.convert_to_tensor(xData_train)
    xData_test = tf.convert_to_tensor(xData_test)
    yData_train = tf.convert_to_tensor(yData_train)
    yData_test = tf.convert_to_tensor(yData_test)
    
    # lossの記録
    train_loss_list = tf.keras.metrics.Mean()
    test_loss_list = tf.keras.metrics.Mean()

    # 入力データをdatasetに
    # シャッフルとバッチを作成
    dataset_train = tf.data.Dataset.from_tensor_slices((xData_train, yData_train))
    dataset_train = dataset_train.shuffle(buffer_size=num_data_train)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    dataset_test = tf.data.Dataset.from_tensor_slices((xData_test, yData_test))
    dataset_test = dataset_train.shuffle(buffer_size=num_data_test)
    dataset_test = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    time_record = 0

    #--------------------------------------------------------------------------------
    # 学習とテストの反復
    for ite in range(nIte):
        
        start = time.time()
        # ミニバッチ法で学習
        # 各バッチデータから計算した損失の平均
        train_loss_mean = fit(classifier_model, dataset_train, train_loss_list, training=True)

        end = time.time() - start

        time_record += end

        print('#{0}, train loss : {1}'.format(ite, train_loss_mean))

        # test_rate回に1回、テスト時の損失を計算
        if ite % test_rate==0:
            test_loss_mean = fit(classifier_model, dataset_test, test_loss_list)
            print('#{0}, test loss : {1}'.format(ite, test_loss_mean))
        
    #--------------------------------------------------------------------------------
    print(time_record)