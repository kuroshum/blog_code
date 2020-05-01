import tensorflow as tf
import numpy as np
import pdb
from keras.datasets import mnist
import time

# GPUの設定
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


#--------------------------------------------------------------------------------
# 分類モデルのパラメータ

# 隠れ層のノード数
HIDDEN_DIM = 128

# クラス数
CLASS_NUM = 10

# バッチサイズ
BATCH_SIZE = 64
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# mnistデータをロードし、学習データとテストデータを返す
def load_mnist_data():
    # mnistデータをロード
    (xData_train, yData_train), (xData_test, yData_test) = mnist.load_data()

    # 画像データ　データ数*784
    xData_train = xData_train.reshape([-1, 28*28]).astype(np.float32)
    xData_test = xData_test.reshape([-1, 28*28]).astype(np.float32)
    
    # 0-1に正規化する
    xData_train /= 255 
    xData_test /= 255 

    # ラベルデータ70000
    yData_train = yData_train.astype(np.int32) 
    yData_test = yData_test.astype(np.int32) 

    return (xData_train, yData_train), (xData_test, yData_test)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#mnistデータをロードし、学習データとテストデータを返す
def load_mnist_image_data():
    # mnistデータをロード
    (xData_train, yData_train), (xData_test, yData_test) = mnist.load_data()

    # 画像データ　データ数*28*28
    xData_train = xData_train.astype(np.float32)[:,:,:,np.newaxis]
    xData_test = xData_test.astype(np.float32)[:,:,:,np.newaxis]

    # 0-1に正規化する
    xData_train /= 255 
    xData_test /= 255 

    # ラベルデータ70000
    yData_train = yData_train.astype(np.int32) 
    yData_test = yData_test.astype(np.int32) 

    return (xData_train, yData_train), (xData_test, yData_test)
#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------
# 分類モデル
class ClassifierModel(tf.keras.Model):
    
    # モデルのレイヤー定義
    # args : 
    #   output_dim : 出力結果の次元  
    def __init__(self, **kwargs):
        super(ClassifierModel, self).__init__(**kwargs)
        
        # 1層目        
        # wx + bを出力
        # 活性化関数はsoftmax
        self.fc1 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        #self.dropout1 = tf.keras.layers.Dropout(keep_prob)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 2層目
        self.fc2 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        #self.dropout2 = tf.keras.layers.Dropout(keep_prob)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 3層目
        self.fc3= tf.keras.layers.Dense(CLASS_NUM, activation='softmax')
    
    # モデルの実行
    def call(self, x_t, training=False):
        
        # 1層目
        x = self.fc1(x_t)
        x = self.bn1(x, training=training)

        # 2層目
        x = self.fc2(x)
        x = self.bn2(x, training=training)

        # 3層目
        x = self.fc3(x)
    
        return x
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 分類モデル
class ClassifierCnnModel(tf.keras.Model):
    
    # モデルのレイヤー定義
    # args : 
    #   output_dim : 出力結果の次元  
    def __init__(self, data_format='channels_last', **kwargs):
        super(ClassifierCnnModel, self).__init__(**kwargs)

        # 1層目        
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, data_format=data_format, activation='relu')
        #self.bn1 = tf.keras.layers.BatchNormalization()

        #self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, data_format=data_format, activation='relu')
        #self.bn2 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()

        # 3層目
        self.fc1 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        #self.bn3 = tf.keras.layers.BatchNormalization()

        self.fc2 = tf.keras.layers.Dense(CLASS_NUM, activation='softmax')
        #self.bn4 = tf.keras.layers.BatchNormalization()
    
    # モデルの実行
    def call(self, x_t, training=tf.cast(False, tf.bool)):

        # 1層目
        x = self.conv1(x_t)
        #x = self.bn1(x, training=training)

        #x = self.conv2(x)
        #x = self.bn2(x, training=training)

        x = self.flatten(x)

        # 3層目
        x = self.fc1(x)
        #x = self.bn2(x, training=training)

        x = self.fc2(x)
    
        return x
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# クロスエントロピー損失関数
# args : 
#   model : 定義したモデル(Classifier model)
#   x_t : 学習 or テスト データ
#   y_t : 教師データ
def cross_entropy_loss(fx, y_t):
    # categorical_crossentropyの入力データはone-hot表現なので
    # y_t, model(x_t)はどちらも(データ数, 10)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.cast(y_t,tf.float32), tf.cast(fx,tf.float32)))
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
# 1エポックで行う学習(ミニバッチ)
# args : 
#   model : 定義したモデル(Classifier model)
#   dataset : 学習データと教師データをtf.data.Datasetのコレクションに変換したやつ
#   loss_list : lossを記録する
#   optimizer : 最適化関数
@tf.function
def train_step(model, dataset, loss_list, acc_list, optimizer=tf.keras.optimizers.Adam()):
    # 損失関数から勾配を計算する
    #loss_and_grads = tfe.implicit_value_and_gradients(cross_entropy_loss)
    
    # ミニバッチ
    for x, y in dataset:
            
        #　自動微分のAPI
        with tf.GradientTape() as tape:
            # cross-entropy-lossの計算
            loss = cross_entropy_loss(model(x, training=tf.cast(True, tf.bool)), y)
        
        # 損失から勾配を計算
        grad = tape.gradient(loss, sources=model.trainable_variables)
        
        # 損失と勾配を計算
        #loss, grads = loss_and_grads(model, x, y)

        # 勾配を更新
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # 損失を記録
        loss_list.update_state(loss)

        # accuracyを記録
        acc_list.update_state(y, model(x))
    
    # ミニバッチで記録した損失の平均を返す
    return loss_list.result(), acc_list.result()
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 1エポックで行うテスト(ミニバッチ)
# args : 
#   model : 定義したモデル(Classifier model)
#   dataset : 学習データと教師データをtf.data.Datasetのコレクションに変換したやつ
#   loss_list : lossを記録する
@tf.function
def test_step(model, dataset, loss_list, acc_list):

    # ミニバッチ
    for x, y in dataset:
        
        # テストの損失を計算
        loss = cross_entropy_loss(model(x), y)

        # テストの損失を記録
        loss_list.update_state(loss)

        # テストaccuracyを記録
        acc_list.update_state(y, model(x))
    
    return loss_list.result(), acc_list.result()



if __name__ == "__main__":

    # mnistデータをロード
    (xData_train, yData_train), (xData_test, yData_test) = load_mnist_image_data()

    # ラベルデータをone-hot表現に変換
    yData_train = np.squeeze(np.identity(CLASS_NUM)[yData_train])
    yData_test = np.squeeze(np.identity(CLASS_NUM)[yData_test])

    # モデルの定義
    classifier_model = ClassifierCnnModel(data_format='channels_last')

    # イテレーションの反復回数
    nIte = 10

    # テスト実行の割合(test_rate回につき1回)
    test_rate = 10

    # 学習データ・テストデータの数
    num_data_train = xData_train.shape[0]
    num_data_test = xData_test.shape[0]

    # numpyデータをtensorに
    xData_train = tf.convert_to_tensor(xData_train)
    xData_test = tf.convert_to_tensor(xData_test)
    yData_train = tf.convert_to_tensor(yData_train)
    yData_test = tf.convert_to_tensor(yData_test)
    
    # lossの記録
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    # accuracyの記録
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    # 入力データをdatasetに
    # シャッフルとバッチを作成
    # drop_reminder=True にしておけばバッチに分けたあとの余ったデータを捨てる
    dataset_train = tf.data.Dataset.from_tensor_slices((xData_train, yData_train))
    dataset_train = dataset_train.shuffle(buffer_size=num_data_train)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    dataset_test = tf.data.Dataset.from_tensor_slices((xData_test, yData_test))
    dataset_test = dataset_test.shuffle(buffer_size=num_data_test)
    dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)

    time_record = 0

    #--------------------------------------------------------------------------------
    # 学習とテストの反復
    start = time.time()
    for ite in range(nIte):

        start_train = time.time()
        
        # ミニバッチ法で学習
        # 各バッチデータから計算した損失の平均
        train_loss_mean, train_acc_mean = train_step(classifier_model, dataset_train, train_loss, train_acc)

        end_train = time.time() - start_train

        print('#{0}, time : {1} train loss : {2}, train-acc : {3}'.format(ite, end_train, train_loss_mean, train_acc_mean))

        # test_rate回に1回、テスト時の損失を計算
        if ite % test_rate==0:
            test_loss_mean, test_acc_mean = test_step(classifier_model, dataset_test, test_loss, test_acc)
            print('#{0}, test loss : {1}, test-acc : {2}'.format(ite, test_loss_mean, test_acc_mean))
        
    #--------------------------------------------------------------------------------
    end = time.time() - start
    print(end)