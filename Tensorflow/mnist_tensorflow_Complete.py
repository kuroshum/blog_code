import tensorflow as tf
import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
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
# 重みの初期化
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# バイアスの初期化
def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 線形回帰モデル
def linear_regression(x_t, xDim, yDim, reuse=False):
    with tf.variable_scope('linear_regression') as scope:
        if reuse:
            scope.reuse_variables()
        
        # 重みを初期化
        w = weight_variable('w', [xDim, yDim])
        # バイアスを初期化
        b = bias_variable('b', [yDim])

        # softmax回帰を実行
        y = tf.nn.softmax(tf.add(tf.matmul(x_t, w), b))

        return y

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 線形回帰モデル
def classifier_model(x_t, xDim, yDim, reuse=False):
    with tf.variable_scope('classifier_model') as scope:
        if reuse:
            scope.reuse_variables()
        
        # 重みを初期化
        w1 = weight_variable('w1', [xDim, 128])
        # バイアスを初期化
        b1 = bias_variable('b1', [128])

        # softmax回帰を実行
        h1 = tf.nn.relu(tf.add(tf.matmul(x_t, w1), b1))

        # 重みを初期化
        w2 = weight_variable('w2', [128, yDim])
        # バイアスを初期化
        b2 = bias_variable('b2', [yDim])

        # softmax回帰を実行
        y = tf.nn.softmax(tf.add(tf.matmul(h1, w2), b2))

        return y

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

    #--------------------------------------------------------------------------------
    # Tensorflowで用いる変数を定義
    
    # 説明変数のカテゴリー数(次元)を取得
    xDim = xData.shape[1]

    #pdb.set_trace()

    # 特徴量(x_t)とターゲット(y_t)のプレースホルダー
    x_t = tf.placeholder(tf.float32,[None,xDim])
    y_t = tf.placeholder(tf.float32,[None,yDim])
    learning_rate = tf.constant(0.01, dtype=tf.float32)
    #--------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    # Tensorflowで用いるグラフを定義

    # 線形回帰を実行
    output_train = classifier_model(x_t, xDim, yDim)
    output_test = classifier_model(x_t, xDim, yDim, reuse=True)

    # 損失関数(クロスエントロピー)
    loss_square_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_t, logits=output_train))
    loss_square_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_t, logits=output_test))

    # 最適化
    opt = tf.train.AdamOptimizer(learning_rate)
    training_step = opt.minimize(loss_square_train)
    #--------------------------------------------------------------------------------

    # セッション作成
    sess = tf.Session()
    
    # 変数の初期化
    init = tf.global_variables_initializer()
    sess.run(init)

    #--------------------------------------------------------------------------------
    # 学習とテストを実行

    # lossの履歴を保存するリスト
    loss_train_list = []
    loss_test_list = []

    # accuracyの履歴を保存するリスト
    accuracy_train_list = []
    accuracy_test_list = []

    # イテレーションの反復回数
    nIte = 500

    # テスト実行の割合(test_rate回につき1回)
    test_rate = 10

    # バッチサイズ
    BATCH_SIZE = 500

    # 学習データ・テストデータの数
    num_data_train = xData_train.shape[0]
    num_data_test = xData_test.shape[0]
        
    # 学習とテストの反復
    for ite in range(nIte):
        pern = np.random.permutation(num_data_train)
        for i in range(0, num_data_train, BATCH_SIZE):

            batch_x = xData_train[pern[i:i+BATCH_SIZE]]
            batch_y = yData_train[pern[i:i+BATCH_SIZE]]

            # placeholderに入力するデータを設定
            train_dict = {x_t: batch_x, y_t: batch_y}

            # 勾配降下法と最小二乗誤差を計算
            sess.run([training_step], feed_dict=train_dict)
        
        loss_train = sess.run(loss_square_train, feed_dict=train_dict)

        output = sess.run(output_train, feed_dict=train_dict)
        accuracy_train = accuracy_score(np.argmax(batch_y,axis=1), np.argmax(output,axis=1))

        
        # 反復10回につき一回lossを表示
        if ite % test_rate == 0:
            test_dict = {x_t: xData_test, y_t: yData_test}
            
            loss_test = sess.run(loss_square_test, feed_dict=test_dict)
            accuracy_test = accuracy_score(np.argmax(yData_test, axis=1), np.argmax(sess.run(output_test, feed_dict=test_dict), axis=1))

            # lossの履歴を保存
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)

            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)


            print('#{0}, train loss : {1}'.format(ite, loss_train))
            print('#{0}, test loss : {1}'.format(ite, loss_test))
    #--------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    # 学習とテストのlossの履歴をplot

    fig = plt.figure()

    # Plot lines
    plt.xlabel('epoch')
    plt.plot(range(len(loss_train_list)), loss_train_list, label='train_loss')
    plt.plot(range(len(loss_test_list)), loss_test_list, label='test_loss')
    plt.legend()
    plt.show()

    fig = plt.figure()

    # Plot lines
    plt.xlabel('epoch')
    plt.plot(range(len(accuracy_train_list)), accuracy_train_list, label='train_accuracy')
    plt.plot(range(len(accuracy_test_list)), accuracy_test_list, label='test_accuracy')
    plt.legend()
    plt.show()

    #--------------------------------------------------------------------------------