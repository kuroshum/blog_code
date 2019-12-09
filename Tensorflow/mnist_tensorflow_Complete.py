import tensorflow as tf
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

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
    output_train = linear_regression(x_t, xDim, yDim)
    output_test = linear_regression(x_t, xDim, yDim, reuse=True)

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

    # lossの履歴を保存
    loss_train_list = []
    loss_test_list = []

    # イテレーションの反復回数
    nIte = 100

    # テスト実行の割合(test_rate回につき1回)
    test_rate = 10

    # バッチサイズ
    BATCH_SIZE = 64

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
        # lossの履歴を保存
        loss_train_list.append(loss_train)

        output = sess.run(output_train, feed_dict=train_dict)
        
        print('#{0}, train loss : {1}'.format(ite, loss_train))
        
        # 反復10回につき一回lossを表示
        if ite % test_rate == 0:
            pern = np.random.permutation(num_data_test)
            for i in range(0, num_data_test, BATCH_SIZE):
                batch_x = xData_test[pern[i:i+BATCH_SIZE]]
                batch_y = yData_test[pern[i:i+BATCH_SIZE]]
        
                test_dict = {x_t: batch_x, y_t: batch_y}
            
            loss_test = sess.run(loss_square_test, feed_dict=test_dict)

            loss_test_list.append(loss_test)

            print('#{0}, test loss : {1}'.format(ite, loss_test))
    #--------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    # 学習とテストのlossの履歴をplot

    #--------------------------------------------------------------------------------