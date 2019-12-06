import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import pdb

#--------------------------------------------------------------------------------
# データの読み込み
def load_data():
    # scikit-leanからbostonデータセットを読み込み
    boston = load_boston()

    # bostonデータセットの説明変数をpandasのデータフレームに変換
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

    # bostonデータ・セットの目的変数をデータフレームに追加
    boston_df['target'] = boston.target

    return boston_df
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 正規化
def norm(xData):
    mean = np.mean(xData, axis=0)
    std = np.std(xData, axis=0)
    return (xData - mean) / std
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# データの前処理
def pre_processing(boston_df):

    # 説明変数と目的変数に分割(ndarrayで保存)
    yData = boston_df.target.values
    xData = boston_df[boston_df.columns[boston_df.columns!='target']].values

    xData = norm(xData)

    return xData, yData
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# 学習データとテストデータに分割
def split_data(xData, yData):

    xData_train, xData_test, yData_train, yData_test = train_test_split(xData, yData, test_size=0.2, random_state=42)

    # 目的変数は(404,)と(102,)のベクトルになっているので、(404,1)と(102,1)の行列に変更する
    yData_train = yData_train[:,np.newaxis]
    yData_test = yData_test[:,np.newaxis]

    return xData_train, xData_test, yData_train, yData_test
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
def linear_regression(x_t, xDims, reuse=False):
    with tf.variable_scope('linear_regression') as scope:
        if reuse:
            scope.reuse_variables()
        
        # 重みを初期化
        w = weight_variable('w', [xDims, 1])
        # バイアスを初期化
        b = bias_variable('b', [1])

        # 線形回帰を実行
        y = tf.add(tf.matmul(x_t, w), b)

        return y

#--------------------------------------------------------------------------------


if __name__=="__main__":

    #--------------------------------------------------------------------------------
    # データの前処理

    # データのロード
    boston_df = load_data()

    # データの前処理
    xData, yData = pre_processing(boston_df)

    # 学習データとテストデータに分割
    xData_train, xData_test, yData_train, yData_test =  split_data(xData, yData)
    #--------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    # Tensorflowで用いる変数を定義
    
    # 説明変数のカテゴリー数(次元)を取得
    xDim = xData.shape[1]

    # 特徴量(x_t)とターゲット(y_t)のプレースホルダー
    x_t = tf.placeholder(tf.float32,[None,xDim])
    y_t = tf.placeholder(tf.float32,[None,1])
    learning_rate = tf.constant(0.01, dtype=tf.float32)
    #--------------------------------------------------------------------------------

    # 線形回帰を実行
    output_train = linear_regression(x_t, xDim)
    output_test = linear_regression(x_t, xDim, reuse=True)

    # 損失関数(最小二乗誤差)
    loss_square_train = tf.reduce_mean(tf.square(y_t - output_train))
    loss_square_test = tf.reduce_mean(tf.square(y_t - output_test))

    # 最適化
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = opt.minimize(loss_square_train)

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
    nIte = 1000
    
    # placeholderに入力するデータを設定
    train_dict = {x_t: xData_train, y_t: yData_train}
    test_dict = {x_t: xData_test, y_t: yData_test}
    
    # 学習とテストの反復
    for ite in range(nIte):
        
        # 勾配降下法と最小二乗誤差を計算
        _, loss_train = sess.run([training_step, loss_square_train], feed_dict=train_dict)
        
        # lossの履歴を保存
        loss_train_list.append(loss_train)
        
        # 反復10回につき一回lossを表示
        if ite % 10 == 0:
            print('#{0}, train loss : {1}'.format(ite, loss_train))

        # 反復100回につき1回テストを実行
        if ite % 100 == 0:
            loss_test = sess.run(loss_square_test, feed_dict=test_dict)
            loss_test_list.append(loss_test)
            print('#{0}, test loss : {1}'.format(ite, loss_test))
    #--------------------------------------------------------------------------------