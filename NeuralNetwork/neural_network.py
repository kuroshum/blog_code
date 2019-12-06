import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os

class neuralnetwork(object):
    
    def __init__(self, x, t, hDim=20):
        #pdb.set_trace()
        # 入力データの次元
        xDim = x.shape[1]
        # 隠れ層のノード数
        hDim  = hDim
        # カテゴリ数
        tDim = t.shape[1]

        dNum = x.shape[0]
    
        self.W1 = np.random.normal(0.0, pow(hDim, -0.5), (xDim + 1, hDim))
        self.W2 = np.random.normal(0.0, pow(tDim, -0.5), (hDim + 1, tDim))

        self.x = x
        self.t = t

        self.losses = np.array([])

        self.accuracies = np.array([])

        # データインデックスの初期化
        self.randInd = np.random.permutation(dNum)
        self.validInd = np.random.permutation(dNum)[:(int)(dNum*0.1)]

    def update(self, x, t, alpha=0.1, printEval=True):
        #pdb.set_trace()
        # データ数
        dNum = x.shape[1]

        # 中間層の計算
        h = self.hidden(x)

        # 事後確率の予測と真値の差分
        predict = self.predict(x,h)
        predict_error =  predict - t

        #pdb.set_trace()

        # 入力層に値「1」のノードを追加
        x_with_one = np.append(x, np.ones([1,x.shape[1]]),axis=0)
        
        #pdb.set_trace()
        # W1の更新
        hidden_error = np.matmul(self.W2,predict_error)
        self.W1 -= alpha * np.matmul(x_with_one,(hidden_error[:-1] * h * (1-h)).T)/dNum

        # 中間層に値「1」のノードを追加
        h_with_one = np.append(h, np.ones([1,h.shape[1]]),axis=0)

        # W2の更新
        self.W2 -= alpha * np.matmul(predict_error, h_with_one.T).T/dNum
        #pdb.set_trace()

        if printEval:
            self.losses = np.append(self.losses, self.loss(self.x.T[:,self.validInd],self.t.T[:,self.validInd]))
            self.accuracies = np.append(self.accuracies, self.accuracy(self.x.T[:,self.validInd],self.t.T[:,self.validInd]))
            print("loss:{0:02.3f}, accuracy:{1:02.3f}".format(self.losses[-1],self.accuracies[-1]))
        
    #------------------------------------
	# 6) 交差エントロピー損失と正解率のグラフを保存
    def plotEval(self,Losses,Accuracy):
        path = os.getcwd()

        plt.figure(1)
        plt.plot(range(Losses.shape[0]),Losses, '-o', label="cross-entropy loss")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("loss")
        filename0 = "losses"
        plt.savefig(path + '/visualization/' + filename0)

        plt.figure(2)
        plt.plot(range(Accuracy.shape[0]),Accuracy, '-o', label="Accuracy")
        plt.legend()
        plt.xlabel("Iterarion")
        plt.ylabel("Accuracy")
        filename1 = "accuracies"
        plt.savefig(path + '/visualization/' + filename1)
        plt.show()
    #------------------------------------
            
    #------------------------------------
    # ネクストバッチ
    def nextBatch(self, x, t, batchCnt, batchSize):
        #pdb.set_trace()
        sInd = batchSize * batchCnt
        eInd = sInd + batchSize

        if eInd+batchSize > x.shape[0]:
            batchCnt = 0
        else:
            batchCnt += 1
        
        xBatch = x[self.randInd[sInd:eInd]]
        tBatch = t[self.randInd[sInd:eInd]]

        return xBatch, tBatch, batchCnt
    #------------------------------------

    #------------------------------------
    # 正解率の計算
    # x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
    def accuracy(self, x, t):
        dNum = x.shape[1]

        # 最大の事後確率をとるカテゴリ
        maxInd = np.argmax(self.predict(x),axis=0)

        # TR(True Positive)の数
        tpNum = np.sum([t[maxInd[i],i] for i in np.arange(dNum)])

        # 正解率=TP/データ数
        return tpNum/dNum
    #------------------------------------

    #------------------------------------
    # 4) 交差エントロピーの計算
    # x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
    def loss(self, x,t):
        #pdb.set_trace()
        dNum = x.shape[1]
        crossEntropy = -np.sum(t*np.log(self.predict(x)))/dNum
        #pdb.set_trace()
        return crossEntropy
        #------------------------------------
    #------------------------------------
    # 7) 事後確率の計算
    # x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # h: 中間層のデータh（中間層のノード数×データ数のnumpy.array）
    def predict(self, x, h = []):
        if not len(h):
            h = self.hidden(x)
        return self.softmax(np.matmul(self.W2[:-1].T, h) + self.W2[-1][np.newaxis].T)
    #------------------------------------

    #------------------------------------
    # 9) 中間層
    # x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    def hidden(self, x):
        h = self.sigmoid(np.matmul(self.W1[:-1].T, x) + self.W1[-1][np.newaxis].T)
        return h
    #------------------------------------

    #------------------------------------
    # 8) シグモイドの計算
    # x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    def sigmoid(self,x):
        sigmoid = 1.0/(1.0+np.exp(-x))
        return sigmoid
    #------------------------------------

    #------------------------------------
    # 2) ソフトマックスの計算
    # x: カテゴリ数×データ数のnumpy.array
    def softmax(self,x):
        # x-max(x):expのinfを回避するため
        e = np.exp(x-np.max(x))
        return e/np.sum(e,axis=0)
    #------------------------------------
#------------------------------------------------
# Holdout法
# 入力されたデータを訓練データ・テストデータに分割
# x    : 説明変数
# t    : 目的変数
# rate : データ分割の割合
def Holdout(x, t, rate):
    # 全データ数
    data_num = len(x)
    # 訓練データ数(全データ数のrate割)
    train_num = int(data_num*rate)

    np.random.seed(1)

    # データ数
    randInd = np.random.permutation(data_num)

    # 訓練データ( 0 ~ train_num )
    xTrain = x[randInd[:train_num]]
    tTrain = t[randInd[:train_num]]

    # テストデータ( train_num ~ data_num)
    xTest = x[randInd[train_num:]]
    tTest = t[randInd[train_num:]]

    return xTrain, tTrain, xTest, tTest
#------------------------------------------------

#------------------------------------------------
# Clossvalidation法
# x    : 説明変数
# t    : 目的変数
# sInd : テストデータの始まりのインデックス
# eInd : テストデータの終わりのインデックス
def Clossvalidation(x, t, sInd, eInd,randInd):
    
    #pdb.set_trace()

    # 訓練データ( 0 ~ sInd )
    # 訓練データ( eInd ~ data_num )
    xTrain = x[randInd[0:sInd]]
    tTrain = t[randInd[0:sInd]]
    xTrain = np.append(xTrain, x[randInd[eInd:data_num]],axis=0)
    tTrain = np.append(tTrain, t[randInd[eInd:data_num]],axis=0)

    # テストデータ( sInd ~ eInd)
    xTest = x[randInd[sInd:eInd]]
    tTest = t[randInd[sInd:eInd]]

    return xTrain, tTrain, xTest, tTest
#------------------------------------------------

#------------------------------------------------
# クラス数を10から、2 or 5に変更する
def SetClassNum(tData, standard, classNum=2):
    if classNum == 5:
        index = 0
        for i in tData:
            if 10 > i and i > 7:
                tData[index] = 4
            elif 8 > i and i > 5:
                tData[index] = 3
            elif 6 > i and i > 3:
                tData[index] = 2
            elif 4 > i and i > 1:
                tData[index] = 1
            else:
                tData[index] = 0
            index += 1

    elif classNum == 2:
        tData = np.array([1 if i > standard else 0 for i in tData])[np.newaxis].T
    
    # 目的変数をone-hot表現に
    tData = np.eye(classNum)[tData[:,0]]
    
    return tData
#------------------------------------------------

def Standardization(xData, Flag=True):
    xData_cent_list = []
    if Flag == True:
        for i in range(xData.shape[1]):
            mean_vals = np.mean(xData[:,i], axis=0)
            std_val = np.std(xData[:,i])
            xData_cent = ((xData[:,0] - mean_vals) / std_val)
            xData_cent_list.append(xData_cent)
        xData_cent_list = np.array(xData_cent_list).reshape([xData.shape[0],xData.shape[1]])
    else:
        xData_cent = xData

    return xData_cent_list

if __name__ == "__main__":
    # ワインデータ(csv)の読み込み
    #wine_data_set = pd.read_csv('winequality-red.csv',sep=";",header=0)
    wine_data_set = pd.read_csv('winequality-white.csv')
    #pdb.set_trace()
    
    # 説明変数(ワインに含まれる成分(quality以外のカテゴリ))
    xData = np.array(pd.DataFrame(wine_data_set.drop("quality",axis=1)))

    # 標準化
    xData_cent = Standardization(xData, Flag=True)

    # 目的変数(ワインの品質(10段階))
    tData = np.array(pd.DataFrame(wine_data_set["quality"]))

    # 分類するクラス数を 10 から 2 に変更(0:美味しくない or 1:美味しい)
    # Standard : 基準値
    standard = 6
    tData = SetClassNum(tData, standard, classNum=2)
    
    # データの分割の割合
    rate = 0.8

    # 全データ数
    data_num = len(xData)

    # データ数
    randInd = np.random.permutation(data_num)
    # 訓練データとテストデータに分割
    xTrain_cent, tTrain, xTest_cent, tTest = Holdout(xData_cent, tData, rate)
    
    # neuralnetworkクラスのインスタンス化
    classifier = neuralnetwork(xTrain_cent, tTrain, hDim=200)

    # 学習回数
    Nite = 2000
    # 更新幅
    learningRate = 0.01
    # 更新幅の減衰率
    decayRate = 0.99999

    Losses = []
    Accuracy = []

    # バッチサイズ
    batch_size = 79
    batchCnt = 0
    for ite in np.arange(Nite):
        # ネクストバッチ法
        xBatch, tBatch, batchcnt = classifier.nextBatch(xTrain_cent, tTrain, batchCnt, batch_size)
        batchCnt = batchcnt
        #pdb.set_trace()

        print("Training ite:{} ".format(ite+1),end='')
        
        # 重み W1 W2 の更新
        classifier.update(xBatch.T, tBatch.T, alpha=learningRate)

        # 5）更新幅の減衰
        learningRate *= decayRate

        Losses = np.append(Losses,classifier.losses[-1])
        Accuracy = np.append(Accuracy,classifier.accuracies[-1])

    classifier.plotEval(Losses,Accuracy)
    
    # 6) 評価
    loss = classifier.loss(xTest_cent.T,tTest.T)
    accuracy = classifier.accuracy(xTest_cent.T,tTest.T)
    print("Test loss:{}, accuracy:{}".format(loss,accuracy))