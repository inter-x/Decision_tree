import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
class ID3_DecisionTree(object):
    '''
    基于ID3算法的决策树
    '''
    def __init__(self):
        self.tree = {}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        n = len(y)
        counts = {}
        for value in y:
            counts[value] = counts.get(value, 0) + 1
        entropy = 0
        for count in counts.values():
            p = count / n
            entropy -= p * math.log2(p)
        return entropy

    def _information_gain(self, X, y, feature):
        n = len(y)
        values = set([x[feature] for x in X])
        entropy = 0
        for value in values:
            subset_x = [x for x in X if x[feature] == value]
            subset_y = [y[i] for i in range(len(y)) if X[i][feature] == value]
            entropy += len(subset_y) / n * self._entropy(subset_y)
        information_gain = self._entropy(y) - entropy
        return information_gain

    def _majority_vote(self, y):
        counts = {}
        for value in y:
            counts[value] = counts.get(value, 0) + 1
        majority = max(counts, key=counts.get)
        return majority

    def _build_tree(self, X, y):
        if len(set(y)) == 1:
            return y[0]
        if len(X[0]) == 0:
            return self._majority_vote(y)
        best_feature = max(range(len(X[0])), key=lambda i: self._information_gain(X, y, i))
        tree = {best_feature: {}}
        values = set([x[best_feature] for x in X])
        for value in values:
            subset_x = [x for x in X if x[best_feature] == value]
            subset_y = [y[i] for i in range(len(y)) if X[i][best_feature] == value]
            subtree = self._build_tree(subset_x, subset_y)
            tree[best_feature][value] = subtree
        return tree
    
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                value = X[i][feature]
                node = node[feature][value]
            y_pred.append(node)
        return self.y_pred
    
    def accuracy(self,x_test,y_test):
        y_predict = self.predict(x_test)
        return np.sum(y_test==y_predict)/y_test.shape[0]

    def macro_precision(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = self.predict(x_test)
        P = []
        for i in y_unique:
            P.append(np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_predict==i))
        return np.sum(P)/len(P)

    def macro_recall(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = self.predict(x_test)
        R = []
        for i in y_unique:
            R.append(np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_test==i))
        return np.sum(R) / len(R)

    def macro_f1_score(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = self.predict(x_test)
        F = []
        for i in y_unique:
            p = np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_predict==i)
            r = np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_test==i)
            F.append((2*p*r)/(p+r))
        return np.sum(F) / len(F)


    def precision(self,x_test,y_test):
        y_predict = self.predict(x_test)
        y_unique = np.unique(y_test)
        return np.sum(y_predict[y_predict==y_test]==y_unique[0])/np.sum(y_predict==y_unique[0]),np.sum(y_predict[y_predict==y_test]==y_unique[1])/np.sum(y_predict==y_unique[1])
    def recall(self,x_test,y_test):
        y_predict = self.predict(x_test)
        y_unique = np.unique(y_test)
        return np.sum(y_predict[y_predict==y_test]==y_unique[0])/np.sum(y_test==y_unique[0]),np.sum(y_predict[y_predict==y_test]==y_unique[1])/np.sum(y_test==y_unique[1])
    def f1_score(self,x_test,y_test):
        precision = self.precision(x_test,y_test)
        recall = self.recall(x_test,y_test)
        return 2*precision[0]*recall[0]/(precision[0]+recall[0]),2*precision[1]*recall[1]/(precision[1]+recall[1])
    def report(self,x_test,y_test,labels):
        y_predict = self.predict(x_test)
        y_predict = y_predict.astype(y_test.dtype)
        report = classification_report(y_test,y_predict,labels=labels)
        return report
    
if __name__ == "__main__":
    # ///////////////////////////////////////////////// Iris
    #----------------- 决策树
    data = pd.read_csv('data/D2.csv').drop(['Unnamed: 0'],axis = 1)
    X = np.array(data.drop(columns=['Species']))
    y = np.array(data['Species'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    DT = ID3_DecisionTree()
    DT.fit(X_train,y_train)
    prediction = DT.predict(X_test)
    print('预测结果',prediction)
    print('训练集下：------------------------------------')
    print(DT.report(X_train, y_train,labels=['Iris-setosa','Iris-versicolor','Iris-virginica']))
    print('测试集下：------------------------------------')
    print('accuracy:',DT.accuracy(X_test,y_test))
    print('macro_precision:',DT.macro_precision(X_test,y_test))
    print('macro_recall:',DT.macro_recall(X_test,y_test))
    print('macro_f1_score:',DT.macro_f1_score(X_test,y_test))
    print('micro版本==accuracy:',DT.accuracy(X_test,y_test))
    print(DT.report(X_test,y_test,labels=['Iris-setosa','Iris-versicolor','Iris-virginica']))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()
    C2 = confusion_matrix(y_test, DT.predict(X_test), labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
    sns.heatmap(C2, annot=True, ax=ax,xticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'],yticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'])  # 画热力图
    ax.set_title('ID3决策树混淆矩阵')  # 标题
    ax.set_xlabel('预测值')  # x轴
    ax.set_ylabel('真实值')  # y轴
    plt.savefig('ID3图片/ID3_1.pdf')
    plt.show()
    # ///////////////////////////////////////////////// heart
    #----------------- 决策树
    data1 = pd.read_csv('data/D1.csv').drop(['Unnamed: 0'],axis = 1)
    X1 = np.array(data1.drop(columns=['target']))
    y1 = np.array(data1['target'])
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=1)
    DT1 = ID3_DecisionTree(3,30)
    DT1.fit(X_train1,y_train1)
    print(np.unique(y1))
    print('训练集下：------------------------------------')
    print(DT1.report(X_train1, y_train1, labels=[0, 1]))
    print('测试集下：------------------------------------')
    print('accuracy:',DT1.accuracy(X_test1,y_test1))
    print('precision:',DT1.precision(X_test1,y_test1))
    print('recall:',DT1.recall(X_test1,y_test1))
    print('f1_score:',DT1.f1_score(X_test1,y_test1))

    print('micro版本==accuracy:',DT1.accuracy(X_test1,y_test1))
    print('macro_precision:',DT1.macro_precision(X_test1,y_test1))
    print('macro_recall:',DT1.macro_recall(X_test1,y_test1))
    print('macro_f1_score:',DT1.macro_f1_score(X_test1,y_test1))
    print('micro版本==accuracy:',DT1.accuracy(X_test1,y_test1))
    print(DT1.report(X_test1,y_test1,labels=[0,1]))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()
    C3 = confusion_matrix(y_test1, DT1.predict(X_test1), labels=[0,1])
    sns.heatmap(C3, annot=True, ax=ax)  # 画热力图
    ax.set_title('决策树混淆矩阵')  # 标题
    ax.set_xlabel('预测值')  # x轴
    ax.set_ylabel('真实值')  # y轴
    plt.savefig('ID3图片/ID3_2.pdf')
    plt.show()