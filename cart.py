import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
class CART_DecisionTree(object):
    """
    基于cart算法的决策树
    """
    def __init__(self, _max_depth, _min_splits):
        self.max_depth = _max_depth # 最大的深度
        self.min_splits = _min_splits # 最小的叶子节点数

    def fit(self, _feature, _label):
        self.feature = _feature
        self.label = _label
        self.train_data = np.column_stack((self.feature,self.label))
        self.build_tree() # 构造二叉树


    def compute_gini_similarity(self, groups, class_labels):
        """
        计算基尼系数
        """
        num_sample = sum([len(group) for group in groups]) # 计算总数目
        gini_score = 0
        # 对在该划分下求基尼系数
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for label in class_labels:
                porportion = (group[:,-1] == label).sum() / size
                score += porportion * porportion
            gini_score += (1.0 - score) * (size/num_sample)
        return gini_score

    def terminal_node(self, _group):
        """
        获取叶子节点中的最多数量的类的类别作为分类类别
        """
        class_labels, count = np.unique(_group[:,-1], return_counts= True)
        return class_labels[np.argmax(count)]

    def split(self, index, val, data):
        """
        根据值划分成两组，分组过程,其中蕴含排序
        """
        data_left = np.array([]).reshape(0,self.train_data.shape[1])
        data_right = np.array([]).reshape(0, self.train_data.shape[1])
        # 进行划分 小于等于的val的在data_left，大于val的在data_right
        for row in data:
            if row[index] <= val :
                data_left = np.vstack((data_left,row))


            if row[index] > val:
                data_right = np.vstack((data_right, row))
        return data_left, data_right

    def best_split(self, data):
        """
        找到最优切分点
        """
        class_labels = np.unique(data[:,-1]) # 获取类别
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        for idx in range(data.shape[1]-1):
            for row in data:
                groups = self.split(idx, row[idx], data)
                gini_score = self.compute_gini_similarity(groups,class_labels) # 得到当前划分下的基尼系数

                if gini_score < best_score:
                    # 小于当前的最小基尼系数的时候更新
                    best_index = idx
                    best_val = row[idx]
                    best_score = gini_score
                    best_groups = groups
        result = {}
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        return result


    def split_branch(self, node, depth):
        """
        以递归的方式分枝，直到满足对最大深度或者小于最小的叶节点数目
        """
        left_node , right_node = node['groups']
        del(node['groups']) # 删除这一key


        if depth >= self.max_depth:
            # 超过深度
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return None

        if len(left_node) <= self.min_splits:
            # 小于最小节点数量的时候
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            # 递归
            self.split_branch(node['left'],depth + 1)


        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            # 递归
            self.split_branch(node['right'],depth + 1)

    def build_tree(self):
        """
        递归的方式创建树
        """
        self.root = self.best_split(self.train_data)
        self.split_branch(self.root, 1)
        return self.root

    def _predict(self, node, row):
        """
        通过递归的方式查找树来预测
        """
        if row[node['index']] < node['val']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']

        else:
            if isinstance(node['right'],dict):
                return self._predict(node['right'],row)
            else:
                return node['right']

    def predict(self, test_data):
        """
        返回的是概率
        """
        self.predicted_label = np.array([])
        for idx in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root,idx))

        return self.predicted_label

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
    DT = CART_DecisionTree(4,30)
    DT.fit(X_train,y_train)
    prediction = DT.predict(X_test)
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
    ax.set_title('决策树混淆矩阵')  # 标题
    ax.set_xlabel('预测值')  # x轴
    ax.set_ylabel('真实值')  # y轴
    # plt.savefig('图片/2_9.pdf')
    plt.show()
    # ///////////////////////////////////////////////// heart
    #----------------- 决策树
    data1 = pd.read_csv('data/D1.csv').drop(['Unnamed: 0'],axis = 1)
    X1 = np.array(data1.drop(columns=['target']))
    y1 = np.array(data1['target'])
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=1)
    DT1 = CART_DecisionTree(3,30)
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
    # plt.savefig('图片/2_10.pdf')
    plt.show()

