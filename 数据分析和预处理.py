from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def describe(data,num):
    '''数值描述图'''
    sns.set(font='SimHei', font_scale=1.2)
    desc = data.describe().T  # 对于count mean std min 25% 50% 75% max 进行展示
    desc_df = pd.DataFrame(index=data.columns, columns=data.describe().index, data=desc)
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(desc_df, annot=True, cmap="Purples", fmt='.3f',
                ax=ax, linewidths=5, cbar=False,
                annot_kws={"size": 16})  # 绘制热力图
    plt.xticks(size=18)
    plt.yticks(size=14, rotation=0)
    # plt.savefig('实验二图片/2_{}.pdf'.format(num))
    plt.show()
def count(data,attr,num):
    '''类别数量图'''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 22})
    sns.countplot(data[attr], label="数量")
    plt.ylabel('数量')
    # plt.savefig('实验二图片/2_{}.pdf'.format(num))
    plt.show()
def correlation_map(data,num):
    '''相关热力图'''
    f, ax = plt.subplots(figsize=(15, 15))
    sns.set(font_scale=2)
    sns.heatmap(data.corr(), annot=True, linewidths=2, fmt='.1f', ax=ax,annot_kws={"size": 20})
    # plt.savefig('实验二图片/2_{}.pdf'.format(num))
    plt.show()

if __name__=='__main__':
    # \\\\\\\\\\ iris数据集
    data = pd.read_csv('data/Iris.csv').drop(['Id'],axis=1)
    count(data,'Species',1)
    describe(data.drop(['Species'],axis=1),2)
    correlation_map(data.drop(['Species'],axis=1),3)
    # data.to_csv('data/D2.csv')

    # \\\\\\\\\\ heart数据集
    data1 = pd.read_csv('data/heart.csv')
    print(np.unique(data1['target'], return_counts=True))
    count(data1,'target',4)
    describe(data1.drop(['target'],axis=1),5)
    correlation_map(data1.drop(['target'],axis=1),6)
    data1.drop_duplicates(keep='first', inplace=True)  # 将重复数据删除
    # data1.to_csv('data/D1.csv')
