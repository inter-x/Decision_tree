# Decision_tree

#### ID3

ID3 使用的分类标准是信息增益，它表示得知特征 A 的信息而使得样本集合不确定性减少的程度。

数据集的**信息熵**：

$H(D) = -\sum_{k=1}^K\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|}$

其中$C_k$表示集合$D$中属于第$k$类样本的样本子集.

针对某个特征$A$对于数据集$D$的**条件熵**：

$H(D|A)$为：$H(D|A) = \sum_{i=1}^n \frac{|D_i|}{|D|}H(D_i) = -\sum_{i=1}^n\frac{|D_i|}{|D|}(\sum_{k = 1}^K\frac{|D_ik|}{|D_i|}log_2\frac{|D_ik|}{|D_i|})$

其中$D_i$表示$D$中特征$A$取第$i$个值的样本子集$D_ik$表示$D_i$中属于第$k$类的样本子集

**信息增益 = 信息熵 - 条件熵**

$Gain(D,A) = H(D) - H(D|A)$
C4.5;
Cart
