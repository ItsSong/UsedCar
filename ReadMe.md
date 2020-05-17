# 二手车交易价格预测项目
本项目数据来自天池大赛，目的是预测二手车交易价格。数据集分训练集和测试集，训练集共15万条数据，测试集5万条数据。
数据包含31列变量信息，其中15列为匿名变量，“Price”是预测目标。
***
## 一、数据质量探索
### 1.查看数据格式<br>
```python
# 所有引用的包放在最前边
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# 1.先看看数据格式
train = pd.read_csv('used_car_train_20200313.csv',sep=' ') # 第一次没有设置分隔符，发现只有一列数据（原表是按照空格分隔的）
test = pd.read_csv('used_car_testB_20200421.csv',sep=' ')
print(train.shape)
print(train.head())
print(test.shape)
print(test.head())
```
![二手车原始数据格式](https://img-blog.csdnimg.cn/20200516164959175.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

### 2.探索数据质量：缺失值、异常值
#### （1）合并训练集和测试集，同时查找缺失值进行处理，减少工作量
```python
# 2.探索数据质量
# 2.1 缺失值
# 2.1.1 合并训练集和测试集，减少数据处理工作量
n_train = train.shape[0]      # 记录下训练集和测试集的行数，以便数据处理完将训练集和测试集再次划分开
n_test = test.shape[0]
y_train = train.price.values  # 训练集比测试集多“Price”列
all_data = pd.concat([train,test])
all_data.reset_index(inplace=True)
print("all_data shape:{}".format(all_data.shape))
```
合并之后数据为：all_data shape:(200000, 32)，原因重置了索引，多了index列

#### （2）缺失值检测
```python
# 2.1.2 计算缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征
missingData = pd.DataFrame({"MissingData":all_data_na})
print(missingData)
```
![缺失值](https://img-blog.csdnimg.cn/20200517190715573.png)

有缺失值的特征：price、fuelType（燃油类型）、gearbox（变速箱）、bodyType（车身类型）、model（车型编码）。
数据质量还可以hhh！<br>
其中，price缺失率是25%，刚好测试集占总数据集合的25%，所以训练集的price值没有缺失！接下来对其他缺失值进行处理<br>
#### （3）缺失值处理
```python
# 2.1.3 缺失值处理
# fuelType（燃油类型）gearbox（变速箱）bodyType（车身类型）model（车型编码）:考虑使用众数填补
# print(all_data['fuelType'].mode())    # mode函数求众数，显示第一个值为众数，第二个值为众数的数据类型
for i in ('fuelType','gearbox','bodyType','model'):
    all_data[i] = all_data[i].fillna(all_data[i].mode()[0])
# 2.1.4 检测缺失值处理结果
all_data_naAfter = (all_data.isnull().sum()/len(all_data))*100
all_data_naAfter = all_data_naAfter.drop(all_data_naAfter[all_data_naAfter==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征
missingDataAfter = pd.DataFrame({"MissingDataAfterPreprocess":all_data_naAfter})
print(missingDataAfter)
```
缺失值处理结果：<br>
![处理缺失值](https://img-blog.csdnimg.cn/20200517191222994.png)

OK，没有缺失值了！不过还存在特殊符号替代的值，这里视为缺失值进行处理。
```python
# 2.1.5 除了为空的数值以外，发现"notRepairedDamage"列存在特殊符号代替的值“-”，视为缺失值进行处理
mo = all_data["notRepairedDamage"].mode()[0] # "notRepairedDamage"列的众数填补
def change(x):
    if x=="-":
        return mo
    else:
        return x
all_data["notRepairedDamage"]=all_data["notRepairedDamage"].apply(change)
```

#### （4）异常值检测及处理<br>
使用箱线图对异常值进行检测和处理，这里封装一个异常值检测及处理的函数：
```python
# 2.2 异常值检测及处理
# 2.2.1 包装一个箱线图函数，剔除掉Ql-1.5(Qu-Ql)和Qu+1.5(Qu-Ql)以外的数据
def outliers(data, col_name):
    # data : 接收pandas数据
    # col_name : pandas列名
    def box_plot_outlier(data_ser):
        # data_ser：接收pandas.Series格式
        quantileSpace = 1.5 * (data_ser.quantile(0.75) - data_ser.quantile(0.25))  # 1.5倍的分位数间距
        outliersLOW = data_ser.quantile(0.25) - quantileSpace   # 正常值下边界
        outliersUp = data_ser.quantile(0.75) + quantileSpace
        rule_low = (data_ser < outliersLOW)  # 小于下边界的异常值
        rule_up = (data_ser > outliersUp)
        return (rule_low, rule_up),(outliersLOW,outliersUp)
    data_new = data.copy()
    data_series = data_new[col_name] # 某一列
    rule, value = box_plot_outlier(data_series)
    # 取异常值的索引：
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]] # 选择的某列数据的行数shape[0]，rule[0]是小于正常值的异常值，rule[1]是大于正常值的异常值
    print("Delete number is:{}".format(len(index)))
    data_new = data_new.drop(index)
    data_new.reset_index(drop=True,inplace=True) # 删除了异常值后，其索引还在原表中，需要重置索引，将异常值索引删除掉
    print("Now column number is:{}".format(data_new.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]  # 正常值下边界索引
    outliersDataL = data_series.iloc[index_low]
    print("Description of data less than the lower bond is:")
    print(pd.Series(outliersDataL).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliersDataU = data_series.iloc[index_up]
    print("Description of data Larger than the upper bond is:")
    print(pd.Series(outliersDataU).describe())
    # 可视化
    fig,ax = plt.subplots(1,2,figsize=(10,8)) #fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)
    # fig, ax = plt.subplots()等价于：
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(y=data[col_name], data=data, palette='Set3', ax=ax[0])  # palette是色系设置；ax=ax[0]表示将原数据画在左边（1行2列的画图窗口哟）
    sns.boxplot(y=data_new[col_name], data=data_new, palette='Set3', ax=ax[1]) # 将删除掉异常值的图画在右边
    return data_new # 返回删除掉异常值的新数据
```
包装好了异常值检测及处理函数，这里选择哪些特征进行处理呢，我们来看看相关系数矩阵吧!
```python
# 2.2.2 根据相关系数矩阵选择与目标变量相关性强的特征进行异常值检测和处理
corr = all_data.corr()  # 计算相关系数
plt.figure(figsize=(12,9))
sns.heatmap(corr)
plt.show()
```
相关系数矩阵可视化结果：<br>
![相关系数矩阵](https://img-blog.csdnimg.cn/20200517111953113.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

从相关系数矩阵(热力图)可以看出：<br>
（1）regDate（汽车注册日期）、offerType（报价类型）、v_0、v_8、v_12这四个特征与目标变量‘price’的相关性最强。<br>
（2）每一个特征与offerType（报价类型）的相关性接近1，避免多重共线性，考虑剔除掉offerType（报价类型）<br>
（3）v_6和v_1相关性接近1；v_7和v_2相关性接近1；v_9和v_4相关性接近1；v_4和v_13相关性接近1。避免多重共线性，考虑剔除掉v_6、v_7、v_9、v_13<br>

先对目标变量影响大的特征进行处理：regDate（汽车注册日期）、offerType（报价类型）、v_0、v_8、v_12。其中，汽车注册日期我们暂且不考虑，报价类型需要删除掉，因此只对v_0、v_8、v_12处理。
```python
# v_0、v_8、v_12
for col in ('v_0','v_8','v_12'):
    all_data = outliers(all_data, col)
print(all_data.shape)
```
对v_0处理结果：共删除掉6112行数据。处理异常值前后的箱线图如下：<br>
![v_0处理结果](https://img-blog.csdnimg.cn/20200517200227463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

![v_0图](https://img-blog.csdnimg.cn/20200517200351487.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

对v_8处理结果：共删除掉43行数据。处理异常值前后的箱线图如下：<br>
![v_8处理结果](https://img-blog.csdnimg.cn/20200517200523978.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

![v_8图](https://img-blog.csdnimg.cn/20200517200557584.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

对v_12处理结果：共删除掉214行数据。处理异常值前后的箱线图如下：<br>
![v_12图](https://img-blog.csdnimg.cn/20200517200654266.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

![v_12图](https://img-blog.csdnimg.cn/20200517200722820.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

删除掉'v_0','v_8','v_12'三个特征的异常值之后，查看数据格式：
```python
print(all_data.shape)
```
结果：(193631, 32)  
OK，数据处理的差不多了！接下来就是最重要的一步：特征工程！

### 3.特征工程
#### （1）避免多重共线性，剔除部分特征<br>
趁热打铁，刚才计算了变量间的相关性，首先我们剔除掉部分特征以避免多重共线性。
```python
# 3. 特诊工程
# 3.1 避免多重共线性，剔除部分特征
# offerType（报价类型）v_6、v_7、v_9、v_13
for f in ('offerType','v_6','v_7','v_9','v_13'):
    trainData = trainData.drop([f],axis=1)
print(trainData.shape)
print(trainData.head())
```
结果：(193631, 27)<br>

![处理结果](https://img-blog.csdnimg.cn/20200517201404574.png)


#### （2）特征选择<br>
```python
# 3.2 特征选择
# 这里使用皮尔逊相关系数，剔除掉相关系数>0.9的特征（避免多重共线性）
# 注意：除过目标变量price列
threshold = 0.9
price = all_data['price']
trainData = all_data.drop(['price'],axis=1)
corr_matrix = all_data.corr().abs() # 相关系数矩阵
print(corr_matrix.head())
```
![相关系数](https://img-blog.csdnimg.cn/20200517202755840.png)

```python
# 选择上三角阵处理
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool)) # np.triu()获取上三角阵，k表示对角线起始位置
print(upper.head())
```
![上三角阵](https://img-blog.csdnimg.cn/20200517202914970.png)

```python
# 删除相关系数>0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("There are %s columns to remove."%(len(to_drop)))
all_data = all_data.drop(columns = to_drop)
all_data['price'] = price
print("all_data.shape:",all_data.shape)
```
结果：<br>
There are 3 columns to remove.<br>
all_data.shape: (193631, 24)<br>
特征工程中最重要的就是特征选择和特征抽取，特征抽取是增加新的特征，暂且对二手车交易数据集不做特征抽取。<br>
另外，特征工程中还要对数据进行编码，这个二手车数据集已经是编码好的数据，所以这一步省略。

#### （3）检验数据是否符合模型的假设<br>
由于这里想使用最简单的回归分析，那么数据就要符合回归分析的假设：<br>
线性&可加性：X每发生一个单位的变动，Y会发生固定单位的变动，与X的绝对数值无关。X对Y的影响是独立的，即各个自变量间独立<br>
各自变量间不相关：避免多重共线性。当多重共线性性出现的时候，变量之间的联动关系会导致测得的标准差偏大，置信区间变宽。采用岭回归，Lasso回归可以一定程度上减少方差，解决多重共线性性问题。因为这些方法，在最小二乘法的基础上，加入了一个与回归系数的模有关的惩罚项，可以收缩模型的系数。<br>
残差：服从正态分布；残差项之间不相关，否则为自相关；方差恒定，否则为异方差<br>
开始检验：<br>
```python
# 3.3 检验数据是否符合模型的假设，若不符合，转化为符合模型假设的格式
# 3.3.1 检验数据是否符合正态分布
trainPrice = all_data[all_data['price'].notnull()] # 由于数据已经处理过了，删除了某些行，这里需要重新选出训练集
sns.distplot(trainPrice['price'],fit=norm) # 拟合正态分布曲线
(mu,sigma) = norm.fit(trainPrice['price']) # 计算均值，方差
print('\n mu={:.2f} and sigma={:.2f} \n'.format(mu,sigma))
plt.legend(['Normal distribution ($\mu=${:.2f} and $\sigma=${:.2f}'.format(mu,sigma)],loc='best')
plt.ylabel('Price Frequency')
plt.title('Used Car Price Distribution After Transform')
plt.grid()
plt.show()
```
结果：mu=5949.81 and sigma=7227.02 <br>
![右偏分布](https://img-blog.csdnimg.cn/20200517210949132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

可以看出数据属于右偏分布。可以再看看PP概率图来检测一番:<br>
```python
# 概率图检测
fig = plt.figure()
pricePP = stats.probplot(trainPrice['price'],plot=plt) # 概率以理论分布的比例（x轴）显示，y轴包含样本数据的未缩放分位数
plt.grid()
plt.show()
```
![pp概率图](https://img-blog.csdnimg.cn/20200517211131383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

显然，确实是右偏分布。接下来需要将数据转化为正态分布，方法有多种，比如：对数变换、倒数变换（适合两端波动较大的数据）、平方根变换（适用于泊松分布或轻度偏态数据）、平方根反正弦（适用二项分布数据）、Box-Cox变换（适用于有负值的数据）等，这里我们使用对数变换。<br>
```python
# 3.3.2 将非正态数据转化为正态分布的数据
trainPrice['price'] = np.log1p(trainPrice['price']) # 取对数：log1p = log（x+1）
sns.distplot(trainPrice['price'],fit=norm)
(mu,sigma) = norm.fit(trainPrice['price'])
print('\n mu={:.2f} and sigma={:.2f} \n'.format(mu,sigma))
plt.legend(['Normal distribution ($\mu=${:.2f} and $\sigma=${:.2f}'.format(mu,sigma)],loc='best')
plt.ylabel('Price Frequency')
plt.title('Used Car Price Distribution After Transform')
plt.grid()
plt.show()
```
转换结果：  mu=8.07 and sigma=1.19  <br>
![转化后结果](https://img-blog.csdnimg.cn/20200517211528995.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

再画个概率图：<br>
```python
# 概率图检测
fig1 = plt.figure()
pricePP_After = stats.probplot(trainPrice['price'],plot=plt)
plt.grid()
plt.show()
```
![处理后pp概率图](https://img-blog.csdnimg.cn/20200517211627872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

转化成功！可以了，接下来进行建模~<br>
### 4.建模
先将训练集和测试集划分开：
```python
# 4. 建模
# 4.1 先将训练集和测试集划分开
train_y = trainPrice['price']  # 这里要存储下训练集的目标变量，拟合模型的时候用
trainDataAfterPre = trainPrice.drop('price',axis=1)  # 上边已经将训练集命名为了trainPrice
print(trainDataAfterPre.shape)
testDataAfterPre = all_data[~all_data['price'].notnull()]  # ‘price’列为空的数据则是测试集
testDataAfterPre = testDataAfterPre.drop(['price'], axis=1)
print(testDataAfterPre.shape)
```
结果显示训练集形状：(145220, 23)；测试集形状：(48411, 23)<br>
接下来选择模型最佳参数：交叉验证法
```python
# 4.2 交叉验证,选择最优参数
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, trainDataAfterPre, train_y, scoring='neg_mean_squared_error', cv=10)) # 按照均方误差评分；10折交叉验证
    return  rmse
alphas = [0.01,0.02,0.03,0.04,0.05, 0.1, 0.2, 0.5, 1,3, 5] # 参数alphas，控制正则项的强弱（防止过拟合和欠拟合）
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
            for alpha in alphas]  # 每调节一次alphas，计算一次模型误差的均值
# 可视化模型误差随alpha参数变化的曲线
cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title="Validation Score")
plt.xlabel('alpha')
plt.ylabel('rmse')
plt.grid()
plt.show()
# 测试完发现alpha取0.01
```
参数选择好，开始拟合模型，并进行预测！
```python
# 4.3 拟合模型
clf = Ridge(alpha=0.01)
clf.fit(trainDataAfterPre,train_y)
```
结果：<br>
![岭回归模型拟合结果](https://img-blog.csdnimg.cn/20200517215322660.png)
```python
# 4.4 预测
predict = clf.predict(testDataAfterPre)
testDataAfterPre['price'] = predict
print(testDataAfterPre.head())
```
看一看预测的前五行：<br>
![预测结果](https://img-blog.csdnimg.cn/20200517215643932.png)

至此，整个数据分析及挖掘的过程就完成了。总结一下：<br>
（1）数据处理过程当中，在删除某行或某列值时记得保存，可能后边又会用到<br>
（2）特征工程很重要：特征编码、特征选择、特征抽取、检验数据是否符合模型假设<br>
（3）建模中：交叉验证法选择最优参数~<br>
加油！