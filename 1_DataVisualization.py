# --- 目的：数据质量探索与可视化 ---

# 所有引用的包放在最前边
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm # 检验数据是否符合正态分布
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# 1.先看看数据格式--------------------------------------------------------------------------------------------
train = pd.read_csv('used_car_train_20200313.csv',sep=' ') # 第一次没有设置分隔符，发现只有一列数据（原表是按照空格分隔的）
test = pd.read_csv('used_car_testB_20200421.csv',sep=' ')
print(train.shape)  # (150000, 31)
print(train.head())
print(test.shape)
print(test.head()) # (50000, 30)


# 2.探索数据质量：缺失值和异常值的检测及处理---------------------------------------------------------------------
# 2.1 缺失值
# 2.1.1 合并训练集和测试集，减少数据处理工作量
n_train = train.shape[0]      # 记录下训练集和测试集的行数，以便数据处理完将训练集和测试集再次划分开
n_test = test.shape[0]
y_train = train.price.values  # 训练集比测试集多“Price”列
all_data = pd.concat([train,test])
all_data.reset_index(inplace=True) # 拼接后索引要重置
print("all_data shape:{}".format(all_data.shape))
# 2.1.2 计算缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征
missingData = pd.DataFrame({"MissingData":all_data_na})
print(missingData)
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
# 2.1.5 除了为空的数值以外，发现"notRepairedDamage"列存在特殊符号代替的值“-”，视为缺失值进行处理
mo = all_data["notRepairedDamage"].mode()[0] # "notRepairedDamage"列的众数填补
def change(x):
    if x=="-":
        return mo
    else:
        return x
all_data["notRepairedDamage"]=all_data["notRepairedDamage"].apply(change)


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
    data_new = data_new.drop(index,axis=0)
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
    sns.boxplot(y=data[col_name], data=data, palette='Set3',ax=ax[0])  # palette是色系设置；ax=ax[0]表示将原数据画在左边（1行2列的画图窗口哟）
    sns.boxplot(y=data_new[col_name], data=data_new, palette='Set3',ax=ax[1]) # 将删除掉异常值的图画在右边
    plt.grid()
    plt.show()
    return data_new # 返回删除掉异常值的新数据
# 2.2.2 根据相关系数矩阵选择与目标变量相关性强的特征进行异常值检测和处理
corr = all_data.corr()  # 计算相关系数
plt.figure(figsize=(12,9))
sns.heatmap(corr)
plt.show()
# v_0、v_8、v_12
for col in ('v_0','v_8','v_12'):
    all_data = outliers(all_data, col)
print(all_data.shape)



# 3.特征工程----------------------------------------------------------------------------
# 3.1 避免多重共线性，剔除部分特征
# offerType（报价类型）v_6、v_7、v_9、v_13
for f in ('offerType','v_6','v_7','v_9','v_13'):
    all_data = all_data.drop([f],axis=1)
print(all_data.shape)
print(all_data.head())

# 3.2 特征选择
# 这里使用皮尔逊相关系数，剔除掉相关系数>0.9的特征（避免多重共线性）
# 注意：除过目标变量price列
threshold = 0.9
price = all_data['price']
trainData = all_data.drop(['price'],axis=1)
corr_matrix = all_data.corr().abs() # 相关系数矩阵
print(corr_matrix.head())
# 选择上三角阵处理
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool)) # np.triu()获取上三角阵，k表示对角线起始位置
print(upper.head())
# 删除相关系数>0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("There are %s columns to remove."%(len(to_drop)))
all_data = all_data.drop(columns = to_drop)
all_data['price'] = price
print("all_data.shape:",all_data.shape)

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
# 概率图检测
fig = plt.figure()
pricePP = stats.probplot(trainPrice['price'],plot=plt) # 概率以理论分布的比例（x轴）显示，y轴包含样本数据的未缩放分位数
plt.grid()
plt.show()
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
# 概率图检测
fig1 = plt.figure()
pricePP_After = stats.probplot(trainPrice['price'],plot=plt)
plt.grid()
plt.show()


# 4. 建模---------------------------------------------------------------------------
# 4.1 先将训练集和测试集划分开
train_y = trainPrice['price']  # 这里要存储下训练集的目标变量，拟合模型的时候用
trainDataAfterPre = trainPrice.drop('price',axis=1)  # 上边已经将训练集命名为了trainPrice
print(trainDataAfterPre.shape)
testDataAfterPre = all_data[~all_data['price'].notnull()]  # ‘price’列为空的数据则是测试集
testDataAfterPre = testDataAfterPre.drop(['price'], axis=1)
print(testDataAfterPre.shape)

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

# 4.3 拟合模型
clf = Ridge(alpha=0.01)
clf.fit(trainDataAfterPre,train_y)
# 4.4 预测
predict = clf.predict(testDataAfterPre)
testDataAfterPre['price'] = predict
print(testDataAfterPre.head())
# 将预测结果输出到新的表中
sub = pd.DataFrame()
sub['SaleID'] = testDataAfterPre['SaleID']
sub['price'] = predict
sub.to_csv('used_car_submit.csv')