import pandas as pd # 数据处理
import numpy as np # 使用数组
import matplotlib.pyplot as plt # 可视化
from matplotlib import rcParams # 图大小

from sklearn.tree import DecisionTreeClassifier as dtc # 树算法
from sklearn.model_selection import train_test_split # 拆分数据
from sklearn.metrics import accuracy_score # 模型准确度
from sklearn.tree import plot_tree # 树图

rcParams['figure.figsize'] = (25, 20)

df = pd.read_csv('data.csv',dtype='str')
del df['User id']

print(df)

for i in df.Age.values:
    if i  == '<=30':
        df.Age.replace(i, 0, inplace = True)
    elif i == '[31,40]':
        df.Age.replace(i, 1, inplace = True)
    elif i == '>40':
        df.Age.replace(i, 2, inplace = True)

for i in df.Incoming.values:
    if i == 'low':
        df.Incoming.replace(i, 0, inplace = True)
    elif i == 'medium':
        df.Incoming.replace(i, 1, inplace = True)
    elif i == 'high':
        df.Incoming.replace(i, 2, inplace = True)

for i in df.Student.values:
    if i == 'no':
        df.Student.replace(i, 0, inplace = True)
    elif i == 'yes':
        df.Student.replace(i, 1, inplace = True)

for i in df.Credit_Rating.values:
    if i == 'fair':
        df.Credit_Rating.replace(i, 0, inplace = True)
    elif i == 'excellent':
        df.Credit_Rating.replace(i, 1, inplace = True)

#print(df)

X_var = df[['Age', 'Incoming', 'Student', 'Credit_Rating']].values # 自变量
y_var = df['Buying'].values # 因变量

#print('X variable samples : {}'.format(X_var[:5])
#print('Y variable samples : {}'.format(y_var[:5]))

X_train = X_var[0:13,:]
y_train = y_var[0:13]
X_test = X_var[13,0:4]
print(X_test)
model = dtc(criterion = 'entropy', max_depth = 4)
model.fit(X_train, y_train)

pred_model = model.predict(X_test.reshape(1,-1))
print(pred_model)

feature_names = df.columns[:5]
target_names = df['Buying'].unique().tolist()

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)

plt.savefig('tree_visualization.png')