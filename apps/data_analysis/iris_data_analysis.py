
from sklearn import datasets
from pandas import DataFrame

data = datasets.load_iris()
print(data)

data_x = data.data
target = data.target
label_name = data.target_names
print('类别名', label_name)

# 1. 表格打印数据
col_name = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
pd_data = DataFrame(data_x, columns=col_name)
pd_data['类别'] = target  # 新增一列
print(pd_data)

# 2. 查看数据统计信息
print("-------------")
print(pd_data.describe())
