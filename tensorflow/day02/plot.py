#导入必要的模块
import numpy as np
import matplotlib.pyplot as plt

#产生测试数据
x = np.arange(1, 10, 0.1)
print(x)

y = np.sin(x)


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 2)

#设置标题
ax1.set_title('Scatter Plot')

#设置X轴标签
plt.xlabel('X')
#设置Y轴标签
plt.ylabel('Y')

#画散点图
ax1.scatter(x, y , c = 'r',marker = 'o')

#设置图标
plt.legend('x1')


ax2 = fig.add_subplot(2, 2, 3)
ax2.scatter(x, y , c = 'b',marker = 'o')




#显示所画的图
plt.show()
