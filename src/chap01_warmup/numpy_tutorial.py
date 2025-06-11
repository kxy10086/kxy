#!/usr/bin/env python3
# coding: utf-8
# numpy 练习题，包含 array 基本操作及可视化演示

# 1. 导入 numpy 库，常用别名为 np
import numpy as np

# 导入 matplotlib 库用于绘图
import matplotlib
import matplotlib.pyplot as plt
# 指定 matplotlib 的后端为 TkAgg，确保图形窗口正常显示（尤其是在本地环境）
matplotlib.use('TkAgg')

# 2. 创建一维数组并查看其属性
print("第二题：\n")
a = np.array([4, 5, 6])               # 创建一维 numpy 数组
print("(1)输出a的类型（type）\n", type(a))       # 显示类型，应为 <class 'numpy.ndarray'>
print("(2)输出a的各维度的大小（shape）\n", a.shape) # 显示形状，(3,)
print("(3)输出a的第一个元素（element）\n", a[0])   # 输出第一个元素，4

# 3. 创建二维数组并索引元素
print("第三题：\n")
b = np.array([[4, 5, 6], [1, 2, 3]])
print("(1)输出各维度的大小（shape）\n", b.shape)   # (2, 3)
print("(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])

# 4. 创建特殊矩阵：全零、全一、单位矩阵、随机矩阵
print("第四题：\n")
a = np.zeros((3, 3), dtype=int)      # 3x3 全零矩阵，元素类型为整型
b = np.ones((4, 5))                  # 4x5 全一矩阵，默认类型为 float
c = np.eye(4)                        # 4x4 单位矩阵
d = np.random.random((3, 2))         # 3x2 随机数矩阵，每个值在[0,1)

# 5. 多维数组索引操作
print("第五题：\n")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)                             # 打印整个数组
print(a[2, 3], a[0, 0])              # 输出第3行第4列和第1行第1列的值

# 6. 数组切片，取子矩阵
print("第六题：\n")
b = a[0:2, 1:3]                      # 取第1、2行和第2、3列，得到2x2子矩阵
print("(1)输出b\n", b)
print("(2) 输出b 的（0,0）这个元素的值\n", b[0, 0])

# 7. 数组切片，取最后两行
print("第七题：\n")
c = a[-2:, :]                        # 取最后两行的所有列
print("(1)输出 c \n", c)
print("(2)输出 c 中第一行的最后一个元素\n", c[0, -1]) # 第一行最后一个元素

# 8. 花式索引，按指定下标取元素
print("第八题：\n")
a = np.array([[1, 2], [3, 4], [5, 6]])
print("输出:\n", a[[0, 1, 2], [0, 1, 0]]) # 输出 1, 4, 5

# 9. 结合 arange 和花式索引，跨行取不同列
print("第九题：\n")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])           # 每一行要取的列下标
print("输出:\n", a[np.arange(4), b]) # 取(0,0),(1,2),(2,0),(3,1)位置元素

# 10. 用花式索引批量修改元素
print("第十题：\n")
a[np.arange(4), b] += 10             # 将上一步取出的四个元素加10
print("输出:", a)

# 11. 查看一维数组的数据类型
print("第十一题：\n")
x = np.array([1, 2])
print("输出:", type(x))              # <class 'numpy.ndarray'>

# 12. 查看一维浮点型数组的数据类型
print("第十二题：\n")
x = np.array([1.0, 2.0])
print("输出:", type(x))              # <class 'numpy.ndarray'>

# 13. 两个二维数组的加法操作
print("第十三题：\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print("x+y\n", x + y)                # 逐元素相加
print("np.add(x,y)\n", np.add(x, y)) # 与 x+y 等价

# 14. 两个二维数组的减法操作
print("第十四题：\n")
print("x-y\n", x-y)                  # 逐元素相减
print("np.subtract(x,y)\n", np.subtract(x, y))

# 15. 逐元素乘法与矩阵乘法的区别
print("第十五题：\n")
print("x*y\n", x * y)                # 逐元素相乘
print("np.multiply(x, y)\n", np.multiply(x, y))
print("np.dot(x,y)\n", np.dot(x, y)) # 标准矩阵乘法

# 16. 逐元素除法
print("第十六题：\n")
print("x/y\n", x / y)                # 逐元素相除
print("np.divide(x,y)\n", np.divide(x, y))

# 17. 计算开方
print("第十七题：\n")
print("np.sqrt(x)\n", np.sqrt(x))    # x 的逐元素开方

# 18. 矩阵乘法两种写法
print("第十八题：\n")
print("x.dot(y)\n", x.dot(y))        # 对象方法
print("np.dot(x,y)\n", np.dot(x, y)) # numpy 函数

# 19. 数组求和操作
print("第十九题：\n")
print("print(np.sum(x)):", np.sum(x))             # 所有元素求和
print("print(np.sum(x, axis=0))", np.sum(x, axis=0)) # 沿列方向求和
print("print(np.sum(x, axis=1))", np.sum(x, axis=1)) # 沿行方向求和

# 20.*

