import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python import pywrap_tensorflow
#加法器
# log = 'd:/venv/log/add_temple'
# a = tf.constant(1.0, name='ta')
# b = tf.constant(2.0, name='tb')
# c = tf.add(a,b, name='tc')
# ss = tf.Session()
# xss = ss.run(c)
# xsum = tf.summary.FileWriter(log, ss.graph)
# ss.close()

#乘法器
# log = 'd:/venv/log/log_temp'
# a = tf.Variable(1.0, name='ta')
# b = tf.constant(1.0, name='tb')
# c = tf.multiply(a, b, name='tc')
# ss = tf.Session()
# xsum = tf.summary.FileWriter(log, ss.graph)
# init = tf.global_variables_initializer()
# ss.run(init)
# xss = ss.run(c)
# print('xss',xss)
# ss.close()

# hello tensorflow
# hello = tf.constant('Hello7')
# ss = tf.Session()
# print(ss.run(hello))
#
# a = tf.constant(2)
# b = tf.constant(3)
# with tf.Session() as sess:
#     print("a=2 , b = 3")
#     print('a+b=', ss.run(a+b))
#     print('a*b=', ss.run(a*b))

#缺省图运算 contant常量
# gg = tf.get_default_graph()
# op1 = gg.get_operations()
# print('op,',op1)
# print('type(op1),',type(op1))
# init_dat = tf.constant(1.0)
# # op2 = gg.get_operations()
# # print('init_dat,',init_dat)
# # print('type(init_dat)',type(init_dat))
# # print('op2:',op2)
# # print('type(op):',type(op2))
# # print('op.node_def,',op2[0].node_def)
# # ss = tf.Session()
# # xdat = ss.run(init_dat)
# # print('xdat',xdat)
# # ss.close()

# a = tf.constant(1.0, name='input')
# b = tf.Variable(0.8, name='weight')
# c = tf.multiply(a, b, name='output')
# rlog = 'd:/venv/log/log_tem'
# ss  = tf.Session()
# xsum = tf.summary.FileWriter(rlog, ss.graph)
# ss.close()

# 占位符运算
# rlog = "d:/venv/log/log_tmp"
# x = tf.placeholder(tf.float32, name='x')
# y = tf.placeholder(tf.float32, name='y')
# z = tf.add(x, y, name='z')
# ss = tf.Session()
# xsum = tf.summary.FileWriter(rlog, ss.graph)
# xss = ss.run(z, feed_dict={x:1, y:2.0})


# rlog = 'd:/venv/log/varible_tmp'
# a = tf.Variable(1.0, name='a')
# b = tf.Variable(1.0, name='b')
# c = tf.add(a, b, name='c')
# ss = tf.Session()
# init = tf.global_variables_initializer()
# ss.run(init)
# tf.summary.FileWriter(rlog, ss.graph)
# xsum = ss.run(c)
# print('xsum',xsum)
# ss.close()

# 变量操作
# rlog = 'd:/venv/log/variable_test_tmp'
# cnt = tf.Variable(0, name='cnt') #cnt = 0
# a = tf.constant(1, name='a') #a = 1
# y = tf.add(cnt, a) # y = 1
# y2 = tf.assign(cnt, y) # cnt = 1, y2 = 1
# init = tf.global_variables_initializer()
# with tf.Session() as ss:
#     ss.run(init)
#     xss = ss.run(cnt)
#     print('x.cnt,',xss)
#     for xc in range(3):
#         ys2 = ss.run(y2)
#         xs2 = ss.run(cnt)
#         print('x2.cn2,',xs2)
#
#     xsum = tf.summary.FileWriter(rlog, ss.graph)

# feed数据提交
# rlog = 'd:/venv/log/feed_data_tmp'
# a = tf.placeholder(tf.float32, name='ta') #占位中没有数据用feed 添加数据
# b = tf.placeholder(tf.float32, name='tb')
# c = tf.multiply(a, b, name='tc')
# init = tf.global_variables_initializer()
# with tf.Session() as ss:
#     ss.run(init)
    # xss = ss.run([c], feed_dict={a:[7.0], b:[2.0]})
    # print('xss',xss)
    # xsum = tf.summary.FileWriter(rlog, ss.graph)

    # feed批量数据提交
    # a_dat = [[1., 2., 3.], [4., 5., 6.]]
    # b_dat = [[2., 3., 4.], [3., 2., 1.]]
    # xss = ss.run([c], feed_dict={a:a_dat, b:b_dat})
    # print('xss',xss)

# fetch 获取数据
# rlog = 'd:/venv/log/fetch_data_tmp'
# a = tf.constant(3.0, name='ta')
# b = tf.constant(2.0, name='tb')
# c = tf.constant(5.0, name='tc')
# m1 = tf.add(b,c, name='m1')
# m2 = tf.multiply(a, m1, name='m2')
# with tf.Session() as ss:
#     xss = ss.run([m2,m1])
#     print('xss', xss)
#     xsum = tf.summary.FileWriter(rlog, ss.graph)
#获取fetch多维数据
# a_dat = [[1.,2.,3.],[4.,5.,6.]]
# b_dat = [[2.,3.,4.],[3.,2.,1.]]
# c_dat = [[3.,2.,1.],[1.,2.,3.]]
# a = tf.constant(a_dat, name='ta')
# b = tf.constant(b_dat, name='tb')
# c = tf.constant(c_dat, name='tc')
# m1 = tf.add(b, c, name='m1')
# m2 = tf.multiply(a, m1, name='m2')
# with tf.Session() as ss:
#     xss = ss.run([m2, m1])
#     print('xss', xss)
#     xsum = tf.summary.FileWriter(rlog, ss.graph)

#单细胞算法 公式: y_model = w*x
# #1
# rlog = 'd:/venv/log/danxibao_tmp'
# x = tf.constant(2.0, name='input')
# w = tf.Variable(0.8, name='weight')
# y_model = tf.multiply(w, x, name='output')
# #2
# print('\n#2,set.dat #2')
# y_ = tf.constant(0.0, name='correct_value')
# loss = tf.pow(y_model - y_, 2, name='loss')
# #3
# print('\n#3, train_step')
# """ 使用梯度下降优化器（Optimizer）作为学习函数，按照误差（loss）导数来调整w权重参数数值。
#     通过学习率（learning rate）调节（moderate）更新权重大小 设置为0.025
# """
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
# #4
# """
#     添加summary日志数据进行合并优化
# """
# print('\n#4,summary.misc')
# for value in [x, w, y_model, y_, loss]:
#     tf.summary.scalar(value.op.name, value)
# summaries = tf.summary.merge_all()
# #5
# print('\n#5,Session')
# ss = tf.Session()
# #6
# print('\n#6,summary, rlog',rlog)
# xsum = tf.summary.FileWriter(rlog, ss.graph)
# #7
# print('\n#7,Session init')
# init = tf.global_variables_initializer()
# ss.run(init)
# #8
# print('\n#8,Session init')
# for i in range(100):
#     #8.a
#     xdat =ss.run(summaries)
#     xsum.add_summary(xdat, i )
#     x2 = ss.run(train_step)
#     #--------
#     #8.b
#     x2, y2, w2 = ss.run(x), ss.run(y_model), ss.run(w)
#     s2 = ss.run(loss)
#     print(i, '#,y2,loss:',y2, s2,',x2,w2 ',x2,w2)
# print('\n#9,Seesion.close')
# # ss.close()

# #2-1创建图，启动图（graph）
# # 创建常量op
# m1 = tf.constant([[3,3]])
# # 创建常量op
# m2 = tf.constant([[2],[3]])
# # 创建一个矩阵乘法op， 把m1,m2 传入
# product = tf.matmul(m1, m2)
# print(product)
# # 定义一个会话，启动默认图
# sess = tf.Session()
# #调用sess的run方法来执行矩阵乘法op
# # run（product）出发图中3个op
# result = sess.run(product)
# tf.summary.FileWriter('d:/venv/log/matmui_tmp', sess.graph)
# sess.close()

# # 2-2变量介绍
# x = tf.Variable([1,2])
# a = tf.constant([3,3])
# #增加一个减法op
# sub = tf.subtract(x,a)
# #增加一个加法op
# add = tf.add(x, sub)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     result_sub = sess.run(sub)
#     result_add = sess.run(add)
#     print(result_sub)
#     print(result_add)
#     tf.summary.FileWriter("d:/venv/log/subtracr_tmp", sess.graph)
# # 创建一个变量初始化为0
# state = tf.Variable(0, name='counter')
# # 创建一个op, 作用是使state加1
# new_value = tf.add(state,1)
# # 赋值op
# update = tf.assign(state, new_value)
# # 变量初始化
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))

# 2-3 feed
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1, input2)
# with tf.Session() as sess:
#     # feed数据以字典形式传入
#     print(sess.run(output,feed_dict={input1:[7.], input2:[2.]}))

#使用numpy生成100个随机数 (样本)
# x_data = np.random.rand(100)
# y_data = x_data*0.1 + 0.2
# #  构造一个线性模型
# b = tf.Variable(0.)
# k = tf.Variable(0.)
# y = k*x_data + b
#
# # 二次代价函数
# loss = tf.reduce_mean(tf.square(y_data - y))
# # 定义一个梯度下降法来进行训练的优化器
# optimizer = tf.train.GradientDescentOptimizer(0.2)
# # 最小化代价函数
# train = optimizer.minimize(loss)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(201):
#         sess.run(train)
#         if step % 2 == 0:
#             print(step,sess.run([k,b]))
#     tf.summary.FileWriter("d:/venv/log/test1_tmp", sess.graph)

# 线性回归
# x = tf.constant([1,2,3,4,5,6],tf.float32)
# y = tf.constant([3,4,7,8,11,14],tf.float32)
# k = tf.Variable(1.0,dtype=tf.float32)
# b = tf.Variable(1.0,dtype=tf.float32)
# # 求loss loss为差值平方
# loss = tf.reduce_sum(tf.square(y - (k*x+b)))
# # 初始化变量
# init = tf.global_variables_initializer()
# # 创建会话
# with tf.Session() as sess:
#     sess.run(init)
#     opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
#     msg = []
#     for i in range(500):
#         sess.run(opti)
#         msg.append(sess.run(loss))
#         if i % 50 ==0:
#             print((sess.run(k), sess.run(b)))
#     plt.figure(1)
#     plt.plot(msg)
#     plt.show()
#     #     画出六个点及最后计算出的直线
#     plt.figure(2)
#     x_arry, y_arry = sess.run([x, y])
#     plt.plot(x_arry, y_arry, 'o')
#     xx = np.arange(0, 10, 0.05)
#     yy = sess.run(k) * xx + sess.run(b)
#     plt.plot(xx, yy)
#     plt.show()


# 多元线性回归
xy = tf.placeholder(tf.float32, [None, 2], name='xy')
z = tf.placeholder(tf.float32, [None, 1], name='z')
# 初始化z = w1*x + w2*y + b
w = tf.Variable(tf.constant([[1], [1]], tf.float32), dtype=tf.float32, name='w')
b = tf.Variable(1.0, dtype=tf.float32, name='b')

# 损失函数
loss = tf.reduce_sum(tf.square(z - (tf.matmul(xy, w)+b)))
# 初始化变量
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
# 梯度下降
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
# 记录方差
msg = []
# 训练数据
xy_train = np.array([
    [0.1, 0.8], [0.2, 0.3], [0.6, 0.8], [0.8, 0.4]
], np.float32)

z_train = np.array([[1080], [2030], [7080], [8040]], np.float32)

# 训练模型 循环500次
for i in range(10000):
    #梯度下降
    session.run(opti,feed_dict={xy: xy_train, z: z_train})
    #计算每一次迭代损失值 ,追加到列表中保存
    msg.append(session.run(loss, feed_dict={xy: xy_train, z: z_train}))
    #每隔100次打印a b 值
    if i % 100 == 0:
        print("-----------第", i, "次的迭代值----------")
        print(session.run([w, b]))
#建立需要导出的预测模型
model = tf.add(tf.matmul(xy, session.run(w)), session.run(b), name='model')
#导出pb预测模型
constant_graph = tf.compat.v1.graph_util.extract_sub_graph(session.graph_def, ['model'])
with tf.gfile.GFile("./model/model.pb", "w+") as f:
    f.write(constant_graph.SerializeToString())
# 画出顺势函数的值
plt.figure(1)
plt.plot(msg)
plt.show()
# 测试数据
print(session.run(model, feed_dict={xy: [[0.8, 0.8]]}))

