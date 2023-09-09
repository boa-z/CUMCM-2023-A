# 请阅读代码，分析 point 数量不随函数输入的改变而改变

# 导入matplotlib库，用于绘制图形
import matplotlib.pyplot as plt
import numpy as np

# 定义一个递增函数f(x) = 1.2x
def f(x):
    return 1.2 * x

#卡卡大厦基石
def fondational_kabuilding(r,R,d,k):
    # 圆的半径,内圆间隔，点的间隔，圆的个数
    # R目前为无用参数，后续可用于绘制卡卡大厦的外圈，目前用f(r)代替，k控制内圆的个数

    # 创建一个新的图形
    plt.figure()
    # 设置坐标轴的范围和比例
    plt.axis([-r, r, -r, r])
    plt.axis('equal')
    # 绘制一个以原点为中心，半径为r的圆
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y)

    i=100
    # 在x轴100之后半径递增生成一个个圆圈
    for m in range(0, k): 
        i=f(i)
        x = i * np.cos(theta)
        y = i * np.sin(theta)
        plt.plot(x, y)

    # 定义一个空列表，用于存储所有点的坐标
    points = []
    # 定义一个变量，用于统计点的个数
    count = 0

    # 在各圆圈上填充定日镜的点
    for m in range(0, k):
        # 计算当前圆圈的半径
        i = f(100 * (1.2 ** m))
        # 计算当前圆圈上可以放置多少个点
        n = int(np.floor(2 * np.pi * i / d))
        # 计算当前圆圈上每个点之间的角度差
        delta = 2 * np.pi / n
        # 遍历当前圆圈上的每个点
        for k in range(n):
            # 计算当前点的角度
            angle = k * delta
            # 计算当前点的坐标
            x = i * np.cos(angle)
            y = i * np.sin(angle)
            # 将当前点的坐标添加到列表中
            points.append((x, y))
            # 将当前点绘制在图形上
            plt.scatter(x, y, color='red')
            # 点的个数加一
            count += 1
    print("The coordinates of all the points are:")
    for point in points:
        print(point)
    print("The total number of points is:", count)

    plt.show()
    
    return points


fondational_kabuilding(700,100,12,20)
#print(fondational_kabuilding(500,20,12,8))