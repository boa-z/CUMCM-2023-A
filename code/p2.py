# 导入matplotlib库，用于绘制图形
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import p1

tower_para = (0, 0, 84)
eta_sb = 0.9
eta_trunc = 0.98
eta_ref = 0.92
eta_at = 0.978
dni = 1.05  # 来不及算了


# 将指定的条件下的镜场排列输出到列表中
def plot_coordinate(r,R,d,circle_amount,ro):
    # 圆的半径,内圆间隔，点的间隔，圆的个数, ro 递增参数
    # R目前为无用参数，后续可用于绘制卡卡大厦的外圈，目前用f(r)代替，k控制内圆的个数

    # 绘制一个以原点为中心，半径为r的圆
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 4

    radius = 100
    # 在x轴100之后半径递增生成一个个圆圈
    for m in range(0, circle_amount): 
        radius = radius * ro
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

    # 定义一个空列表，用于存储所有点的坐标
    points = []
    # 定义一个变量，用于统计点的个数
    count = 0

    # 在各圆圈上填充定日镜的点
    print("ro is:", ro)
    for m in range(0, circle_amount):
        # 计算当前圆圈的半径
        radius = (100 * (1.2 ** m) * ro)
        # 计算当前圆圈上可以放置多少个点
        n = int(np.floor(2 * np.pi * radius / d))
        # 计算当前圆圈上每个点之间的角度差
        delta = 2 * np.pi / n
        # 遍历当前圆圈上的每个点
        for circle_amount in range(n):
            # 计算当前点的角度
            angle = circle_amount * delta
            # 计算当前点的坐标
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 4
            # 将当前点的坐标添加到列表中
            points.append((x, y, z))

            # 点的个数加一
            count += 1
    # print("The coordinates of all the points are:")
    # for point in points:
    #     print(point)
    print("The total number of points is:", count)
    
    return points


# 输入为包含坐标列表，输出为适应度值
def calc_fitness(point_list):
    # 初始化一个变量，用于存储适应度值
    fitness = 0
    heli_eta_cos_output = []
    e_field_per_area_output = []
    # 循环遍历坐标列表中的每一个坐标
    for point in point_list:
        tmp = p1.Station(point, tower_para)
        heli_eta_cos_output.append(tmp.eta_cos)

        # 计算光学效率
        eta = tmp.eta_cos * eta_sb * eta_trunc * eta_ref * eta_at
        # 计算单位面积输出热功率
        e_field_per_area = dni * eta
        e_field_per_area_output.append(e_field_per_area)

    fitness = sum(e_field_per_area_output) / len(e_field_per_area_output)
    # 返回适应度值
    print(fitness)
    return fitness


# test_points = plot_coordinate(200,5,10,10)
# test_fitness = fitness(test_points)
# print(test_fitness)

# 请bing开秒杀

# 定义圆的半径,内圆间隔，点的间隔，圆的个数k为固定值
r = 700
R = 5
d = 10
circle_amount = 10

# 定义ro的取值范围和精度
ro_min = 1
ro_max = 2
ro_precision = 0.01

# 定义AGSA的参数
N = 10 # 种群规模
G0 = 100 # 初始引力常数
alpha = 20 # 引力衰减因子
beta = 2 # 惯性因子
max_iter = 3 # 最大迭代次数

# 初始化种群和适应度值
population = np.random.uniform(ro_min, ro_max, N) # 随机生成N个ro值
print(population)
fitness_values = np.zeros(N) # 初始化适应度值为零

# 计算初始适应度值
for i in range(N):
    fitness_values[i] = calc_fitness(plot_coordinate(r,R,d,circle_amount,population[i])) # 调用fitness_func函数

# 记录最佳适应度值和最佳ro值
best_fitness = np.max(fitness_values) # 最佳适应度值
best_ro = population[np.argmax(fitness_values)] # 最佳ro值

# 进行迭代优化
for iter in range(max_iter):
    # 计算当前引力常数
    G = G0 * np.exp(-alpha * iter / max_iter)

    # 计算每个个体的质量和总质量
    mass = (fitness_values - np.min(fitness_values)) / (np.max(fitness_values) - np.min(fitness_values)) # 归一化适应度值作为质量
    total_mass = np.sum(mass) # 总质量

    # 计算每个个体的加速度
    acc = np.zeros(N) # 初始化加速度为零
    for i in range(N):
        for j in range(N):
            if i != j: # 排除自身对自身的作用力
                Fij = G * mass[i] * mass[j] / (population[j] - population[i] + 1e-6) ** 2 # 计算两个个体之间的引力（加1e-6防止分母为零）
                if mass[i] != 0 and abs(Fij) > 1e-10:  # use a small threshold to avoid numerical errors
                    acc[i] += Fij / mass[i]  # calculate the acceleration of the i-th individual
                else:
                    acc[i] += 0  # set the acceleration to zero if the division is invalid

    # 更新每个个体的速度和位置（即ro值）
    for i in range(N):
        vel = random.random() * beta * acc[i] # 计算第i个个体的速度（乘以随机数和惯性因子）
        population[i] += vel # 更新第i个个体的位置（加上速度）
        population[i] = max(min(population[i], ro_max), ro_min) # 将位置限制在ro的取值范围内

    # 计算更新后的适应度值
    for i in range(N):
        fitness_values[i] = calc_fitness(plot_coordinate(r,R,d,circle_amount,population[i])) # 调用fitness_func函数

    # 更新最佳适应度值和最佳ro值
    if np.max(fitness_values) > best_fitness:
        best_fitness = np.max(fitness_values)
        best_ro = population[np.argmax(fitness_values)]

    # 打印当前迭代次数和最佳结果
    print("Iteration:", iter + 1)
    print("Best fitness:", best_fitness)
    print("Best ro:", best_ro)

# 输出最终结果
print("The optimal ro value is:", best_ro)
print("The maximum fitness value is:", best_fitness)

# 将ro值代入函数中，绘制最佳结果

best_points = plot_coordinate(r,R,d,circle_amount,best_ro)

# 将列表中的坐标分别赋值给x, y, z变量
x = [p[0] for p in best_points]
y = [p[1] for p in best_points]
z = [p[2] for p in best_points]

# 创建一个图形对象
fig = plt.figure()
# 在图形对象中添加一个三维坐标系
ax = fig.add_subplot(111, projection='3d')
# 在三维坐标系中绘制散点图，使用红色圆点表示
ax.scatter(x, y, z, c='r', marker='o')
# 设置坐标轴的标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 显示图形
plt.show()