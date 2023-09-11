# 导入matplotlib库，用于绘制图形
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math
import random
import p1

tower_para = (0, 0, 84)
eta_sb = 0.9
eta_trunc = 0.98
eta_ref = 0.92
eta_at = 0.978
dni = 1.05


# 将指定的条件下的镜场排列输出到列表中
def plot_coordinate(r,w,h,z,circle_amount,ro):
    # update: 
    # R目前为无用参数，后续可用于绘制卡卡大厦的外圈，目前用f(ro)代替，k控制内圆的个数

    # 绘制一个以原点为中心，半径为r的圆
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

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
    d = math.sqrt(w**2 + h**2)
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
            # 将当前点的坐标添加到列表中
            points.append((x, y, z, w, h))

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

# 定义圆的半径,内圆间隔，点的间隔，圆的个数k为固定值
r = 700

# 定义相关参数的取值范围和精度
ro_min = 1
ro_max = 2
ro_precision = 0.01
w_min = 2
w_max = 10
w_precision = 0.1
h_min = 2
h_max = 8
h_precision = 0.1
z_min = 2
z_max = 6
z_precision = 0.1

# 定义搜索空间的维度和大小
dim = 5 # 每个解包含w,h,z,circle_amount,ro五个变量
size = 100 # 搜索空间中的解的数量

# 定义AGSA的参数
G0 = 100 # 初始引力常数
alpha = 20 # 引力常数衰减因子
beta = 2 # 引力指数因子
max_iter = 100 # 最大迭代次数


# 定义一个函数，根据取值范围和精度生成一个随机解
def generate_random_solution():
    # w = round(random.uniform(w_min, w_max), w_precision)
    # h = round(random.uniform(h_min, h_max), h_precision)
    # z = round(random.uniform(z_min, z_max), z_precision)
    w = random.uniform(w_min, w_max)
    h = random.uniform(h_min, h_max)
    z = random.uniform(z_min, z_max)
    circle_amount = random.randint(1, 10)
    ro = random.uniform(ro_min, ro_max), ro_precision
    # ro = round(random.uniform(ro_min, ro_max), ro_precision)
    return [w, h, z, circle_amount, ro]

# 定义一个函数，根据适应度值计算质量值
def calc_mass(fitness):
    min_fitness = min(fitness)
    max_fitness = max(fitness)
    if min_fitness == max_fitness:
        return [1] * size # 如果所有解的适应度值相同，则质量值都为1
    else:
        mass = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness] # 否则，按照公式计算质量值
        mass_sum = sum(mass)
        mass = [m / mass_sum for m in mass] # 归一化质量值，使其和为1
        return mass

# 定义一个函数，根据质量值和引力常数计算引力矢量和加速度矢量
def calc_force_and_acceleration(mass, G):
    force = np.zeros((size, dim)) # 初始化引力矢量矩阵，每一行表示一个解受到的引力矢量，每一列表示一个维度上的分量
    acceleration = np.zeros((size, dim)) # 初始化加速度矢量矩阵，每一行表示一个解受到的加速度矢量，每一列表示一个维度上的分量
    for i in range(size): # 遍历每个解
        for j in range(size): # 遍历每个其他解
            if i != j: # 如果不是同一个解，则计算引力作用
                distance = np.linalg.norm(space[i] - space[j]) # 计算两个解之间的欧氏距离
                if distance > 0: # 如果距离不为零，则计算引力分量
                    force[i] += random.random() * G * mass[i] * mass[j] * (space[j] - space[i]) / distance ** beta # 按照公式计算引力分量，并乘以一个随机因子[0,1]
        acceleration[i] = force[i] / mass[i] # 按照公式计算加速度分量，加速度与质量成反比
    return force, acceleration

# 定义一个函数，根据加速度矢量和随机因子更新解的位置，并保证解在取值范围内
def update_position(acceleration):
    global space # 声明全局变量，搜索空间
    for i in range(size): # 遍历每个解
        for j in range(dim): # 遍历每个维度
            space[i][j] += acceleration[i][j] * random.random() # 按照公式更新解的位置，并乘以一个随机因子[0,1]
            # 保证解在取值范围内，如果超出范围，则重新生成一个随机值
            if j == 0: # w的取值范围
                if space[i][j] < w_min or space[i][j] > w_max:
                    space[i][j] = round(random.uniform(w_min, w_max), w_precision)
            elif j == 1: # h的取值范围
                if space[i][j] < h_min or space[i][j] > h_max:
                    space[i][j] = round(random.uniform(h_min, h_max), h_precision)
            elif j == 2: # z的取值范围
                if space[i][j] < z_min or space[i][j] > z_max:
                    space[i][j] = round(random.uniform(z_min, z_max), z_precision)
            elif j == 3: # circle_amount的取值范围
                if space[i][j] < 1 or space[i][j] > 10:
                    space[i][j] = random.randint(1, 10)
            elif j == 4: # ro的取值范围
                if space[i][j] < ro_min or space[i][j] > ro_max:
                    space[i][j] = round(random.uniform(ro_min, ro_max), ro_precision)

# 初始化搜索空间，生成size个随机解
space = np.array([generate_random_solution() for _ in range(size)])

# 初始化适应度值列表，计算每个解的适应度值
fitness = [calc_fitness(plot_coordinate(r, *s)) for s in space]

# 初始化最优解和最优适应度值
best_solution = space[np.argmax(fitness)]
best_fitness = max(fitness)

# 初始化迭代次数和引力常数
iter = 0
G = G0

# 进入主循环，直到达到最大迭代次数或引力常数趋近于零
while iter < max_iter and G > 0.001:
    # 计算每个解的质量值
    mass = calc_mass(fitness)
    # 计算每个解受到的引力矢量和加速度矢量
    force, acceleration = calc_force_and_acceleration(mass, G)
    # 更新每个解的位置
    update_position(acceleration)
    # 计算每个解的适应度值
    fitness = [calc_fitness(plot_coordinate(r, *s)) for s in space]
    # 更新最优解和最优适应度值，如果有更好的解出现，则替换之前的最优解和最优适应度值
    current_best_solution = space[np.argmax(fitness)]
    current_best_fitness = max(fitness)
    if current_best_fitness > best_fitness:
        best_solution = current_best_solution
        best_fitness = current_best_fitness
    # 更新迭代次数和引力常数，引力常数按照指数衰减规律更新
    iter += 1
    G = G0 * np.exp(-alpha * iter / max_iter)

# 输出最优解和最优适应度值
print("The best solution is:", best_solution)
print("The best fitness value is:", best_fitness)

# 将ro值代入函数中，绘制最佳结果


# # 将列表中的坐标分别赋值给x, y, z变量
# x = [p[0] for p in best_points]
# y = [p[1] for p in best_points]
# z = [p[2] for p in best_points]

# # 创建一个图形对象
# fig = plt.figure()
# # 在图形对象中添加一个三维坐标系
# ax = fig.add_subplot(111, projection='3d')
# # 在三维坐标系中绘制散点图，使用红色圆点表示
# ax.scatter(x, y, z, c='r', marker='o')
# # 设置坐标轴的标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # 显示图形
# plt.show()

# best_points_for_output = {
#     'w': [p[3] for p in best_points],
#     'h': [p[4] for p in best_points],
#     'x': [p[0] for p in best_points],
#     'y': [p[1] for p in best_points],
#     'z': [p[2] for p in best_points],
# }
# df = pd.DataFrame(best_points_for_output)
# csv_name = 'best_points.csv'
# df.to_csv(csv_name, mode='w', header=False)