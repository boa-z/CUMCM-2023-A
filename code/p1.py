from typing import List, Tuple, Any

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/boa/Documents/2023-2024-1暑假/CUMCM2023/data/附件.xlsx')

D = 0

latitude = 39.4128  # 纬度
elevation = 3000  # 海拔
local_time = datetime(2023, 1, 21, 9, 0)  # 当地时间，原题中ST

split_num = 6  # 镜面拆分的份数

# 输入当地时间和纬度，输出太阳高度角、太阳方位角、太阳赤纬角
def calc_solar(local_time, latitude, elevation):
    # Constants
    G0 = 1366  # Solar constant in W/m^2 (太阳常数)
    H = elevation / 1000  # Convert elevation from meters to kilometers

    # Calculate D, the number of days since spring equinox (March 21)
    spring_equinox = 21  # March 21
    if local_time.month < 3 or (local_time.month == 3 and local_time.day < spring_equinox):
        D = local_time.timetuple().tm_yday + (365 - (spring_equinox - 1))
    else:
        D = local_time.timetuple().tm_yday - (spring_equinox - 1)

    # Calculate the solar declination angle (delta)
    sin_delta = math.sin(2 * math.pi * D / 365) * math.sin(2 * math.pi * 23.45 / 360)
    # delta = math.radians(0.39795 * math.sin(math.radians(278.97 + 0.9856 * D)))
    delta = math.asin(sin_delta)

    # Calculate the solar hour angle (omega)
    solar_hour_angle = math.radians((180 / 12) * (local_time.hour - 12))

    # Calculate the solar altitude angle (a_s)
    sin_as = math.cos(math.radians(latitude)) * math.cos(delta) * math.cos(solar_hour_angle) + \
             math.sin(math.radians(latitude)) * math.sin(delta)
    as_radians = math.asin(sin_as)
    as_degrees = math.degrees(as_radians)

    # Calculate the solar azimuth angle (gamma_s)
    cos_gamma_s = (math.sin(delta) - math.sin(as_radians) * math.sin(math.radians(latitude))) / \
                  (math.cos(as_radians) * math.cos(math.radians(latitude)))
    cos_gamma_s = min(1, max(-1, cos_gamma_s))
    gamma_s_radians = math.acos(cos_gamma_s)
    gamma_s_degrees = math.degrees(gamma_s_radians)

    # Calculate DNI using the formula
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    dni = G0 * (a + b * math.exp(-c / math.sin(as_radians)))

    return {
        as_degrees, gamma_s_degrees, math.degrees(delta), dni
    }

# 计算镜面法向量 n 和入射光线反方向的单位向量 i , method 1
def calc_ni_m1(solar_altitude_angle, solar_azimuth_angle, AO):
    # Calculate unit vector i
    alpha_s = np.radians(90 - solar_altitude_angle)  # Convert solar altitude angle to radians
    gamma_s = np.radians(360 - solar_azimuth_angle)  # Convert solar azimuth angle to radians
    i = np.array([-np.cos(alpha_s) * np.sin(gamma_s),
                  -np.cos(alpha_s) * np.cos(gamma_s),
                  -np.sin(alpha_s)])

    # Calculate unit vector n
    norm_AO = np.linalg.norm(AO)
    normalized_AO = AO / norm_AO
    n = (i + normalized_AO) / np.linalg.norm(i + normalized_AO)

    return n, i

# 计算镜面法向量 n 和入射光线反方向的单位向量 i , method 2
# 用于拆分平面后的计算，小平面的法向量和平面法向量相同
# TODO: 未完成
def calc_ni_m2(solar_altitude_angle, solar_azimuth_angle, AO):
    # Calculate unit vector i
    alpha_s = np.radians(90 - solar_altitude_angle)  # Convert solar altitude angle to radians
    gamma_s = np.radians(360 - solar_azimuth_angle)  # Convert solar azimuth angle to radians
    i = np.array([-np.cos(alpha_s) * np.sin(gamma_s),
                  -np.cos(alpha_s) * np.cos(gamma_s),
                  -np.sin(alpha_s)])

    # Calculate unit vector n
    norm_AO = np.linalg.norm(AO)
    normalized_AO = AO / norm_AO
    n = (i + normalized_AO) / np.linalg.norm(i + normalized_AO)

    return n, i

# 计算镜面反射光线的方向向量 r , method 1
def calc_r_m1(n, i):
    r = i - 2 * np.dot(i, n) * n
    r /= np.linalg.norm(r)
    return r

# 计算 eta_cos 的值
def calc_eta_cos(n, i) -> 'float':
    # Calculate cosine of the angle theta (eta_cos)
    eta_cos = np.dot(i, n)

    return eta_cos


# 阴影遮挡

# 计算距离，筛选可能遮挡的点，输出其坐标，并计算
# 计算两点之间的欧氏距离
def calc_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


# 查找距离小于l的点的坐标
def find_points_within_distance(df, input_coord, k, l) -> 'list[tuple[float,float,float,float,float,float,float]]':
    distances: list[tuple[Any, Any, int, int, int, int, list[Any]]] = []

    for index, row in df.iterrows():
        if index != k:
            distance = calc_distance(input_coord, [row['x坐标 (m)'], row['y坐标 (m)'], 4])
            if distance < l:
                distances.append((row['x坐标 (m)'], row['y坐标 (m)'], 4, 6, 6, []))

    return distances


# 计算顶点坐标
def mirror_point(heli_para, alpha_s, gamma_s) -> 'tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]':
    x0, y0, h0, w, h = heli_para[:5]
    # 计算m
    m = math.sqrt(x0 ** 2 + y0 ** 2 + h0 ** 2)

    # 计算theta_z
    numerator_theta_z = math.sin(alpha_s) * m + h0
    denominator_theta_z = math.sqrt(
        x0 ** 2 + y0 ** 2 + m ** 2 * math.cos(alpha_s) ** 2 - 2 * math.cos(alpha_s) * m * (
                    x0 * math.sin(gamma_s) - y0 * math.cos(alpha_s)))
    theta_z = math.atan(numerator_theta_z / denominator_theta_z)

    # 计算theta_s
    numerator_theta_s = x0 - math.cos(alpha_s) * math.sin(gamma_s) * m
    denominator_theta_s = math.sqrt(
        x0 ** 2 + y0 ** 2 + m ** 2 * math.cos(alpha_s) ** 2 - 2 * math.cos(alpha_s) * m * (
                    x0 * math.sin(gamma_s) - y0 * math.cos(alpha_s)))
    theta_s = math.asin(numerator_theta_s / denominator_theta_s)

    x1 = x0 + 0.5 * w * np.cos(theta_s) - 0.5 * w * np.sin(theta_s)
    x2 = x0 - 0.5 * w * np.cos(theta_s) - 0.5 * w * np.sin(theta_s)
    x3 = x0 - 0.5 * w * np.cos(theta_s) + 0.5 * w * np.sin(theta_s)
    x4 = x0 + 0.5 * w * np.cos(theta_s) + 0.5 * w * np.sin(theta_s)
    y1 = y0 + 0.5 * h * np.cos(theta_s) + 0.5 * w * np.sin(theta_s)
    y2 = y0 + 0.5 * h * np.cos(theta_s) - 0.5 * w * np.sin(theta_s)
    y3 = y0 - 0.5 * h * np.cos(theta_s) - 0.5 * w * np.sin(theta_s)
    y4 = y0 - 0.5 * h * np.cos(theta_s) + 0.5 * w * np.sin(theta_s)
    z1 = h0 + 0.5 * h * np.sin(theta_z)
    z2 = h0 + 0.5 * h * np.sin(theta_z)
    z3 = h0 - 0.5 * h * np.sin(theta_z)
    z4 = h0 - 0.5 * h * np.sin(theta_z)
    return np.array([x1, y1, z1]), np.array([x2, y2, z2]), np.array([x3, y3, z3]), np.array([x4, y4, z4])


# 镜面拆分，
def mirror_split(heli_para) -> 'list[np.ndarray]':
    # split_num 镜面拆分的份数
    x_0, y_0, z_0, w, h = heli_para[:5]

    O = np.array([x_0, y_0, z_0])
    solar_altitude_angle, solar_azimuth_angle, _, _ = calc_solar(local_time, latitude, elevation)
    A, B, C, D = mirror_point(heli_para, solar_altitude_angle, solar_azimuth_angle)

    # 计算矩形平面的方向向量
    AB = B - A
    AD = D - A

    # 计算矩形的两个边向量
    u = AB / np.linalg.norm(AB)
    v = AD / np.linalg.norm(AD)

    # 计算每个小矩形的宽度和高度
    delta_u = w / split_num
    delta_v = h / split_num
    # 初始化中心点坐标列表
    data = []

    # 循环生成每个小矩形的中心点坐标、宽度和高度，并存储在一个大的NumPy数组中
    for i in range(split_num):
        for j in range(split_num):
            # 计算每个小矩形的中心点坐标
            center = O + (i - split_num / 2) * delta_u * u + (j - split_num / 2) * delta_v * v
            center_data = np.concatenate((center, np.array([delta_u, delta_v])))
            data.append(center_data)

    little_mirror_centers = np.array(data)

    return little_mirror_centers


# 计算阴影
# TODO: 目前计算遮挡有重复，需要移除重复的小块
def calc_shadow(heli_para, tower_para, solar_altitude_angle, solar_azimuth_angle):
    pre_shadows = find_points_within_distance(df, heli_para[:3], 2, 20)  # 可能干扰的坐标的合集
    print(pre_shadows)
    shadow_count = 0
    for pre_shadow in pre_shadows:
        little_mirror = Station(pre_shadow, tower_para)
        little_mirror_centers = mirror_split(little_mirror.heli_para)
        little_mirror_blocked = []
        print(little_mirror_centers)
        for little_mirror_center in little_mirror_centers:
            # 计算镜面中心到接受塔中心的单位向量
            little_mirror_center_to_tower = np.array([tower_para[0] - little_mirror_center[0], tower_para[1] - little_mirror_center[1], tower_para[2] - little_mirror_center[2]])
            norm_little_mirror_center_to_tower = np.linalg.norm(little_mirror_center_to_tower)

            pre_shadow_full = pre_shadow
            a, b, c, d = mirror_point(pre_shadow_full, solar_altitude_angle, solar_azimuth_angle)

            # 计算镜面中心到接受塔中心的单位向量
            little_mirror_center_to_tower = np.array([tower_para[0] - little_mirror.heli_para[0], tower_para[1] - little_mirror.heli_para[1], tower_para[2] - little_mirror.heli_para[2]])
            norm_little_mirror_center_to_tower = np.linalg.norm(little_mirror_center_to_tower)  # calculate the norm of AO
            little_mirror_center_to_tower_normalized = little_mirror_center_to_tower / norm_little_mirror_center_to_tower
            # reflect_vector = little_mirror.heli_para[5][0]  # 入射光线反方向的单位向量 i，此处疑似有误
            reflect_vector = calc_ni_m1(solar_altitude_angle, solar_azimuth_angle, little_mirror_center_to_tower)[1]

            # Möller–Trumbore 算法，用于判断点是否在三角形内
            s = little_mirror_center_to_tower - a
            e1 = b - a
            e2 = c - a
            s1 = np.cross(reflect_vector, e2)
            s2 = np.cross(s, e1)

            s1e1 = np.dot(s1, e1)
            t = np.dot(s2, e2) / s1e1
            b1 = np.dot(s1, s) / s1e1
            b2 = np.dot(s2, reflect_vector)

            if t >= 0 and b1 >= 0:
                continue
            shadow_count += 1
    print(shadow_count)
    etc_sb = shadow_count / (split_num*split_num - shadow_count)  # 阴影遮挡效率
    return etc_sb


# 计算接收塔的阴影影响
def shadow_tower(solar_altitude_angle):
    tmp = (80 * math.tan(solar_altitude_angle) - 96.5)
    if tmp > 0:
        s = tmp
    else:
        s = 0
    return s

# 计算集热器截断效率
# TODO: 已知长方体的所有顶点，一条射线的起点，方向向量，求射线与长方体是否有交点，可以使用什么算法？
def collector_cut_off_efficiency():

    pass

# heliostat n. 定日镜

class Station:
    def __init__(self, heli_para, tower_para):
        # heli_x, heli_y, heli_z, heli_h, heli_w, []
        self.heli_para: 'tuple' = heli_para  # 创建一个元组表示所有定日镜参数，元组中的列表用来放法向量或者其他参数
        self.tower_para: 'tuple' = tower_para  # 创建一个元组表示集热塔的参数

        heli_loc = np.array([heli_para[0], heli_para[1], heli_para[2]])
        tower_loc = np.array([tower_para[0], tower_para[1], tower_para[2]])
        heli_loc = np.array(heli_loc)
        tower_loc = np.array(tower_loc)
        dhr = np.linalg.norm(heli_loc - tower_loc)

        # 计算镜面中心到接受塔中心的单位向量
        ao = np.array([tower_para[0] - heli_para[0],  tower_para[1] - heli_para[1],
                       tower_para[2] - heli_para[2]])  # mirror to receiver

        norm_ao = np.linalg.norm(ao)  # calculate the norm of AO
        ao_normalized = ao / norm_ao  # divide AO by its norm
        self.ao = ao_normalized

        solar_altitude_angle, solar_azimuth_angle, solar_declination_angle, dni = calc_solar(local_time, latitude,
                                                                                             elevation)
        n, i = calc_ni_m1(solar_altitude_angle, solar_azimuth_angle, ao)
        self.eta_cos: 'float' = calc_eta_cos(n, i)

        # self.heli_para[5].append(n), self.heli_para[5].append(i)
        # 现在的问题是每次查找似乎都会append一次，导致列表中有很多重复的元素，故删除这一功能

# 答辩测试区
heli_test = Station((107.250, 111.664, 4, 6, 6, []), [0, 0, 84])
solar_altitude_angle, solar_azimuth_angle, _, _ = calc_solar(local_time, latitude, elevation)
# mirror_point_test = mirror_point(heli_test.heli_para, solar_altitude_angle, solar_azimuth_angle)
# print(mirror_point_test)

heli_test_shadow_count = calc_shadow(heli_test.heli_para, heli_test.tower_para, solar_altitude_angle, solar_azimuth_angle)
print(str(heli_test_shadow_count))

# 计算所有镜子的平均效率啥的
# prompt: 下面是一个关于Station的类，其中初始化输入 heli_para、tower_para 为列表，表示x,y,z轴坐标。
# heli_para 的x,y从df中读取，df的格式见附录，z为定值4，tower_para = [0, 0, 84]
# 请写一段python代码，通过.eta_cos可以获取 eta_cos的值，遍历df所有的值，输出到列表 heli_output 中

# 设置起始日期，这里以2023年1月21日为例
start_date = datetime(2023, 1, 21)

# 设置结束日期，这里以2023年12月21日为例
end_date = datetime(2023, 12, 21)

# 定义要打印的时间列表
times = ["6:00", "9:00", "10:30", "12:00", "13:30", "15:00"]

heli_output_avg = []
avg_month_eta_cos = []
avg_month_eta_sb = []

# # 循环遍历每个月的21日
# current_date = start_date
# while current_date <= end_date:
#     avg_times_eta_cos = []
#     avg_times_eta_sb = []
#     # 打印当前日期的每个时间
#     for time in times:
#         hour, minute = map(int, time.split(':'))
#         local_time = datetime(current_date.year, current_date.month, 21, hour, minute, 0)
#         heli_eta_cos_output = []
#         heli_eta_sb_output = []
#
#         for i, row in df.iterrows():
#             heli_para = [row['x坐标 (m)'], row['y坐标 (m)'], 4, 6, 6, []]
#             tower_para = [0, 0, 84]
#             station = Station(heli_para, tower_para)
#             heli_eta_cos_output.append(station.eta_cos)
#             solar_altitude_angle, solar_azimuth_angle, _, _ = calc_solar(local_time, latitude, elevation)
#             heli_eta_sb_output.append(calc_shadow(station.heli_para, station.tower_para, solar_altitude_angle, solar_azimuth_angle))
#
#         # 计算平均值并添加到平均时间列表
#         avg_time_eta_cos = sum(heli_eta_cos_output) / len(heli_eta_cos_output)
#         avg_times_eta_cos.append(avg_time_eta_cos)
#         avg_time_eta_sb = sum(heli_eta_sb_output) / len(heli_eta_sb_output)
#         avg_times_eta_sb.append(avg_time_eta_sb)
#         print(local_time, avg_time_eta_cos, avg_time_eta_sb)
#     # 计算5个时间的平均值的平均值并添加到月平均列表
#     avg_month_eta_cos.append(sum(avg_times_eta_cos) / len(avg_times_eta_cos))
#     avg_month_eta_sb.append(sum(avg_times_eta_sb) / len(avg_times_eta_sb))
#     print(avg_month_eta_cos, avg_month_eta_sb)
#
#     # 增加一个月
#     if current_date.month == 12:
#         current_date = current_date.replace(year=current_date.year + 1, month=1)
#     else:
#         current_date = current_date.replace(month=current_date.month + 1)
