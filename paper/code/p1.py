from typing import List, Tuple, Any

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/boa/Documents/2023-2024-1暑假/CUMCM2023/data/附件.xlsx')

D = 0

latitude = 39.4128  # 纬度
elevation = 3  # 海拔
local_time = datetime(2023, 6, 21, 9, 0)  # 当地时间，原题中ST

boa_error = []  # 用来存储计算出错的镜面的索引

eta_ref = 0.9  # 镜面反射率

split_num = 6  # 镜面拆分的份数

tower_box_vertices = np.array([
    [3.5, 3.5, 88],
    [-3.5, 3.5, 88],
    [3.5, -3.5, 88],
    [-3.5, -3.5, 88],
    [3.5, 3.5, 80],
    [-3.5, 3.5, 80],
    [3.5, -3.5, 80],
    [-3.5, -3.5, 80],
])


# 输入当地时间和纬度，输出太阳高度角、太阳方位角、太阳赤纬角
def calc_solar(local_time, latitude, elevation):
    # Constants
    G0 = 1366  # Solar constant in W/m^2 (太阳常数)
    # H = elevation / 1000  # Convert elevation from meters to kilometers
    H = elevation

    # Calculate D, the number of days since spring equinox (March 21)
    spring_equinox = 21  # March 21
    if local_time.month < 3 or (local_time.month == 3 and local_time.day < spring_equinox):
        D = local_time.timetuple().tm_yday + (365 - (spring_equinox - 1))
    else:
        D = local_time.timetuple().tm_yday - (spring_equinox - 1)

    # Calculate the solar declination angle (delta)
    sin_delta = math.sin(2 * math.pi * D / 365) * math.sin(2 * math.pi * 23.45 / 360)
    delta = math.asin(sin_delta)

    # Calculate the solar hour angle (omega)
    solar_hour_angle = math.radians((180 / 12) * (local_time.hour - 12))

    # Calculate the solar altitude angle (a_s)
    sin_as = math.cos(math.radians(latitude)) * math.cos(delta) * math.cos(solar_hour_angle) + \
             math.sin(math.radians(latitude)) * math.sin(delta)
    cos_as = math.sqrt(1 - sin_as ** 2)

    as_radians = math.asin(sin_as)
    as_degrees = math.degrees(as_radians)

    # Calculate the solar azimuth angle (gamma_s)
    cos_gamma_s = (math.sin(delta) - math.sin(as_radians) * math.sin(math.radians(latitude))) / \
                  (math.cos(as_radians) * math.cos(math.radians(latitude)))

    if local_time.hour < 12:
        sin_gamma_s = math.sqrt(1 - cos_gamma_s ** 2)
    elif local_time.hour == 12:
        sin_gamma_s = 0
    else:
        sin_gamma_s = math.sqrt(1 - cos_gamma_s ** 2)

    if sin_gamma_s < 0:
        print(sin_gamma_s)

    # Calculate DNI using the formula
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    dni = G0 * (a + b * math.exp(-c / sin_as))

    return {
        # as_degrees, gamma_s_degrees, math.degrees(delta), dni
        # as_radians, gamma_s_radians, delta, dni
        # 指出不要计算度数，直接计算输出cos/sin值
        sin_as, cos_as, cos_gamma_s, sin_gamma_s, sin_delta, dni
    }

# 计算镜面法向量 n 和入射光线反方向的单位向量 i , method 1
def calc_ni_m1(AO):
    sin_as, cos_as, cos_gamma_s, sin_gamma_s, _, _ = calc_solar(local_time, latitude, elevation)
    # Calculate unit vector i
    # i = np.array([-cos_as * sin_gamma_s,
    #               -cos_as * cos_gamma_s,
    #               -sin_as])
    i = np.array([cos_as * sin_gamma_s,
                  cos_as * cos_gamma_s,
                  sin_as])

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
def mirror_point(heli_para) -> 'tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]':
    sin_as, cos_as, cos_gamma_s, sin_gamma_s , _ , _ = calc_solar(local_time, latitude, elevation)
    x0, y0, h0, w, h = heli_para[:5]
    # 计算m
    m = math.sqrt(x0 ** 2 + y0 ** 2 + h0 ** 2)

    # 计算theta_z
    numerator_theta_z = sin_as * m + h0
    denominator_theta_z = math.sqrt(
        x0 ** 2 + y0 ** 2 + m ** 2 * cos_as ** 2 - 2 * cos_as * m * (
                    x0 * sin_gamma_s - y0 * cos_as))

    theta_z = math.atan(numerator_theta_z / denominator_theta_z)

    # 计算theta_s
    numerator_theta_s = x0 - cos_as * sin_gamma_s * m
    denominator_theta_s = math.sqrt(
        x0 ** 2 + y0 ** 2 + m ** 2 * cos_as ** 2 - 2 * cos_as * m * (
                    x0 * sin_gamma_s - y0 * cos_as))

    # TODO: asin 会遇到参数超过 -1,1 的情况，需要处理
    if numerator_theta_s / denominator_theta_s > 1 or numerator_theta_s / denominator_theta_s < -1:
        print(heli_para)
        boa_error.append(heli_para)
        theta_s = math.asin(1)
    else:
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


# 镜面拆分
def mirror_split(heli_para) -> 'list[np.ndarray]':
    # split_num 镜面拆分的份数
    x_0, y_0, z_0, w, h = heli_para[:5]

    O = np.array([x_0, y_0, z_0])
    A, B, C, D = mirror_point(heli_para)

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


# Möller–Trumbore 算法，已知长方体的所有顶点，一条射线的起点和方向向量，求射线与长方体是否有交点
def ray_box_intersection(ray_origin, ray_direction, box_vertices):
    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        if abs(ray_direction[i]) < 1e-6:
            if ray_origin[i] < min(box_vertices[:, i]) or ray_origin[i] > max(box_vertices[:, i]):
                return False
        else:
            t1 = (min(box_vertices[:, i]) - ray_origin[i]) / ray_direction[i]
            t2 = (max(box_vertices[:, i]) - ray_origin[i]) / ray_direction[i]
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))

    return t_near <= t_far and t_far >= 0


# 计算阴影。计算遮挡有重复，需要移除重复的小块
def calc_shadow(heli_para, tower_para):
    pre_shadows = find_points_within_distance(df, heli_para[:3], 2, 20)  # 可能干扰的坐标的合集
    shadow_count = 0

    # little_mirror = Station(heli_para, tower_para)
    little_mirror_centers = mirror_split(heli_para)
    # 将ndarray 转化为 list
    little_mirror_centers = little_mirror_centers.tolist()
    for pre_shadow in pre_shadows:
        little_mirror_blocked = []
        for little_mirror_center in little_mirror_centers:
            # 计算镜面中心到接受塔中心的单位向量
            little_mirror_center_to_tower = np.array([tower_para[0] - little_mirror_center[0], tower_para[1] - little_mirror_center[1], tower_para[2] - little_mirror_center[2]])
            norm_little_mirror_center_to_tower = np.linalg.norm(little_mirror_center_to_tower)

            pre_shadow_full = pre_shadow
            a, b, c, d = mirror_point(pre_shadow_full)

            # 计算拆分后镜面中心到接受塔中心的单位向量
            little_mirror_center_to_tower = np.array([tower_para[0] - heli_para[0], tower_para[1] - heli_para[1], tower_para[2] - heli_para[2]])
            norm_little_mirror_center_to_tower = np.linalg.norm(little_mirror_center_to_tower)  # calculate the norm of AO
            little_mirror_center_to_tower_normalized = little_mirror_center_to_tower / norm_little_mirror_center_to_tower
            # reflect_vector = little_mirror.heli_para[5][0]  # 入射光线反方向的单位向量 i，此处疑似有误
            reflect_vector = calc_ni_m1(little_mirror_center_to_tower)[1]

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
            little_mirror_centers.remove(little_mirror_center)
            shadow_count += 1

    if shadow_count == split_num*split_num:
        etc_sb = 1
    else:
        etc_sb = shadow_count / (split_num*split_num) + shadow_tower()  # 阴影遮挡效率

    return etc_sb


# 计算接收塔的阴影影响
def shadow_tower():
    sin_as, cos_as, cos_gamma_s, sin_gamma_s, sin_delta, dni = calc_solar(local_time, latitude, elevation)
    solar_altitude_angle = math.asin(sin_as)
    tmp = (80 * math.tan(solar_altitude_angle) - 96.5)
    if tmp > 0:
        s = tmp
    else:
        s = 0
    return s


# 计算集热器截断效率
# TODO: 已知长方体的所有顶点，一条射线的起点，方向向量，求射线与长方体是否有交点，可以使用什么算法？
def collector_cut_off_efficiency(heli_para, tower_para):
    origin_mirror = Station(heli_para, tower_para)
    origin_mirror_n, origin_mirror_i = calc_ni_m1(origin_mirror.ao)
    little_mirror_r = calc_r_m1(origin_mirror_n, origin_mirror_i) # 根据原镜面的法向量和入射光线反方向的单位向量计算拆分后小镜面反射光线的方向向量

    little_mirror_centers = mirror_split(origin_mirror.heli_para)
    collector_cut_off_count = 0
    for little_mirror_center in little_mirror_centers:
        if ray_box_intersection(little_mirror_center, little_mirror_r, tower_box_vertices):
            collector_cut_off_count += 1
            # print(little_mirror_center, collector_cut_off_count) # 看看小块计数

    etc_trunc = 1 - collector_cut_off_count / (split_num * split_num - collector_cut_off_count)  # collector_cut_off_efficiency
    return etc_trunc


# 大气透射率
def calc_eta_at(heli_para, tower_para):
    origin_mirror = Station(heli_para, tower_para)
    dhr = origin_mirror.dhr
    eta_at = 0.993121 - 0.0001176 * dhr + 1.97e-8 * dhr ** 2
    return eta_at

# 计算光学效率 eta = eta_cos * eta_sb * eta_trunc * eta_ref * eta_at
def each_mirror_output_all(heli_para, tower_para):
    eta_cos = Station(heli_para, tower_para).eta_cos
    eta_sb = calc_shadow(heli_para, tower_para)
    eta_trunc = collector_cut_off_efficiency(heli_para, tower_para)
    eta_at = calc_eta_at(heli_para, tower_para)
    eta = eta_cos * eta_sb * eta_trunc * eta_ref * eta_at

    # 计算单位面积镜面平均输出热功率 E_{field} = DNI \times \sum_{i}^{N} A_i \eta_i
    # 此处计算后，真的 e_field 还需要求和
    # heliostat_lighting_area 直接安排上， 6*6 即可
    _, _, _, _, _, dni = calc_solar(local_time, latitude, elevation)
    e_field = dni * split_num * split_num * eta
    mirror_output_all = [eta_cos, eta_sb, eta_trunc, eta_at, eta, e_field]
    # print(mirror_output_all)
    # print(dni)
    return mirror_output_all

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
        self.dhr = np.linalg.norm(heli_loc - tower_loc)

        # 计算镜面中心到接受塔中心的单位向量
        ao = np.array([tower_para[0] - heli_para[0],  tower_para[1] - heli_para[1],
                       tower_para[2] - heli_para[2]])  # mirror to receiver

        norm_ao = np.linalg.norm(ao)  # calculate the norm of AO
        ao_normalized = ao / norm_ao  # divide AO by its norm
        self.ao = ao_normalized

        n, i = calc_ni_m1(ao)
        self.eta_cos: 'float' = calc_eta_cos(n, i)

        # self.heli_para[5].append(n), self.heli_para[5].append(i)
        # 现在的问题是每次查找似乎都会append一次，导致列表中有很多重复的元素，故删除这一功能


# 答辩测试区
heli_test = Station((-97.911, -45.299, 4, 6, 6, []), [0, 0, 84]) # 这个点会导致 sin 出错
