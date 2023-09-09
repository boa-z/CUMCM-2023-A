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
        'solar_altitude_angle': as_degrees,
        'solar_azimuth_angle': gamma_s_degrees,
        'solar_declination_angle': math.degrees(delta),
        'dni': dni
    }


# TODO: what is this?
# def calc_cos_eff(solar_angles, observer_location):  # cosine efficiency
#     # Extract solar angles
#     as_radians = np.radians(solar_angles['solar_altitude_angle'])
#     gamma_s_radians = np.radians(solar_angles['solar_azimuth_angle'])
#
#     # Calculate i vector
#     # i = np.array([
#     #     -np.cos(as_radians) * np.sin(gamma_s_radians),
#     #     -np.cos(as_radians) * np.cos(gamma_s_radians),
#     #     -np.sin(as_radians)
#     # ])
#     i = np.array([
#         np.cos(as_radians) * np.sin(gamma_s_radians),
#         np.cos(as_radians) * np.cos(gamma_s_radians),
#         np.sin(as_radians)
#     ])
#
#     # Convert observer_location to a numpy array
#     AO = np.array(observer_location)
#
#     # Calculate n vector
#     i_plus_AO = i + AO
#     n = i_plus_AO / np.linalg.norm(i_plus_AO)
#
#     # Calculate cosine of the angle between i and n
#     eta_cosine = np.dot(i, n)
#
#     return {
#         'n_vector': n,
#         'i_vector': i,
#         'eta_cosine': eta_cosine
#     }

def calc_eta_cos(local_solar_altitude, local_solar_azimuth, AO):
    # Calculate unit vector i
    # alpha_s = np.radians(90 - local_solar_altitude)  # Convert solar altitude angle to radians
    # gamma_s = np.radians(360 - local_solar_azimuth)  # Convert solar azimuth angle to radians
    alpha_s = np.radians(local_solar_altitude)
    gamma_s = np.radians(local_solar_azimuth)
    # i = np.array([-np.cos(alpha_s) * np.sin(gamma_s),
    #               -np.cos(alpha_s) * np.cos(gamma_s),
    #               -np.sin(alpha_s)])

    i = np.array([np.cos(alpha_s) * np.sin(gamma_s),
                  np.cos(alpha_s) * np.cos(gamma_s),
                  np.sin(alpha_s)])
    # r = np.array([-x, -y, 80])
    # r中镜子的坐标

    # Calculate unit vector n
    norm_AO = np.linalg.norm(AO)
    normalized_AO = AO / norm_AO
    n = (i + normalized_AO) / np.linalg.norm(i + normalized_AO)
    # print(i, n)

    # Calculate cosine of the angle theta (eta_cos)
    eta_cos = np.dot(i, n)

    return n, i, eta_cos


# 阴影遮挡
def point(w, h, x0, y0, h0, theta_s, theta_z):
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
    return 0


# heliostat n. 定日镜

class Station:
    def __init__(self, heli_para, tower_para):
        # heli_x, heli_y, heli_z, heli_h, heli_w,
        # self.heli_x = heli_x  # ~ heli, center
        # self.heli_y = heli_y
        # self.heli_z = heli_z
        # self.heli_h = heli_h  # height
        # self.heli_w = heli_w  # wight
        self.heli_para = heli_para  # 创建一个列表表示所有定日镜参数
        self.tower_para = tower_para  # 创建一个列表表示集热塔的参数
        # self.tower_x = tower_x
        # self.tower_y = tower_y
        # self.tower_z = tower_y

        heli_loc = np.array([heli_para[0], heli_para[1], heli_para[2]])
        tower_loc = np.array([tower_para[0], tower_para[1], tower_para[2]])
        heli_loc = np.array(heli_loc)
        tower_loc = np.array(tower_loc)
        dhr = np.linalg.norm(heli_loc - tower_loc)
        # print("dhr:" + str(dhr))

        ao = np.array([tower_para[0] - heli_para[0],  tower_para[1] - heli_para[1],
                       tower_para[2] - heli_para[2]])  # mirror to receiver
        # print("ao:" + str(ao))
        norm_ao = np.linalg.norm(ao)  # calculate the norm of AO
        ao_normalized = ao / norm_ao  # divide AO by its norm

        solar_altitude_angle, solar_azimuth_angle, solar_declination_angle, dni = calc_solar(local_time, latitude,
                                                                                             elevation)
        solar_data = calc_solar(local_time, latitude, elevation)
        # print("solar_data:" + str(solar_data))
        solar_altitude_angle = solar_data['solar_altitude_angle']
        solar_azimuth_angle = solar_data['solar_azimuth_angle']
        n, i, self.eta_cos = calc_eta_cos(solar_altitude_angle, solar_azimuth_angle, ao)

    def add_heli(self, heli):
        self.heli_para.append(heli)


heli_test = Station([114, 114, 4], [0, 0, 84])

# 计算所有镜子的平均效率啥的
# prompt: 下面是一个关于Station的类，其中初始化输入 heli_para、tower_para 为列表，表示x,y,z轴坐标。
# heli_para 的x,y从df中读取，df的格式见附录，z为定值4，tower_para = [0, 0, 84]
# 请写一段python代码，通过.eta_cos可以获取 eta_cos的值，遍历df所有的值，输出到列表 heli_output 中
#
# 附录：
#       x坐标 (m)  y坐标 (m)
# 0     107.250   11.664
# 1     105.360   23.191
# 2     102.235   34.447

# 设置起始日期，这里以2023年1月21日为例
start_date = datetime(2023, 1, 21)

# 设置结束日期，这里以2023年12月21日为例
end_date = datetime(2023, 12, 21)

# 定义要打印的时间列表
times = ["9:00", "10:30", "12:00", "13:30", "15:00"]


avg_month_eta_cos = []

# 循环遍历每个月的21日
current_date = start_date
while current_date <= end_date:
    avg_times_eta_cos = []
    # 打印当前日期的每个时间
    for time in times:
        hour, minute = map(int, time.split(':'))
        local_time = datetime(current_date.year, current_date.month, 21, hour, minute, 0)
        heli_eta_cos_output = []

        for i, row in df.iterrows():
            heli_para = [row['x坐标 (m)'], row['y坐标 (m)'], 4]
            tower_para = [0, 0, 84]
            station = Station(heli_para, tower_para)
            heli_eta_cos_output.append(station.eta_cos)
            print(len(heli_eta_cos_output))

        # 计算平均值并添加到平均时间列表
        avg_time_eta_cos = sum(heli_eta_cos_output) / len(heli_eta_cos_output)
        avg_times_eta_cos.append(avg_time_eta_cos)
        # print(local_time, avg_time_eta_cos)
    # 计算5个时间的平均值的平均值并添加到月平均列表
    avg_month_eta_cos.append(sum(avg_times_eta_cos) / len(avg_times_eta_cos))
    print(avg_month_eta_cos)

    # 增加一个月
    if current_date.month == 12:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(month=current_date.month + 1)

# for i, row in df.iterrows():
#     heli_para = [row['x坐标 (m)'], row['y坐标 (m)'], 4]
#     tower_para = [0, 0, 84]
#     station = Station(heli_para, tower_para)
#     heli_output.append(station.eta_cos)
#
# print(heli_output)
# avg = sum(heli_output) / len(heli_output)
# print(avg)