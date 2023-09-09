import math
import pandas as pd
import matplotlib.pyplot as plt

def calculate_mirror_positions(R1, DM, Nhel1, LH, LW, desp):
    def calculate_delta_alpha(R1, DM):
        return 2 * math.asin(DM / (2 * R1))

    def calculate_min_increment(DM, R1):
        h = R1 - math.sqrt(R1**2 - (DM**2) / 4)
        return DM * math.cos(math.radians(30)) - h

    def calculate_Nhel_n(Nhel1, n):
        return Nhel1 * (2**(n - 1))

    def calculate_R_n(R1, n):
        return R1 * (2**(n - 1))

    def calculate_Nrows_n(R1, delta_R_min, n):
        return int((2**(n - 1) * R1) / delta_R_min)

    delta_alpha_z1 = calculate_delta_alpha(R1, DM)
    delta_R_min = calculate_min_increment(DM, R1)
    
    result = []
    for n in range(1, Nhel1 + 1):
        Nhel_n = calculate_Nhel_n(Nhel1, n)
        R_n = calculate_R_n(R1, n)
        Nrows_n = calculate_Nrows_n(R1, delta_R_min, n)
        
        for i in range(Nrows_n):
            row_radius = R_n - i * delta_R_min
            for j in range(Nhel_n):
                theta = j * (2 * math.pi / Nhel_n)
                x = row_radius * math.cos(theta)
                y = row_radius * math.sin(theta)
                result.append((x, y))
    
    return result

def create_mirror_field_dataframe(R1, DM, Nhel1, LH, LW, desp):
    mirror_positions = calculate_mirror_positions(R1, DM, Nhel1, LH, LW, desp)
    df = pd.DataFrame(mirror_positions, columns=['X', 'Y'])
    return df

def plot_mirror_field(df):
    plt.figure(figsize=(10, 10))  # 设置图像大小
    plt.scatter(df['X'], df['Y'], marker='o', s=20, color='blue')  # 绘制散点图
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('镜场布置')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴等比例
    plt.show()


# 示例用法
R1 = 350  # 镜场首行的半径
DM = 10   # 定日镜的特征圆直径
Nhel1 = 6  # 镜场首行的定日镜数量
LH = 2    # 定日镜的宽度
LW = 4    # 定日镜的长度
desp = 5  # 安全距离

df = create_mirror_field_dataframe(R1, DM, Nhel1, LH, LW, desp)
plot_mirror_field(df)