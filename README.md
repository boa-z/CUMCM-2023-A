# 备战 CUMCM 2023

## 比赛时间节点

2023年建模国赛的竞赛时间为9月7日（周四）18时至 9月10日（周日）20 时。

[关于2023年“高教社杯”全国大学生数学建模竞赛报名的通知](https://news.shiep.edu.cn/ab/ee/c2679a240622/page.htm)

题目下载与作品提交 [全国大学生数学建模竞赛](https://cumcm.cnki.net/)

## A题题目

构建以新能源为主体的新型电力系统，是我国实现“碳达峰”“碳中和”目标的一项重要 措施。塔式太阳能光热发电是一种低碳环保的新型清洁能源技术[1]。

定日镜是塔式太阳能光热发电站（以下简称塔式电站）收集太阳能的基本组件，其底座由 纵向转轴和水平转轴组成，平面反射镜安装在水平转轴上。纵向转轴的轴线与地面垂直，可以 控制反射镜的方位角。水平转轴的轴线与地面平行，可以控制反射镜的俯仰角，定日镜及底座示意图见图 1。两转轴的交点（也是定日镜中心）离地面的高度称为定日镜的安装高度。塔式 电站利用大量的定日镜组成阵列，称为定日镜场。定日镜将太阳光反射汇聚到安装在镜场中吸 收塔顶端上的集热器，加热其中的导热介质，并将太阳能以热能形式储存起来，再经过热交换 实现由热能向电能的转化。太阳光并非平行光线， 而是具有一定锥形角的一束锥形光线，因此 太阳入射光线经定日镜任意一点的反射光线也是一束锥形光线[2]。定日镜在工作时，控制系统 根据太阳的位置实时控制定日镜的法向，使得太阳中心点发出的光线经定日镜中心反射后指向 集热器中心。集热器中心的离地高度称为吸收塔高度。

现计划在中心位于东经 98.5 ∘ ，北纬 39.4∘ ，海拔 3000 m，半径 350 m 的圆形区域内建设 一个圆形定日镜场（图 2）。以圆形区域中心为原点，正东方向为 $x$ 轴正向，正北方向为 $y$ 轴 正向，垂直于地面向上方向为 $z$ 轴正向建立坐标系，称为镜场坐标系。

规划的吸收塔高度为 80 m，集热器采用高 8 m、直径 7 m 的圆柱形外表受光式集热器。吸 收塔周围 100 m 范围内不安装定日镜，留出空地建造厂房，用于安装发电、储能、控制等设备。 定日镜的形状为平面矩形，其上下两条边始终平行于地面，这两条边之间的距离称为镜面高度， 镜面左右两条边之间的距离称为镜面宽度，通常镜面宽度不小于镜面高度。镜面边长在 2 m 至 8 m 之间，安装高度在 2 m 至 6 m 之间，安装高度必须保证镜面在绕水平转轴旋转时不会触及 地面。由于维护及清洗车辆行驶的需要，要求相邻定日镜底座中心之间的距离比镜面宽度多 5 m 以上。
为简化计算，本问题中所有“年均”指标的计算时点均为当地时间每月 21 日 9:00、10:30、 12:00、13:30、15:00。

请建立模型解决以下问题：
问题 1 若将吸收塔建于该圆形定日镜场中心，定日镜尺寸均为 6 m×6 m，安装高度均为 4 m，且给定所有定日镜中心的位置（以下简称为定日镜位置，相关数据见附件），请计算该定 日镜场的年平均光学效率、年平均输出热功率，以及单位镜面面积年平均输出热功率（光学效 率及输出热功率的定义见附录）。请将结果分别按表 1 和表 2 的格式填入表格。

问题 2 按设计要求，定日镜场的额定年平均输出热功率（以下简称额定功率）为 60 MW。 若所有定日镜尺寸及安装高度相同，请设计定日镜场的以下参数：吸收塔的位置坐标、定日镜 尺寸、安装高度、定日镜数目、定日镜位置，使得定日镜场在达到额定功率的条件下，单位镜 面面积年平均输出热功率尽量大。请将结果分别按表 1、2、3 的格式填入表格，并将吸收塔 的位置坐标、定日镜尺寸、安装高度、位置坐标按模板规定的格式保存到 result2.xlsx 文件中。

问题 3 如果定日镜尺寸可以不同，安装高度也可以不同，额定功率设置同问题 2，请重新 设计定日镜场的各个参数，使得定日镜场在达到额定功率的条件下，单位镜面面积年平均输 出热功率尽量大。请将结果分别按表1、表2 和表3 的格式填入表格，并将吸收塔的位置坐 标、各定日镜尺寸、安装高度、位置坐标按模板规定的格式保存到 result3.xlsx 文件中。

表1

|    日期    | 平均光学效率 | 平均余弦效率 | 平均阴影遮挡效率 | 平均截断效率 | 单位面积镜面平均输出热功率 (kW/m2 ) |
| :--------: | :----------: | :----------: | :--------------: | :----------: | :----------------------------------: |
| 1 月 21 日 |              |              |                  |              |                                      |
| 2 月 21 日 |              |              |                  |              |                                      |
| 3 月 21 日 |              |              |                  |              |                                      |
| 4 月 21 日 |              |              |                  |              |                                      |
| 5 月 21 日 |              |              |                  |              |                                      |
| 6 月 21 日 |              |              |                  |              |                                      |

表2 问题 X 年平均光学效率及输出功率表

| 年平均光学效率 | 年平均余弦效率 | 年平均阴影遮挡效率 | 年平均 截断效率 | 年平均输出热功率(MW) | 单位面积镜面年平均输出热功率 (kW/m2) |
| :------------: | :------------: | :----------------: | :-------------: | :------------------: | :----------------------------------: |
|                |                |                    |                 |                      |                                      |

表3 问题 X 设计参数表

| 吸收塔位置坐标 | 定日镜尺寸 | 安装高度 | 定日镜数目 | 定日镜总面积 |
| :------------: | :--------: | :------: | :--------: | :----------: |
|                |            |          |            |              |

附录 相关计算公式

1 太阳高度角 $a_s$

$$
\sin a_s = \cos \varphi \cos \delta \cos \omega + \sin \varphi \sin \delta
$$

太阳方位角 $\gamma_s$

$$
\cos \gamma_s = \frac{\sin \delta - \sin \alpha_s \sin \varphi}{\cos a_s \cos \varphi}
$$

其中 $\varphi$ 为当地纬度，北纬为正； $\omega$ 为太阳时角

$$
\omega = \frac{\pi}{12} \times (ST - 12)
$$

其中 ST 为当地时间， $\delta$ 为太阳赤纬角[5]

$$
\sin \delta = \sin \frac{2 \pi D}{365} \sin \frac{2 \pi}{360} 23.45
$$

其中 D 为以春分作为第 0 天起算的天数，例如，若春分是 3 月 21 日，则 4 月 1 日对应 D = 11。

2 法向直接辐射辐照度 DNI（单位：kW/m2）是指地球上垂直于太阳光线的平面单位面 上、单位时间内接收到的太阳辐射能量，可按以下公式近似计算[6]

$$
DNI = G_0 [a + b \exp(-\frac{c}{\sin \alpha_s})] \\
a = 0.4237 - 0.00821(6 - H)^2 \\
b = 0.5055 + 0.00595(6.5 - H)^2 \\
c = 0.2711 + 0.01858(2.5 - H)^2 \\
$$

其中 $G_0$ 为太阳常数，其值取为 1.366 kW/m2 ，H 为海拔高度 (单位：km)。

3 定日镜场的输出热功率 $E_{field}$ 为

$$
E_{field} = DNI \times \sum_{i}^{N} A_i \eta_i
$$

其中 DNI 为法向直接辐射辐照度；$N$ 为定日镜总数（单位：面）；$A_i$ 为第 $i$ 面定日镜采光面积（单位：m2 ）； $\eta_i$ 为第 $i$ 面镜子的光学效率。

4 定日镜的光学效率 $\eta$ 为

$$
\eta = \eta_{sb} \eta_{cos} \eta_{at} \eta_{ref} \eta_{trunc}
$$

其中

阴影遮挡效率 $\eta_{sb} = 1 - 阴影遮挡损失$

余弦效率 $\eta_{cos} = 1 - 余弦损失$

大气投射率 $\eta_{at} = 0.99321 - 0.0001176 d_{HR} + 1.97 \times 10^{-8} \times d_{HR}^2$

集热器截断效率 $\eta_{trunc} = \frac{集热器接收能量}{镜面全反射能量 − 阴影遮挡损失能量}$

镜面反射效率 $\eta_{ref}$ 可取常数，例如 0.92，

其中 $d_{HR}$ 为镜面中心与集热器中心的距离（单位：m）。

## A题分析

### 第一问

约束条件

请推倒给出阴影遮挡损失和余弦损失的计算方法。

关注张平论文

> 2.2 阴影挡光效率理论模型

②后排定日镜接收的太阳光被前方定日镜所阻挡被称为阴影损失。
③后排定日镜在反射太阳光时被前方定日镜阻挡 而未到达吸热器上被称为挡光损失。

请写一个python文件，读取一个xlsx文件，提取x轴坐标，y轴坐标，建立坐标轴，将每一个点绘制在坐标轴上。

---

关注兰州大学论文

余弦损失

$$
\eta_{\cos} = \cos \theta = \vec{i} \cdot \vec{n} \\
\vec{i} = [- \cos (\alpha_s) \sin (\gamma_s), - \cos (\alpha_s) \cos (\gamma_s), - \sin (\alpha_s)] \\
$$

修改，移除“-”号

$$
\eta_{\cos} = \cos \theta = \vec{i} \cdot \vec{n} \\
\vec{i} = [\cos (\alpha_s) \sin (\gamma_s), \cos (\alpha_s) \cos (\gamma_s), \sin (\alpha_s)] \\
$$

其中，

$$
\vec{n} = \frac{\vec{i} + \vec{AO}}{|\vec{i} + \vec{AO}|}
$$

P15 + 20

判断step9

```python
AO = np.array([heli_x - tower_x, heli_y - tower_y, heli_z - tower_z])
```

AO 为一个已知向量，请再写一个函数，计算 $\vec{n}$, $\vec{i}$, $\eta_{\cos}$。

$$
\eta_{\cos} = \cos \theta = \vec{i} \cdot \vec{n} \\
\vec{i} = [- \cos (\alpha_s) \sin (\gamma_s), - \cos (\alpha_s) \cos (\gamma_s), - \sin (\alpha_s)] \\
$$

其中，

$$
\vec{n} = \frac{\vec{i} + \vec{AO}}{|\vec{i} + \vec{AO}|}
$$

阴影遮挡损失

蒙特卡罗光线追踪法

prompt:
假设 n 个点，已知每个点的坐标，求第k个点到其他点的距离。应该使用什么算法？

假设我想知道第k个点到其他点的距离小于l的所有点的坐标，应该使用什么算法？

此处距离l先随便设置一个，例如20m。

现在所有点的坐标在dataframe中。请编写一个函数，输入为一个坐标，输入k，l，输出这个点到其他点的距离小于l的所有点的坐标。

---

假设已知三维空间中的一个矩形平面的四个顶点ABCD、矩形中心点O的坐标、矩形的长宽，请将矩形划分为n个小矩矩形，
请写一段python 代码，求每个小矩阵的中心点的坐标和长宽，输出一个np.array，包含中心点的坐标和长宽？

---

由定日镜的镜面法向量进而可计算得出定日镜的俯仰角 $\theta_z$ 和方位角 $\theta_s$:

$$
\tan (\theta_z) = \frac{\sin (\alpha_s) \cdot m + h}{\sqrt{x_{o, A}^2 + y_{o, A}^2 + m^2 \cdot \cos ^2(\alpha_s) - 2 \cos (\alpha_s) \cdot m \cdot (x_{o, A} \cdot \sin (\gamma_s) - y_{o, A} \cdot \cos (\alpha_s))} } \\
\sin (\theta_s) = \frac{x_{o, A} - \cos (\alpha_s) \cdot \sin (\gamma_s) \cdot m}{\sqrt{x_{o, A}^2 + y_{o, A}^2 + m^2 \cdot \cos ^2(\alpha_s) - 2 \cos (\alpha_s) \cdot m \cdot (x_{o, A} \cdot \sin (\gamma_s) - y_{o, A} \cdot \cos (\alpha_s))} } \\
m = \sqrt{x_{o, A}^2 + y_{o, A}^2 + h_0^2} \\
$$

已知 $\alpha_s, \gamma_s, x, y, h$ 基于上面的公式，请写一段python程序，求 $\theta_z, \theta_s$。

---

写一个函数，解析列表 heli_para, heli_para 形如 [114, 114, 4, 6, 6] 中的值分别为 w, h, x0, y0, h0

---

写一个python函数，输入两个三维空间坐标，由此计算出一条直线的方程，再输入三个三维空间坐标，计算平面方程，判断交点是否在平面内。

---

little_mirror_centers 是由多个nparray组成的nparray，请在`if t >= 0 and b1 >= 0: continue` 后续写一段代码，判断成立时，将little_mirror_center这个值添加到little_mirror_blocked，随后从little_mirror_centers中删除

```python
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

            ........

            s1e1 = np.dot(s1, e1)
            t = np.dot(s2, e2) / s1e1
            b1 = np.dot(s1, s) / s1e1
            b2 = np.dot(s2, reflect_vector)

            if t >= 0 and b1 >= 0:
                continue
```

---

已知长方体的所有顶点，一条射线的起点和方向向量，求射线与长方体是否有交点，可以使用什么算法？

已知平面法向量 n 和入射光线反方向的单位向量 i，请写一段python代码，计算反射光线的单位向量 r。

有一个直径为7m, 高度为8m 的圆柱体，请写一段代码，将其放在box_vertices中，求出这个box_vertices的所有顶点。

### 第二问

请根据以下文本，编写一个python函数，输入定日镜场的半径，输出镜场的定日镜的位置坐标，将输出坐标等封装为dataframe并写入到p2_answer.csv, 再编写一个函数，将dataframe绘制为图像。

Campo规则布置镜场首先从第一个布置区域的第一行开始。第一行的半径 $R_1$ 由第一行的定日镜数量Nhel1 计算得出。布置过程 开始时，首环第一个定日镜放置在镜场的正北方向，其余定日镜以不发生机械碰撞为原 则进行周向均匀布置。第二行定日镜数目与第一行相同，与第一行定日镜交错放置且使 相邻行定日镜的特征圆相切。$R_1$ 为镜场首行定日镜的半径， $\Delta \alpha_{z1}$ 为首行定日镜之间的方位夹角。

$$
R_1 = (DM \cdot N_{hel1} + 2 \cdot DM) / 2 \cdot \sin (\Delta \alpha_{z1} / 2) \\
\Delta a z_1 = 2 \arcsin[DM / (2R_1)]
$$

DM为定日镜的特征圆直径，定日镜的长为LW、宽为LH、安全距离desp、DH为定日镜的对角线长度

$$
DM = DH + desp
$$

在最密集的排列方式下，相邻行半径的最小增量为 $\Delta R_{min}$

$$
\Delta R_{min} = DM \cdot \cos 30 - h \\
h = R_1 - \sqrt{R_1^2 - (DM)^2 / 4}
$$

各区域内每 行的定日镜数量 $Nheln$ 与各区域首行的行半径 $R_1^n$， n 为被布置区域的序号

$$
Nhel_n = Nhel_1 \cdot 2^{n - 1} \\
R_n = R_1 \cdot 2^{n-1}
$$

第 n 区域内所能允许布置的最大行数 $Nrows_n$

$$
Nrows_n = \frac{2^{n-1} \cdot R_1}{\Delta R_{min}}
$$

计算预期镜场中的每个区域的行数、每个区域行上的镜子数、每一行的行半径即可得到由Campo布置方法生成的密集型定日镜场。

---

粒子群算法

一个粒子对应一个状态，一个状态对应一个解，一个解对应一个函数值。

要函数值最优，目标函数见pdf。

prompt: 现在要使用 AGSA 算法编写一个求解优化问题的函数，下面，我将根据模块要求提问，请分段给出代码。

> 通过Campo布置规则对初始镜场进行编码。假设种群数量为N，初始种群将是一个N行R列的矩阵。其中，矩阵的行数N代表种群中粒子的个数，即每一行表示一种 镜场布置方案。矩阵的列数R代表粒子的维度，即镜场中定日镜的行数，其数值对应镜场中每一行的行半径Rn 。为了增加种群的多样性，相邻行之间的间距变化在编码时设置为密集排布方式下所能允许的最小增量的1~2倍，即 $\Delta R = \Delta R_{min} +rand * \Delta R_{min} $。

根据这个要求，编写一个函数，用于初始种群。

现在有一个函数fondational_kabuilding(r,R,d,k)，输入为圆的半径,内圆间隔，点的间隔，圆的个数，输出为点的坐标。假设圆的半径,内圆间隔，点的间隔为固定值，圆的个数为可变值，请编写一个函数，输入为包含坐标列表，输出为适应度值。

适应度值的计算方法如下：对于坐标列表中的每一个坐标(x, y, z)，使用p1.Station(x, y, z)创建一个定日镜，使用Station.eta_cos()计算余弦损失，使用Station.eta_sb()计算阴影损失，使用Station.eta_at()计算大气投射率，使用Station.eta_ref()计算镜面反射效率，使用Station.eta_trunc()计算集热器截断效率，使用Station.eta()计算光学效率，使用Station.E_field()计算输出热功率，使用Station.E_field_per_area()计算单位面积输出热功率，将所有定日镜的单位面积输出热功率相加，得到适应度值。

---

约束条件

* 最小间隔
* 功率 大于 60MW
* 镜 高度 [2, 6], 长宽 [2, 8]
* 临近日镜底座中心距离大于镜面宽度+5

目标函数

max 单位面积镜面年平均输出热功率 $\overline{W}$

$$
\overline{W} = \frac{E_{field}}{n \cdot (ab)} \\
$$

其中，$E_{field}$ 为定日镜场的输出热功率，$n$ 为定日镜总数，$a$ 为定日镜的长度，$b$ 为定日镜的宽度。

为了简化计算，先关注定日镜的坐标，计算余弦损失，剩余参数使用定值代替。

---

现在有一个函数 plot_coordinate(r,R,d,circle_amount,ro)，r,R,d,circle_amount,ro 分别为圆的半径,内圆间隔，点的间隔，圆的个数，输出为点的坐标。假设圆的半径,内圆间隔，点的间隔为固定值，circle_amount、ro为可变值，通过fitness(plot_coordinate(r,R,d,circle_amount,ro)) 可以得到适应度值，请使用自适应引力搜索算法（Adaptive Gravitational Search Algorithm，AGSA），编写一个段程序，求解适应度值最大的circle_amount 和 ro。circle_amount 可能的范围在[1, 100]之间，ro可能的范围在[1, 1.5]之间。

请注意 fondational_kabuilding 和 fitness 函数不需要编写，直接调用即可。

现在有一个函数 plot_coordinate(r,w,h,z,circle_amount,ro)，假设r为固定值，w,h,z,circle_amount,ro为可变值，通过fitness(plot_coordinate(r,w,h,z,circle_amount,ro)) 可以得到适应度值，请使用自适应引力搜索算法（Adaptive Gravitational Search Algorithm，AGSA），编写一个段程序，求解适应度值最大的w,h,z,circle_amount,ro

定义相关参数的取值范围和精度:
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

请注意 plot_coordinate 和 calc_fitness 函数不需要编写，直接调用即可。

---

修改代码，将best_points中所有满足条件的点，添加到 new_point_tmp 中，再计算fitness，将new_points_tmp中fitness值最大的点添加到new_points中。

```python
def plot_coordinate_real(best_points):
    tower_para = (tower_x, 0, 84)
    fitness_values_max = 0
    for tower_x in range(-350, 350, 10):
        tower_para = (tower_x, 0, 84)
        new_points = []
        for points in best_points:
            new_points_tmp = []
            x = points[0]
            y = points[1]
            z = points[2]
            point = (x, y, z)
            distance = (x-tower_para[0])**2 + (y-tower_para[1])**2
            if distance < 350**2:
                new_points_tmp.append(point)
            fitness_values_new = calc_fitness(new_points_tmp)
            print("fitness_values_new is:", fitness_values_new)
        
    return tower_para, new_points
```

### 第三问

kmeans是什么

请写一段python代码

## 2023 A 赛后总结

代码问题太多，编程能力需要加强
