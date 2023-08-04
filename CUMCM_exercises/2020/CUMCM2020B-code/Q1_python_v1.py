import numpy as np
import pandas as pd
import pulp as pl

p = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
])

T = 30  # 游戏总天数
N = len(p)  # 总的区域个数
M = 999999  # 相对大的数
G = 1000  # 挖矿的基础收益
L = 1200  # 负重上限
S_0 = 10000  # 初始资金
W_1 = 3  # 水的单位重量
W_2 = 2  # 食物的单位重量
P_1 = 5  # 水的单位价格
P_2 = 10  # 食物的单位价格

# 每天水和食物的基础消耗量（箱）
A = [5, 8, 10]  # 水
B = [7, 6, 10]  # 食物

# 天气状况：1表示晴朗、2表示高温、3表示沙暴
weather = [2, 2, 1, 3, 1, 2, 3, 1, 2, 2, 3, 2, 1, 2,
           2, 2, 3, 3, 2, 2, 1, 1, 2, 1, 3, 2, 1, 1, 2, 2]

# 区域类型：1表示起点、2表示矿山、3表示村庄、4表示终点
area_type = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
             0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
print("地图p的形状为：", p.shape)
print("总的区域个数为：", N)
print("游戏总天数为：", T)
print("初始资金为：", S_0)


##2##

# 创建一个优化问题对象，名称为"Rainforest Adventure"，目标类型为最大化
model = pl.LpProblem("Rainforest Adventure", pl.LpMaximize)
# 创建决策变量x_{ti}，表示第t天玩家是否位于区域i
x = pl.LpVariable.dicts("x", [(t, i) for t in range(T+1)
                        for i in range(N)], 0, 1, pl.LpInteger)

# 创建决策变量y_t，表示第t天玩家拥有的水数量
y = pl.LpVariable.dicts("y", [t for t in range(T+1)], 0, None, pl.LpInteger)

# 创建决策变量z_t，表示第t天玩家拥有的食物数量
z = pl.LpVariable.dicts("z", [t for t in range(T+1)], 0, None, pl.LpInteger)

# 创建决策变量S_t，表示第t天玩家拥有的资金
S = pl.LpVariable.dicts("S", [t for t in range(T+1)], 0, None, pl.LpInteger)
S[-1] = S_0

# 创建决策变量r_{ti}，表示第t天玩家是否在区域i挖矿
r = pl.LpVariable.dicts("r", [(t, i) for t in range(T+1)
                        for i in range(N)], 0, 1, pl.LpInteger)

# 创建决策变量b_{ti}，表示第t天玩家购买的第i种资源数量
b = pl.LpVariable.dicts("b", [(t, i) for t in range(T+1)
                        for i in range(2)], 0, None, pl.LpInteger)
# 定义目标函数：最大化玩家的剩余资金
model += S[T] + 0.5 * (P_1 * y[T] + P_2 * z[T]), "Objective"
# 定义初始条件：玩家在第0天位于起点，并用初始资金购买水和食物
model += x[0, 0] == 1, "Initial position"
model += S[0] == S_0 - P_1 * b[0, 0] - P_2 * b[0, 1], "Initial fund"
model += y[0] == b[0, 0], "Initial water"
model += z[0] == b[0, 1], "Initial food"

# 定义终止条件：玩家在第T天或之前到达终点，并退回剩余的水和食物
model += x[T, N-1] == 1, "Final position"
model += pl.lpSum(x[t, N-1] for t in range(T)) <= 1, "Only one arrival"

# 定义行走条件：玩家每天只能从一个区域到达相邻的另一个区域或原地停留，并且沙暴日必须原地停留
for t in range(T):
    model += pl.lpSum(x[t, i]
                      for i in range(N)) == 1, f"Only one position at day {t}"
    for i in range(N):
        for j in range(N):
            model += x[t, i] + x[t+1, j] <= p[i, j] + \
                1, f"Adjacent movement from {i} to {j} at day {t}"
        model += x[t, i] + \
            x[t+1, i] >= 3, f"Stay at {i} at day {t} if sandstorm"

# 定义资源条件：玩家每天拥有的水和食物质量之和不能超过负重上限，并且不能耗尽
for t in range(T+1):
    model += y[t] + z[t] * W_2 <= L, f"Weight limit at day {t}"
    model += y[t] >= 0, f"Water nonnegative at day {t}"
    model += z[t] >= 0, f"Food nonnegative at day {t}"

# 定义资金条件：玩家每天拥有的资金不能为负数，并且不能多次在起点购买资源
for t in range(T+1):
    model += S[t] >= 0, f"Fund nonnegative at day {t}"
model += pl.lpSum(b[t, 0] + b[t, 1] for t in range(1, T+1)
                  ) <= M * (1 - x[0, 0]), "No more purchase at start point"

# 定义挖矿条件：玩家只能在矿山区域挖矿，并且到达矿山当天不能挖矿
for t in range(1, T+1):
    for i in range(N):
        # model += r[t,
        #            i] <= area_type[i] == 2, f"Only mine at mine area {i} at day {t}"
        model += r[t, i] + \
            x[t-1, i] <= 1, f"No mine at arrival at area {i} at day {t}"

# 定义购买条件：玩家只能在起点或村庄区域购买资源，并且价格与基准价格有关
for t in range(T+1):
    for i in range(2):
        model += b[t,
                   i] >= 0, f"Purchase nonnegative for resource {i} at day {t}"
        model += b[t, i] <= M * (area_type[i] ==
                                 3), f"Only purchase at village area for resource {i} at day {t}"
    model += S[t] == S[t-1] + pl.lpSum(G * r[t, i] * (area_type[i] == 2) for i in range(N)) - pl.lpSum(
        2 * 5 * b[t, i] * (area_type[i] == 3) for i in range(2)), f"Fund change at day {t}"

##3##

# 选择CBC求解器，它是PuLP自带的一个开源求解器
solver = pl.PULP_CBC_CMD()
# 调用求解器来求解优化问题
status = model.solve(solver)

# 打印求解状态
print("Status:", pl.LpStatus[status])

# 如果求解成功，打印最优目标值和最优决策变量值
if status == pl.LpStatusOptimal:
    print("Objective:", model.objective.value())
    for v in model.variables():
        print(v.name, "=", v.varValue)

# 如果求解成功，打印最优策略和相关信息
if status == pl.LpStatusOptimal:
    print("恭喜你，你已经成功完成了雨林探险游戏！")
    print("你的最终剩余资金为：", model.objective.value(), "元")
    print("你的最终剩余水和食物数量为：", y[T].varValue, "箱和", z[T].varValue, "箱")
    print("以下是你每天的决策：")
    for t in range(T+1):
        # 找出玩家所在的区域
        for i in range(N):
            if x[t, i].varValue == 1:
                area = i
                break
        # 打印玩家所在的区域
        print(f"第{t}天，你位于区域{area}，")
        # 如果是起点或村庄，打印玩家购买的资源数量
        if area_type[area] in [1, 3]:
            print(f"你购买了{b[t,0].varValue}箱水和{b[t,1].varValue}箱食物，")
        # 如果是矿山，打印玩家是否挖矿
        if area_type[area] == 2:
            if r[t, area].varValue == 1:
                print("你选择了挖矿，")
            else:
                print("你没有选择挖矿，")
        # 打印玩家消耗的资源数量
        print(f"你消耗了{A[t]}箱水和{B[t]}箱食物。")
