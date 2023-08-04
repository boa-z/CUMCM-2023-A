# 导入numpy库，用于处理数组
import numpy as np

# 定义四种类型的特殊点：0-> 起点； 1-> 村庄；2 ->矿山; 3-> 终点
qua = np.array((0, 1, 2, 3))  
# 定义四个特殊点之间的距离，用二维数组表示
dist = np.array(((0, 6, 8, 3),
                 (6, 0, 2, 3),
                 (8, 2, 0, 5),
                 (3, 3, 5, 0)))
# 定义四个特殊点之间的关系，用二维数组表示，1表示可以从一个点到另一个点，0表示不可以
f = np.array(((0, 1, 1, 1),  
              (0, 0, 1, 1),
              (0, 1, 0, 1),
              (0, 0, 0, 0)))
# 定义30天内的天气，用一维数组表示，每个元素代表一种天气，1-> 晴朗；2 ->高温；3 ->沙暴
wea = np.array((2, 2, 1, 3, 1,
                2, 3, 1, 2, 2,
                3, 2, 1, 2, 2,
                2, 3, 3, 2, 2,
                1, 1, 2, 1, 3,
                2, 1, 1, 2, 2))
# 定义水和食物的重量
mx = my = np.array((3.5))  
# 定义水和食物的价格
cx = cy = np.array((5))  
# 定义第i种天气的水消耗量
sx = np.array((0.5))  
# 定义第i种天气的食物消耗量
sy = np.array((0.5))  

n = len(qua) # 特殊点个数

maxm = np.array((120)) # 背包容量
coins = np.array((10000)) # 初始资金
base = np.array((1000)) # 基础收益
date = len(wea) # 最晚期限
costx = np.zeros((date+1,n,n)) # 第k天从i到j的水消耗量
costy = np.zeros((date+1,n,n)) # 第k天从i到j的食物消耗量
days = np.zeros((date+1,n,n),dtype=int) # 第k天从i到j需要多长时间
ans = np.array((0)) # 最后的资金
act = np.empty(date+1) # 每天的行为：2-> 挖矿；1-> 在矿山休息；0-> 村庄购买食物
rec = np.empty(date+1) # 记录每天在哪里

# ansx、ansact记录最优路径上的信息
ansx = np.empty(date+1)
ansact = np.empty(date+1)
ansg = ansh = np.array((0)) # 记录最优的初始水和食物

# 定义一个深度优先搜索函数，用于遍历所有可能的路径，并更新最优解
def dfs(day: int , now: int , nm: int , c: int , x: int , y: int , type: int ) -> None:
    act[day] = type # 记录当前行为
    rec[day] = now # 记录当前位置
    global ans , ansg , ansh

    if qua[now] == qua[-1]: # 如果到达终点
        if ans <= c+x*cx+y*cy: # 如果当前资金大于等于之前的最优解
            ansg = x # 更新最优初始水量
            ansh = y # 更新最优初始食物量
            ans = c+x*cx+y*cy # 更新最优资金
            for i in range(date+1): # 记录最优路径
                ansx[i] = rec[i]
                ansact[i] = act[i]
        act[day] = -1 # 恢复当前行为
        rec[day] = -1 # 恢复当前位置
        return # 返回上一层

    if day >= date: # 如果超过期限
        act[day] = -1 # 恢复当前行为
        rec[day] = -1 # 恢复当前位置
        return # 返回上一层

    if qua[now] == qua[1]: # 如果在村庄
        nm = maxm - mx*x - my*y # 更新背包剩余空间

    for i in range(n): # 遍历所有可能的下一个位置
        if f[qua[now]][qua[i]]: # 如果可以从当前位置到达下一个位置
            tx = costx[day][now][i] # 计算从当前位置到下一个位置的水消耗量
            ty = costy[day][now][i] # 计算从当前位置到下一个位置的食物消耗量
            ucost = c # 记录当前资金
            um = nm # 记录背包剩余空间
            if x >= tx: # 如果水量足够
                ux = x - tx # 更新水量
            else: # 如果水量不够
                ux = 0 # 水量归零
                ucost -= 2*(tx-x)*cx # 资金扣除两倍的水价差
                um -= (tx - x)*mx # 背包空间扣除水重量差
            if y >= ty: # 如果食物足够
                uy = y - ty # 更新食物量
            else: # 如果食物不够
                uy = 0 # 食物量归零
                ucost -= 2*(ty - y)*cy # 资金扣除两倍的食物价差
                um -= (ty - y)*my # 背包空间扣除食物重量差

            if ucost < 0 or um < 0: # 如果资金或背包空间不足，跳过这个位置
                continue 
            dfs(day+days[day][now][i], i, um, ucost, ux, uy, 0) # 递归搜索下一个位置

    if qua[now] == qua[2]: # 如果在矿山
        attday = day # 记录当前天数
        tx = sx[wea[attday]]*2 # 计算挖矿时的水消耗量，是天气因素的两倍
        ty = sy[wea[attday]]*2 # 计算挖矿时的食物消耗量，是天气因素的两倍
        attday += 1 # 天数加一
        if x >= tx: # 如果水量足够
            x -= tx # 更新水量
            tx = 0 # 水消耗量归零
        else: # 如果水量不够
            tx = tx - x # 水消耗量减去水量
            x = 0 # 水量归零
        if y >= ty: # 如果食物足够
            y -= ty # 更新食物量
            ty = 0 # 食物消耗量归零
        else: # 如果食物不够
            ty = ty - y # 食物消耗量减去食物量
            y = 0 # 食物量归零
        nm -= tx*mx + ty*my # 背包空间减去水和食物的重量差
        c -= 2*tx*cx + 2*ty*cy # 资金减去两倍的水和食物的价差
        c += base # 资金加上基础收益
        if nm >= 0 and c >= 0: # 如果背包空间和资金都足够
            dfs(attday, now, nm, c, x, y, 2) # 递归搜索下一天，位置不变，背包空间、资金、水、食物都更新，行为为挖矿

    rec[day] = -1 # 恢复当前位置
    act[day] = -1 # 恢复当前行为



if __name__ == '__main__':
    for d in range(date+1): # 初始化每天的行为和位置为-1，表示未知
        rec[d] = -1
        act[d] = -1

    for d in range(date): # 预处理每天从一个位置到另一个位置的消耗和时间，用三维数组存储
        for i in range(n):
            for j in range(n):
                if f[qua[i]][qua[j]]: # 如果可以从一个位置到另一个位置
                    now , count , sumx , sumy = 0 , 0 , 0 , 0 
                    while count < dist[i][j]: # 当还没走完距离时，循环计算消耗和时间
                        if wea[now+d] != wea[-1]: # 如果不是沙暴天
                            count += 1 # 距离减一
                            sumx += 2*sx[wea[now+d]] # 水消耗量加上两倍的天气因素
                            sumy += 2*sy[wea[now+d]] # 食物消耗量加上两倍的天气因素
                        else: # 如果是沙暴天
                            sumx += sx[wea[now+d]] # 水消耗量加上一倍的天气因素
                            sumy += sy[wea[now+d]] # 食物消耗量加上一倍的天气因素

                        now += 1 # 时间加一
                        if now + d >= date: # 如果超过期限，跳出循环
                            break
                    if count < dist[i][j]: # 如果没有走完距离，说明不可能到达，设置一个很大的消耗量和时间
                        sumx = sumy = 20000 
                        now = 30 
                    costx[d][i][j] = sumx # 记录从i到j的水消耗量
                    costy[d][i][j] = sumy # 记录从i到j的食物消耗量
                    days[d][i][j] = now # 记录从i到j需要的时间
    print(type(days[0, 0, 0]))

dic = {} # 定义一个字典，用于记录已经搜索过的初始水和食物组合，避免重复搜索
for i in range(maxm+1): # 遍历所有可能的初始水和食物组合
    g = i // mx # 计算初始水量
    h = (maxm-i)//my # 计算初始食物量
    dic.setdefault((g, h), 0) # 如果字典中没有这个组合，设置为0，表示未搜索过
    if not dic[(g, h)]: # 如果未搜索过这个组合
        dfs(0, 0, 0, coins-g*cx-h*cy, g, h, -1) # 调用深度优先搜索函数，从第0天，起点，空背包，初始资金减去水和食物的花费，初始水量，初始食物量，未知行为开始搜索
    dic[(g, h)] = 1 # 标记这个组合已经搜索过

print(ans) # 输出最优资金
