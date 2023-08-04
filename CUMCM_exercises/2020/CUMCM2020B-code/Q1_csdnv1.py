# 请解释下面的代码，代码将分两段给出，在输入“已完成”后开始解释代码并添加注释，请分段回答。下面是第一段：

import numpy as np

qua = np.array((0, 1, 2, 3))  # 4种类型的特殊点：0-> 起点； 1-> 村庄；2 ->矿山; 3-> 终点
dist = np.array(((0, 6, 8, 3),
                 (6, 0, 2, 3),
                 (8, 2, 0, 5),
                 (3, 3, 5, 0)))
f = np.array(((0, 1, 1, 1),  # 判断4中特殊点之间的关系
              (0, 0, 1, 1),
              (0, 1, 0, 1),
              (0, 0, 0, 0)))
wea = np.array((2, 2, 1, 3, 1,  # 30天内的天气
                2, 3, 1, 2, 2,
                3, 2, 1, 2, 2,
                2, 3, 3, 2, 2,
                1, 1, 2, 1, 3,
                2, 1, 1, 2, 2))
mx, my = 3, 2  # 水和事物的重量
cx, cy = 5, 10  # 水和事物的价格
sx = np.array((0, 5, 8, 10))  # 第i种天气的水消耗量
sy = np.array((0, 7, 6, 10))  # 第i种天气的食物消耗量

n = 4  # 特殊点个数

maxm = 1200  # 背包容量
coins = 10000  # 初始资金
base = 1000  # 基础收益
date = 30  # 最晚期限
costx = np.zeros((32, 4, 4))  # 第k天从i到j的水消耗量
costy = np.zeros((32, 4, 4))  # 第k天从i到j的事食物消耗量
days = np.zeros((32, 4, 4), dtype=int)  # 第k天从i到j需要多长时间
ans = 0  # 最后的资金
act = np.empty(32)  # 每天的行为：2-> 挖矿；1-> 在矿山休息；0-> 村庄购买食物
rec = np.empty(32)  # 记录每天在哪里

# ansx、ansact记录最优路径上的信息
ansx = np.empty(32)
ansact = np.empty(32)
ansg, ansh = 0, 0  # 记录最优的初始水和食物


def dfs(day: int, now: int, nm: int, c: int, x: int, y: int, type: int) -> None:
    act[day] = type
    rec[day] = now
    global ans, ansg, ansh

    if qua[now] == 3:
        if ans <= c+x*cx+y*cy:
            ansg = g
            ansh = h
            ans = c+x*cx+y*cy
            for i in range(date+1):
                ansx[i] = rec[i]
                ansact[i] = act[i]
        act[day] = -1
        rec[day] = -1
        return

    if day >= date:
        act[day] = -1
        rec[day] = -1
        return

    if qua[now] == 1:
        nm = maxm - mx*x - my*y

    for i in range(n):
        if f[qua[now]][qua[i]]:
            tx = costx[day][now][i]
            ty = costy[day][now][i]
            ucost = c
            um = nm
            if x >= tx:
                ux = x - tx
            else:
                ux = 0
                ucost -= 2*(tx-x)*cx
                um -= (tx - x)*mx
            if y >= ty:
                uy = y - ty
            else:
                uy = 0
                ucost -= 2*(ty - y)*cy
                um -= (ty - y)*my

            if ucost < 0 or um < 0:
                continue
            dfs(day+days[day][now][i], i, um, ucost, ux, uy, 0)

    if qua[now] == 2:
        attday = day
        tx = sx[wea[attday]]
        ty = sy[wea[attday]]
        attday += 1
        if x > tx:
            x -= tx
            tx = 0
        else:
            tx = tx - x
            x = 0
        if y >= ty:
            y -= ty
            ty = 0
        else:
            ty = ty - y
            y = 0
        nm -= tx*mx + ty*my
        c -= 2*tx*cx + 2*ty*cy
        if nm >= 0 and c >= 0:
            dfs(attday, now, nm, c, x, y, 1)

        attday = day
        tx = sx[wea[attday]]*2
        ty = sy[wea[attday]]*2
        attday += 1
        if x >= tx:
            x -= tx
            tx = 0
        else:
            tx = tx - x
            x = 0
        if y >= ty:
            y -= ty
            ty = 0
        else:
            ty = ty - y
            y = 0
        nm -= tx*mx + ty*my
        c -= 2*tx*cx + 2*ty*cy
        c += base
        if nm >= 0 and c >= 0:
            dfs(attday, now, nm, c, x, y, 2)

    rec[day] = -1
    act[day] = -1


if __name__ == '__main__':
    for d in range(date+1):
        rec[d] = -1
        act[d] = -1

    for d in range(date):
        for i in range(n):
            for j in range(n):
                if f[qua[i]][qua[j]]:
                    now, count, sumx, sumy = 0, 0, 0, 0
                    while count < dist[i][j]:
                        if wea[now+d] != 3:
                            count += 1
                            sumx += 2*sx[wea[now+d]]
                            sumy += 2*sy[wea[now+d]]
                        else:
                            sumx += sx[wea[now+d]]
                            sumy += sy[wea[now+d]]

                        now += 1
                        if now + d >= date:
                            break
                    if count < dist[i][j]:
                        sumx = sumy = 20000
                        now = 30
                    costx[d][i][j] = sumx
                    costy[d][i][j] = sumy
                    days[d][i][j] = now
    print(type(days[0, 0, 0]))

    dic = {}
    for i in range(maxm+1):
        g = i // mx
        h = (maxm-i)//my
        # print(g, h)
        dic.setdefault((g, h), 0)
        if not dic[(g, h)]:
            print((g, h))
            dfs(0, 0, 0, coins-g*cx-h*cy, g, h, -1)
        dic[(g, h)] = 1

    print(ans)
