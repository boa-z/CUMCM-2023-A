import numpy as np

weather = ["高温", "高温", "晴朗", "沙暴", "晴朗",
           "高温", "沙暴", "晴朗", "高温", "高温",
           "沙暴", "高温", "晴朗", "高温", "高温",
           "高温", "沙暴", "沙暴", "高温", "高温",
           "晴朗", "晴朗", "高温", "晴朗", "沙暴",
           "高温", "晴朗", "晴朗", "高温", "高温"
           ]

base_consume_water = [5, 8, 10]
base_consume_food = [7, 6, 10]
base_water_price = 5
base_water_weight = 3
base_food_price = 10
base_food_weight = 2
carry_limit = 1200
init_money = 10000
base_income = 1000

Map = []


class Node:
    def __init__(self, id, state, nodes):
        self.id = id
        self.state = state
        self.neibor = []
        for i in nodes:
            self.neibor.append(i)


# 这个函数用于建立地图
def build_map():
    fp = open("Map1.txt", 'r')  # 根据所求地图修改路径
    node_id = 1
    for i in fp:
        if i is not None:
            i = i.split()
            temp_state = i[0]
            temp_list = []
            for j in range(1, len(i)):
                temp_list.append(int(i[j]))
            temp_node = Node(node_id, temp_state, temp_list)
            Map.append(temp_node)
            node_id += 1


build_map()


def get_weather(i):
    if i == "高温":
        return 1
    if i == "晴朗":
        return 0
    else:
        return 2


def check(i, j):
    if 3 * i + 2 * j > 1200 or 5 * i + 10 * j > 10000:
        return False
    else:
        return True


# 功能算法
def Dijktra(start: int, mgraph: list) -> list:
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]
    dis = mgraph[start]

    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]: idx = i

        nopass.remove(idx)
        passed.append(idx)

        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]: dis[i] = dis[idx] + mgraph[idx][i]
    return dis

dp = np.full((30, 30, 600, 600), -1 * np.inf)

# log 记录
log_list = {}


def dp_main():
    global dp
    cur_day = 0
    food = 0

    # 起点购买物资
    for food in range(0, 600):
        water = (carry_limit - base_food_weight * food) // base_water_weight
        if check(food, water):
            dp[0, 1, food, water] = init_money - base_consume_food[get_weather(cur_day)] * food - base_consume_water[
                get_weather(cur_day)] * water

    for cur_day in range(0, 30):
        for cur_point in range(1, 27):
            # 在村庄物资更新
            if Map[cur_point] == "c":
                for food in range(0, 600):
                    for water in range(0, 400):
                        print(dp[0, 1, food, water])
                        for cur_food in range(0, 6000):
                            for cur_water in range(0, 4000):
                                if check(cur_food, cur_water):
                                    dp[cur_day, cur_point, food + cur_food, water + cur_water] = max(
                                        dp[cur_day, cur_point, food + cur_food, water + cur_water], dp[
                                                                                                        cur_day, cur_point, food, water] -
                                                                                                    base_consume_food[
                                                                                                        get_weather(
                                                                                                            cur_day)] * cur_food -
                                                                                                    base_consume_water[
                                                                                                        get_weather(
                                                                                                            cur_day)] * cur_water)
            # 停留
            for food in range(0, 600):
                for water in range(0, 400):
                    dp[cur_day + 1, cur_point, food - base_consume_water[get_weather(cur_day)], water -
                       base_consume_food[
                           get_weather(cur_day)]] = max(
                        dp[cur_day + 1, cur_point, food - base_consume_water[get_weather(cur_day)], water -
                           base_consume_food[get_weather(cur_day)]], dp[cur_day, cur_point, food, water])
            # 挖矿
            if Map[cur_point].state == "k":
                for food in range(0, 600):
                    for water in range(0, 400):
                        dp[cur_day + 1, cur_point, food - 3 * base_consume_water[get_weather(cur_day)], water - 3 *
                           base_consume_food[
                               get_weather(cur_day)]] = max(dp[cur_day + 1, cur_point, food - 3 * base_consume_water[
                            get_weather(cur_day)], water - 3 * base_consume_food[get_weather(cur_day)]],
                                                            dp[cur_day, cur_point, food, water] + base_income)
                        test_1 = dp[cur_day + 1, cur_point, food - 3 * base_consume_water[get_weather(cur_day)], water - 3 *
                           base_consume_food[
                               get_weather(cur_day)]]
                        print(test_1)
            # 移动
            for cur_point_new in Map[cur_point].neibor:
                dp[cur_day + 1, cur_point_new, food - 2 * base_consume_water[get_weather(cur_day)], water - 2 *
                   base_consume_food[
                       get_weather(cur_day)]] = max(
                    dp[cur_day + 1, cur_point, food - 2 * base_consume_water[get_weather(cur_day)], water - 2 *
                       base_consume_food[get_weather(cur_day)]], dp[cur_day, cur_point, food, water])

dp_main()
