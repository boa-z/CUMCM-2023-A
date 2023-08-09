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


def buy(cur_day, cur_point, cur_food, cur_water):
    max_food = carry_limit / base_food_weight


init_food = 0
init_water = 0

# log 记录
log_list = {}


def dp_main():
    dp = np.full((30, 30, 600, 600), -1 * np.inf)

    global init_food
    global init_water
    cur_day = 0

    # 起点购买物资
    for init_food in range(0, 600):
        init_water = (carry_limit - base_food_weight * init_food) // base_water_weight
        if check(init_food, init_water):
            dp[0][1][init_food][init_water] = init_money - base_consume_food[get_weather(cur_day)] * init_food - \
                                              base_consume_water[
                                                  get_weather(cur_day)] * init_water

    cur_money = 10000
    for cur_day in range(0, 29):
        for cur_point in range(1, 27):

            # 在村庄物资更新
            # if Map[cur_point].state == "c":
            #     for buy_food in range(0, 600):
            #         for buy_water in range(0, 400):
            #             for cur_food in range(0, 600):
            #                 for cur_water in range(0, 400):
            #                     if check(cur_food, cur_water):
            #                         dp[cur_day, cur_point, buy_food + cur_food, buy_water + cur_water] = max(
            #                             dp[cur_day, cur_point, buy_food + cur_food, buy_water + cur_water], dp[
            #                                                                                             cur_day, cur_point, cur_food, cur_water] -
            #                                                                                         base_consume_food[
            #                                                                                             get_weather(
            #                                                                                                 cur_day)] * cur_food -
            #                                                                                         base_consume_water[
            #                                                                                             get_weather(
            #                                                                                                 cur_day)] * cur_water)

            # 挖矿
            # if Map[cur_point].state == "k":
            #     for cur_food in range(0, 600):
            #         for cur_water in range(0, 400):
            #             dp[cur_day + 1, cur_point, cur_food - 3 * base_consume_water[
            #                 get_weather(cur_day)], cur_water - 3 *
            #                base_consume_food[
            #                    get_weather(cur_day)]] = max(dp[cur_day + 1, cur_point, cur_food - 3 *
            #                                                    base_consume_water[get_weather(cur_day)], cur_water - 3 *
            #                                                    base_consume_food[
            #                                                        get_weather(cur_day)]],
            #                                                 dp[cur_day, cur_point, cur_food, cur_water] + base_income)

            # 停留
            # for food in range(0, 600):
            #     for water in range(0, 400):
            #         dp[cur_day + 1, cur_point, food - base_consume_water[get_weather(cur_day)], water -
            #            base_consume_food[get_weather(cur_day)]] = max(
            #             dp[cur_day + 1, cur_point, food - base_consume_water[get_weather(cur_day)], water -
            #                base_consume_food[get_weather(cur_day)]], dp[cur_day, cur_point, food, water])

            # 移动
            for cur_point_new in Map[cur_point].neibor:
                cur_food = 0
                cur_water = 0

                for init_food in range(100, 600):
                    for init_water in range(100, 400):
                        if check(init_food, init_water):
                            cur_food = init_food
                            cur_water = init_water
                            cur_day_new = cur_day + 1
                            cur_food_new = cur_food - 2 * base_consume_food[get_weather(cur_day)]
                            cur_water_new = cur_water - 2 * base_consume_water[get_weather(cur_day)]
                    # dp[cur_day_new, cur_point_new, cur_food_new, cur_water_new] = max(
                    #     dp[cur_day_new, cur_point_new, cur_food_new, cur_water_new],
                    #     dp[cur_day, cur_point, cur_food, cur_water])
                    # print("kaka4", dp[0, 1, 240, 240])
                            dp[cur_day_new, cur_point_new, cur_food_new, cur_water_new] = dp[cur_day, cur_point, cur_food, cur_water]
                            # print(dp[cur_day_new, cur_point_new, cur_food_new, cur_water_new], cur_day_new, cur_point_new, cur_food_new, cur_water_new)
                # for k in range(210, 240):
                #     for l in range(210, 240):                    
                #         print("kaka5", dp[1, 2, k, l], k, l)
                    # dp[cur_day, cur_point, cur_food, cur_water] dp[0, 1]
                    # print(dp[cur_day_new, cur_point_new, cur_food_new, cur_water_new], cur_day, cur_point, cur_food, cur_water)

dp_main()
print(1)