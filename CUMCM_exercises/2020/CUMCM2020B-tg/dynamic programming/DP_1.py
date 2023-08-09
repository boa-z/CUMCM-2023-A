from random import random, choice, shuffle


class Point:
    def __init__(self, index):
        self.name = index
        self.neighbours = []
        self.type = 0
        # 0 1 2 3 4: 普通 村庄 矿山 起点 终点

    def addNeighbour(self, pt):
        self.neighbours.append(pt)

    def setType(self, t):
        self.type = t


# 某一天，上一步可行解，结束后剩下的钱，当前地点，当前的（水，食物）元组
class Solution:
    def __init__(self, step, prev, money, pt, key):
        self.step = step
        # 妙啊，Solution的prev可以理解为一个链表，通过递归地遍历prev属性即倒着得到当前解一直到开始的所有局部可行解
        self.prev: 'Solution' = prev
        self.next = []
        self.money = money
        self.pt_index = pt
        self.key = key
        self.last_supply: 'tuple[int,int]' = key  # 上次补给完毕后剩余的水和食物？
        self.last_cash = money  # 上次补给完毕后所剩的钱


# 传入移动前的地点start、移动后的地点end（地点不一定变）、移动前的天气weather，与是否挖矿getMineral
# 类的其他属性是做出这个决策的代价与收益
class Decision:
    def __init__(self, start, end, weather, getMineral=False):
        self.start, self.end, self.weather = start, end, weather
        self.water = self.food = self.money = 0  # 消耗为正 赚得为负 水或食物单位为箱

        # 如果不动且没有到达终点
        if start.name == end.name and start.type != 4:
            if getMineral:
                # 执行挖矿的收益计算
                self.getMineral(weather)
            else:
                self.water = WATER_CONSUMPTION[weather]
                self.food = FOOD_CONSUMPTION[weather]

        # 如果动了
        if start.name != end.name:
            self.water = 2 * WATER_CONSUMPTION[weather]
            self.food = 2 * FOOD_CONSUMPTION[weather]

    def getMineral(self, weather):
        self.water = 3 * WATER_CONSUMPTION[weather]
        self.food = 3 * FOOD_CONSUMPTION[weather]
        self.money = -PROFIT


WEATHER = []
DAY_NUM = 0
MAX_BURDEN = 0
INIT_MONEY = 0
PROFIT = 0
WATER_WEIGHT = 0
WATER_PRICE = 0
FOOD_WEIGHT = 0
FOOD_PRICE = 0
WATER_CONSUMPTION = {}
FOOD_CONSUMPTION = {}
POINTS: 'list[Point]' = []
POINT_NUM = 0
DESTINATION = 0
MOVE_PLAN = []
MINING = 0


# 读取游戏地图，设置每个点的邻接点（包括自己）
def loadPoints(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            p = Point(int(line[0]))
            POINTS[int(line[0]) - 1] = p
            if line[2].replace('\n', '') != '':
                p.setType(int(line[2]))
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            if line[1] != '':
                for pt in line[1].split(' '):
                    POINTS[int(line[0]) - 1].addNeighbour(POINTS[int(pt) - 1])


def loadEnvir(problem_no):
    global WEATHER
    global DAY_NUM
    global MAX_BURDEN
    global INIT_MONEY
    global PROFIT
    global WATER_WEIGHT
    global WATER_PRICE
    global FOOD_WEIGHT
    global FOOD_PRICE
    global WATER_CONSUMPTION
    global FOOD_CONSUMPTION
    global POINT_NUM
    global DESTINATION
    global MOVE_PLAN
    if problem_no == 1 or problem_no == 2:
        WEATHER = '高温,高温,晴朗,沙暴,晴朗,高温,沙暴,晴朗,高温,高温,沙暴,高温,晴朗,高温,高温,高温,沙暴,沙暴,高温,高温,晴朗,晴朗,高温,晴朗,沙暴,高温,晴朗,晴朗,高温,高温'.split(
            ',')
        DAY_NUM = 30
        MAX_BURDEN = 1200
        INIT_MONEY = 10000
        PROFIT = 1000
        WATER_WEIGHT = 3
        WATER_PRICE = 5
        FOOD_WEIGHT = 2
        FOOD_PRICE = 10
        WATER_CONSUMPTION = {'晴朗': 5, '高温': 8, '沙暴': 10}
        FOOD_CONSUMPTION = {'晴朗': 7, '高温': 6, '沙暴': 10}
        POINT_NUM = 12 if problem_no == 1 else 17
        DESTINATION = 9 if problem_no == 1 else 12
        assert (len(WEATHER) == DAY_NUM)
        for i in range(POINT_NUM):
            POINTS.append([])
            MOVE_PLAN.append([])
            for j in range(POINT_NUM):
                MOVE_PLAN[i].append(0)
        loadPoints('data/problem{}_graph_simple.csv'.format(problem_no))


# 给出在某天某一点的全部策略的列表
def getDecision(point, day):
    decision_list = []
    # 不是沙暴就加上前往所有能去的格子的策略，包含了不动的策略
    if WEATHER[day] != '沙暴':
        for pt in point.neighbours:
            decision_list.append(Decision(point, pt, WEATHER[day]))
    decision_list.append(Decision(point, point, WEATHER[day]))
    if point.type == 2:
        # 如果是矿山就还要加上不动且挖矿策略
        decision_list.append(Decision(point, point, WEATHER[day], getMineral=True))
    return decision_list


# pt= point
def dp_main(init_water, init_food, start_day=0, init_pt=0):
    solution: 'list[list[dict[tuple[int,int],Solution]]]' = []
    for i in range(DAY_NUM + 1):
        solution.append([])
        for j in range(POINT_NUM):
            solution[i].append({})
    # solution是某天某点的可行方案的2维数组，第1维是天，第2维是地图的点的个数，第三维度是一个dict，键是水和食物的组合元组，值是一个Solution
    # 水和食物的组合太多，但可行解少，开个二维数组浪费内存，所以使用元组作为key，可以认为是一个稀疏的4维数组？
    cur_key = (init_water, init_food)

    # 按照开局的水和食物计算开局时还剩多少钱
    init_m = INIT_MONEY - init_water * WATER_PRICE - init_food * FOOD_PRICE

    # 开局第一天的可行解(init_food,init_water)设置为开局的条件
    solution[start_day][init_pt][cur_key] = Solution(start_day, None, init_m, init_pt, cur_key)

    # 从第一天开始找，找到结束，step是当前天数
    for step in range(start_day, DAY_NUM):
        real_date = step  # 顺序法
        pt_list = list(range(POINT_NUM))
        shuffle(pt_list)
        # 扫一遍所有的点
        for pt in pt_list:
            # records= 某一状态的全部solution
            records = solution[step][pt]
            # 如果当前状态到不了这个点，就直接换下一个点
            if len(records) == 0:
                continue

            # 扫一遍当前状态（天+地点）能进行的全部策略，包含挖矿、停留等等，具体看getDecision
            decisions = getDecision(POINTS[pt], real_date)
            shuffle(decisions)
            for d in decisions:
                _water, _food, _money = d.water, d.food, d.money  # 变化量
                # 扫一遍能到达当前天数、当前点的全部可行解
                for key in list(records.keys()):
                    # cur_solution: 当前正在查看的可行解
                    cur_solution = records[key]

                    # 执行当前策略d
                    water, food = key
                    new_water = water - _water
                    new_food = food - _food
                    new_money = cur_solution.money - _money

                    last_cash = cur_solution.last_cash
                    last_supply = cur_solution.last_supply

                    # 如果通过执行策略到达终点或者村庄
                    if d.end.type == 4 or d.end.type == 1:
                        # 上次补给的钱和物资的状态
                        last_water, last_food = last_supply
                        last_cash = new_money

                        # 此时欠多少食物和水就需要买多少
                        water_buy, food_buy = 0, 0
                        if new_water < 0:
                            water_buy = -new_water
                        if new_food < 0:
                            food_buy = -new_food
                        if water_buy > 0 or food_buy > 0:
                            # 如果欠水或者食物，但是上一次补给还是开局，则说明不可行，走不到村庄就寄了，进行剪枝
                            if last_supply == cur_key:
                                continue
                            canBuy = False

                            # 村庄的价格倍率
                            cost_k = 2

                            # 计算下购买物资要花多少钱
                            cost = water_buy * WATER_PRICE * cost_k + food_buy * FOOD_PRICE * cost_k
                            # 如果卖完还能装得下，且上一次到达村庄时，钱足够支付，则进行购买；否则说明中途寄了，进行剪枝
                            if (last_water + water_buy) * WATER_WEIGHT + \
                                    (last_food + food_buy) * FOOD_WEIGHT <= MAX_BURDEN and cost <= last_cash:
                                canBuy = True
                            if canBuy:
                                # 进行购买操作，计算现在的钱和物资
                                new_money = new_money - cost
                                new_water = new_water + water_buy
                                new_food = new_food + food_buy

                                # 记录本次补给后的钱和物资的状态
                                last_cash = new_money
                                last_supply = (new_water, new_food)
                            else:
                                continue
                        else:
                            # 不欠水或者食物，不花钱，但也算经过一次村庄
                            # 记录接下来的“上一次补给信息”为当天结束时的水和食物的信息
                            last_supply = (new_water, new_food)

                    # 到达一个位置，将水和食物作为键，创建一个到达当前状态的可行解
                    new_key = (new_water, new_food)
                    # Point类的name从1开始，POINTS的索引从0开始，所以要d.end.name - 1，下面同理
                    # 记录新解的上一步为当前解
                    new_solution = Solution(step + 1, cur_solution, new_money, d.end.name - 1, new_key)
                    new_solution.last_cash = last_cash
                    new_solution.last_supply = last_supply

                    # 加入当前状态可行解列表
                    cur_solution.next.append(new_solution)

                    # 当前生成的解总是到达下一天的可行解，所以扫一遍第二天能到达新位置的所有可行解中，相同(水，食物)状态的可行解，
                    # 比较剩下的钱，保留钱更多的可行解
                    if new_key in solution[step + 1][d.end.name - 1]:
                        if solution[step + 1][d.end.name - 1][new_key].money > new_solution.money:
                            continue
                    # 没找到比当前解的钱更多的，就记录当前解为当前4个状态的最优解
                    solution[step + 1][d.end.name - 1][new_key] = new_solution

    # dp完成，最优解就藏在下面final的dict中，这是因为游戏的最终状态就是最后一天，并到达终点
    final = solution[DAY_NUM][DESTINATION]
    final_pts = []
    final_pt = None
    max_finals = []
    max_final = 0
    # 扫一遍所有的最终状态，过滤掉到达重点但欠物资的
    for key in final:
        water, food = key
        if water < 0 or food < 0:
            continue
        # 将可行的候选最优解的最终money加入到max_finals数组里，Solution加入到final_pts里面
        max_finals.append(water * WATER_PRICE * 0.5 + food * FOOD_PRICE * 0.5 + final[key].money)
        final_pts.append(final[key])

    # 找到最多的money以及其对应的solution
    for index, value in enumerate(max_finals):
        if value >= max_final:
            max_final = value
            final_pt = final_pts[index]
    with open('output.csv', 'a') as f:
        f.write(str([init_water, init_food, max_final]).replace('[', '').replace(']', '') + '\n')

    # 返回最优值与其对应的最优解
    return final_pt, max_final


def dp_all(problem_no):
    # 多重搜索+动态规划 求解第一关
    loadEnvir(problem_no)
    global_max = 0
    final_pt: 'Solution' = None
    # 得到最优解的水和食物的组合
    max_ij = (0, 0)
    # foodWaterSearched = list()
    # for init_water in range(0, MAX_BURDEN // WATER_WEIGHT + 1):
    for init_water in range(177, 179):
        # init_water = i // WATER_WEIGHT
        init_food = (MAX_BURDEN - init_water * WATER_WEIGHT) // FOOD_WEIGHT
        # if (init_water, init_food) not in foodWaterSearched:
        # foodWaterSearched.append((init_water, init_food))
        final, max_final = dp_main(init_water, init_food)

        # 新的水和食物的组合获得的最优解和之前的全局最优解比较，取最大的
        if max_final > global_max:
            global_max = max_final
            final_pt = final
            max_ij = (init_water, init_food)

    # 递归地打印最优解经过的路线（从终点回溯到起点）
    while final_pt is not None:
        print(final_pt.step, final_pt.pt_index, final_pt.money, final_pt.key)
        final_pt = final_pt.prev

    # 打印最优解的水和食物，以及最多的money
    print(max_ij, global_max)


if __name__ == "__main__":
    dp_all(1)  # 第一关
