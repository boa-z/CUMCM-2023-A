from random import random,choice,shuffle


class Point:
    def __init__(self,index):
        self.name=index
        self.neighbours=[]
        self.type=0
        self.players=0
        # 0 1 2 3 4: 普通 村庄 矿山 起点 终点

    def addNeighbour(self,pt):
        self.neighbours.append(pt)

    def setType(self,t):
        self.type=t

class Solution:
    def __init__(self,step,prev,money,pt,key):
        self.step=step
        self.prev=prev
        self.next=[]
        self.money=money
        self.pt_index=pt
        self.key=key
        self.last_supply=key
        self.last_supply_pt=0
        self.last_cash=money

class Decision:
    def __init__(self,start,end,weather,getMineral=False):
        self.start,self.end,self.weather=start,end,weather
        self.water=self.food=self.money=0 # 消耗为正 赚得为负 水或食物单位为箱
        if start.name==end.name and start.type!=4:
            if getMineral:
                self.getMineral(weather)
            else:
                self.water=WATER_CONSUMPTION[weather]
                self.food=FOOD_CONSUMPTION[weather]
        if start.name!=end.name:
            self.water=2*WATER_CONSUMPTION[weather]
            self.food=2*FOOD_CONSUMPTION[weather]
            if MOVE_PLAN[start.name-1][end.name-1]>1:
                self.water=self.water*MOVE_PLAN[start.name-1][end.name-1]
                self.food=self.food*MOVE_PLAN[start.name-1][end.name-1]


    def getMineral(self,weather):
        self.water=3*WATER_CONSUMPTION[weather]
        self.food=3*FOOD_CONSUMPTION[weather]
        if POINTS[self.start.name-1].players>1:
            if MINING>0:
                self.money=-PROFIT/MINING
            else:
                self.money=-PROFIT
        else:
            self.money=-PROFIT

WEATHER=[]
DAY_NUM=0
MAX_BURDEN=0
INIT_MONEY=0
PROFIT=0
WATER_WEIGHT=0
WATER_PRICE=0
FOOD_WEIGHT=0
FOOD_PRICE=0
WATER_CONSUMPTION={}
FOOD_CONSUMPTION={}
POINTS=[]
POINT_NUM=0
DESTINATION=0
MOVE_PLAN=[]
MINING=0
WATER_ADD={}
FOOD_ADD={}
MONEY_REDUCE={}

def loadPoints(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line=line.split(',')
            p=Point(int(line[0]))
            POINTS[int(line[0])-1]=p
            if line[2].replace('\n','')!='':
                p.setType(int(line[2]))
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line=line.split(',')
            if line[1]!='':
                for pt in line[1].split(' '):
                    POINTS[int(line[0])-1].addNeighbour(POINTS[int(pt)-1])

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

    if problem_no==6:
        DAY_NUM=30
        MAX_BURDEN=1200
        INIT_MONEY=10000
        PROFIT=1000
        WATER_WEIGHT=3
        WATER_PRICE=5
        FOOD_WEIGHT=2
        FOOD_PRICE=10
        WATER_CONSUMPTION={'晴朗':3,'高温':9,'沙暴':10}
        FOOD_CONSUMPTION={'晴朗':4,'高温':9,'沙暴':10}
        POINT_NUM=25
        DESTINATION=24
        for i in range(POINT_NUM):
            POINTS.append([])
            MOVE_PLAN.append([])
            for j in range(POINT_NUM):
                MOVE_PLAN[i].append(0)  
        loadPoints('data/problem6_graph.csv') 

def getDecision(point,day):
    decision_list=[]
    if WEATHER[day]!='沙暴':
        for pt in point.neighbours:
            decision_list.append(Decision(point,pt,WEATHER[day]))
    decision_list.append(Decision(point,point,WEATHER[day]))
    if point.type==2:
        decision_list.append(Decision(point,point,WEATHER[day],getMineral=True))
    return decision_list

def getKey(water,food):
    return str(water).zfill(4)+str(food).zfill(4)

def revertKey(key):
    return int(key[0:4]),int(key[4:8])


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


def dp_game(risk):
    # 多重静态博弈 求解第六关
    water=[200,200,200]
    food=[300,300,300]
    player_pt=[0,0,0]
    money=[6000,6000,6000]
    path=[[],[],[]]
    death=[False,False,False]
    global MINING
    global WATER_ADD
    global FOOD_ADD
    global MONEY_REDUCE
    global WEATHER
    for step in range(0,30):
        next_state=[None,None,None]
        MINING=0
        WATER_ADD,FOOD_ADD,MONEY_REDUCE={},{},{}
        for p in range(0,3):
            if death[p]:
                continue
            final_pt,max_final=dp_main(water[p],food[p],step,player_pt[p],money[p])
            while final_pt!=None:
                # print(final_pt.step,final_pt.pt_index,final_pt.money,final_pt.key)
                if final_pt.step==step+1:
                    next_state[p]=final_pt
                    break
                final_pt=final_pt.prev
            if next_state[p]==None:
                death[p]=True
                money[p]=0
                continue
            if next_state[p].money>money[p]:
                MINING=MINING+1
        for i in range(POINT_NUM):
            POINTS[i].players=0
            for j in range(POINT_NUM):
                MOVE_PLAN[i][j]=0
        for p in range(0,3):
            if death[p]:
                continue
            next_pt_index=next_state[p].pt_index
            POINTS[next_pt_index].players=POINTS[next_pt_index].players+1
            MOVE_PLAN[player_pt[p]][next_pt_index]=MOVE_PLAN[player_pt[p]][next_pt_index]+1
        for p in range(0,3):
            if death[p]:
                continue
            final_pt,max_final,final_pts,max_finals=dp_main(water[p],food[p],step,player_pt[p],money[p],all_path=True)
            best_plan,good_plan=None,None
            while final_pt!=None:
                if final_pt.step==step+1:
                    best_plan=final_pt
                    break
                final_pt=final_pt.prev
            if best_plan==None:
                death[p]=True
                money[p]=0
                continue
            conflict=False
            for other in range(0,3):
                if p==other:
                    continue
                if death[other]:
                    continue
                if best_plan.pt_index==next_state[other].pt_index:
                    conflict=True
            if  conflict:
                if random()<risk[p]:
                    _water1,_food1=revertKey(best_plan.key)
                    _water2,_food2=revertKey(best_plan.prev.key)
                    water[p]=water[p]+_water1-_water2
                    food[p]=food[p]+_food1-_food2
                    if best_plan.pt_index in WATER_ADD:
                        water[p]=water[p]+WATER_ADD[best_plan.pt_index]
                    if best_plan.pt_index in FOOD_ADD:
                        food[p]=food[p]+FOOD_ADD[best_plan.pt_index]                    
                    money[p]=money[p]+best_plan.money-best_plan.prev.money
                    if best_plan.pt_index in MONEY_REDUCE:
                        money[p]=money[p]-MONEY_REDUCE[best_plan.pt_index]
                    player_pt[p]=best_plan.pt_index
                    path[p].append(best_plan.pt_index+1)
                else:
                    good_max=0
                    for index,plan in enumerate(final_pts):
                        while plan!=None:
                            if plan.step==step+1:
                                if plan.pt_index!=best_plan.pt_index:
                                    if max_finals[index]>good_max:
                                        good_max=max_finals[index]
                                        good_plan=plan
                                break
                            plan=plan.prev
                    if good_plan==None:
                        good_plan=best_plan
                    _water1,_food1=revertKey(good_plan.key)
                    _water2,_food2=revertKey(good_plan.prev.key)
                    water[p]=water[p]+_water1-_water2
                    food[p]=food[p]+_food1-_food2
                    if good_plan.pt_index in WATER_ADD:
                        water[p]=water[p]+WATER_ADD[good_plan.pt_index]
                    if good_plan.pt_index in FOOD_ADD:
                        food[p]=food[p]+FOOD_ADD[good_plan.pt_index]
                    money[p]=money[p]+good_plan.money-good_plan.prev.money
                    if good_plan.pt_index in MONEY_REDUCE:
                        money[p]=money[p]-MONEY_REDUCE[good_plan.pt_index]
                    player_pt[p]=good_plan.pt_index
                    path[p].append(good_plan.pt_index+1)
            else:
                _water1,_food1=revertKey(best_plan.key)
                _water2,_food2=revertKey(best_plan.prev.key)
                water[p]=water[p]+_water1-_water2
                food[p]=food[p]+_food1-_food2
                if best_plan.pt_index in WATER_ADD:
                    water[p]=water[p]+WATER_ADD[best_plan.pt_index]
                if best_plan.pt_index in FOOD_ADD:
                    food[p]=food[p]+FOOD_ADD[best_plan.pt_index]
                money[p]=money[p]+best_plan.money-best_plan.prev.money
                if best_plan.pt_index in MONEY_REDUCE:
                    money[p]=money[p]-MONEY_REDUCE[best_plan.pt_index]
                player_pt[p]=best_plan.pt_index
                path[p].append(best_plan.pt_index+1)
        print(path,money)
    with open('game.csv','a') as f:
        f.write(str([risk[0],risk[1],risk[2],money[0],money[1],money[2]]).replace('[','').replace(']','')+'\n')
    with open('path.csv','a') as f:
        f.write(str(path)+'\n')
    with open('weather.csv','a') as f:
        f.write(str(WEATHER).replace('[','').replace(']','')+'\n')

def dp_game_all():
    loadEnvir(6)
    global WEATHER
    global DAY_NUM
    for i in range(1000):
        WEATHER=[]
        for i in range(DAY_NUM):
            rd=random()
            if rd<0.1:
                WEATHER.append('沙暴')
            elif rd<0.55:
                WEATHER.append('晴朗')
            else:
                WEATHER.append('高温')
        risk=[random(),random(),random()]
        print(risk)
        dp_game(risk)


if __name__ == "__main__":
    dp_game_all() # 第六关求解
