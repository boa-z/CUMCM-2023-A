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
INIT_MONRY=0
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
    global INIT_MONRY
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
    if problem_no==1 or problem_no==2:
        WEATHER='高温,高温,晴朗,沙暴,晴朗,高温,沙暴,晴朗,高温,高温,沙暴,高温,晴朗,高温,高温,高温,沙暴,沙暴,高温,高温,晴朗,晴朗,高温,晴朗,沙暴,高温,晴朗,晴朗,高温,高温'.split(',')
        DAY_NUM=30
        MAX_BURDEN=1200
        INIT_MONRY=10000
        PROFIT=1000
        WATER_WEIGHT=3
        WATER_PRICE=5
        FOOD_WEIGHT=2
        FOOD_PRICE=10
        WATER_CONSUMPTION={'晴朗':5,'高温':8,'沙暴':10}
        FOOD_CONSUMPTION={'晴朗':7,'高温':6,'沙暴':10}
        POINT_NUM=12 if problem_no==1 else 17
        DESTINATION=9 if problem_no==1 else 12
        assert(len(WEATHER)==DAY_NUM)
        for i in range(POINT_NUM):
            POINTS.append([])
            MOVE_PLAN.append([])
            for j in range(POINT_NUM):
                MOVE_PLAN[i].append(0)            
        loadPoints('data/problem{}_graph_simple.csv'.format(problem_no))

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

def dp_main(init_water,init_food,start_day=0,init_pt=0,init_money=None,all_path=False):
    global WATER_ADD
    global FOOD_ADD
    global MONEY_REDUCE
    solution=[]
    for i in range(DAY_NUM+1):
        solution.append([])
        for j in range(POINT_NUM):
            solution[i].append({})            
    cur_key=getKey(init_water,init_food)
    if init_money==None:
        init_m=INIT_MONRY-init_water*WATER_PRICE-init_food*FOOD_PRICE
    else:
        init_m=init_money
    solution[start_day][init_pt][cur_key]=Solution(start_day,None,init_m,init_pt,cur_key)
    for step in range(start_day,DAY_NUM):
        real_date=step # 顺序法
        pt_list=list(range(POINT_NUM))
        shuffle(pt_list)
        for pt in pt_list:
            records=solution[step][pt]
            if len(records)==0:
                continue                   
            decisions=getDecision(POINTS[pt],real_date)
            shuffle(decisions)
            for d in decisions:
                _water,_food,_money=d.water,d.food,d.money #变化量
                for key in list(records.keys()):
                    cur_solution=records[key]
                    # if key=='00140228' and step==9 and pt==16:
                    #     print(9)
                    water,food=revertKey(key)
                    new_water=water-_water
                    new_food=food-_food
                    new_money=cur_solution.money-_money
                    last_cash=cur_solution.last_cash
                    last_supply=cur_solution.last_supply
                    last_supply_pt=cur_solution.last_supply_pt
                    if d.end.type==4 or d.end.type==1:
                        last_water,last_food=revertKey(last_supply)
                        last_cash=new_money
                        water_buy,food_buy=0,0
                        if new_water<0:
                            water_buy=-new_water
                        if new_food<0:
                            food_buy=-new_food
                        if water_buy>0 or food_buy>0:
                            if last_supply==cur_key:
                                continue
                            canBuy=False
                            if POINTS[last_supply_pt].players<2:
                                cost_k=2
                            if POINTS[last_supply_pt].players>1:
                                cost_k=4
                            cost=water_buy*WATER_PRICE*cost_k+food_buy*FOOD_PRICE*cost_k
                            if (last_water+water_buy)*WATER_WEIGHT+(last_food+food_buy)*FOOD_WEIGHT<=MAX_BURDEN and cost<=last_cash:
                                canBuy=True
                            if canBuy:
                                new_money=new_money-cost
                                new_water=new_water+water_buy
                                new_food=new_food+food_buy
                                WATER_ADD[last_supply]=water_buy
                                FOOD_ADD[last_supply]=food_buy
                                MONEY_REDUCE[last_supply]=cost
                                last_cash=new_money
                                last_supply=getKey(new_water,new_food)
                                last_supply_pt=d.end.name-1
                            else:
                                continue
                        else:
                            last_supply=getKey(new_water,new_food)
                            last_supply_pt=d.end.name-1
                    new_key=getKey(new_water,new_food)   
                    new_solution=Solution(step+1,cur_solution,new_money,d.end.name-1,new_key)
                    new_solution.last_cash=last_cash
                    new_solution.last_supply=last_supply
                    new_solution.last_supply_pt=last_supply_pt
                    cur_solution.next.append(new_solution)
                    if new_key in solution[step+1][d.end.name-1]:
                        if solution[step+1][d.end.name-1][new_key].money>new_solution.money:
                            continue
                    solution[step+1][d.end.name-1][new_key]=new_solution
    final=solution[DAY_NUM][DESTINATION]
    final_pts=[]
    final_pt=None
    max_finals=[]
    max_final=0
    for key in final:
        water,food=revertKey(key)
        if water<0 or food<0:
            continue
        max_finals.append(water*WATER_PRICE*0.5+food*FOOD_PRICE*0.5+final[key].money)
        final_pts.append(final[key])
    for index,value in enumerate(max_finals):
        if value>=max_final:
            max_final=value
            final_pt=final_pts[index]
    with open('output.csv','a') as f:
        f.write(str([init_water,init_food,max_final]).replace('[','').replace(']','')+'\n')
    if not all_path:
        return final_pt,max_final
    else:
        return final_pt,max_final,final_pts,max_finals

def dp_all(problem_no):
    # 多重搜索+动态规划 求解第一关和第二关
    loadEnvir(problem_no)
    global_max=0
    final_pt=None
    max_ij=(0,0)
    # foodWaterSearched = list()
    # for init_water in range(0, MAX_BURDEN // WATER_WEIGHT + 1):
    for init_water in range(177,179):
        # init_water = i // WATER_WEIGHT
        init_food = (MAX_BURDEN - init_water * WATER_WEIGHT) // FOOD_WEIGHT
        # if (init_water, init_food) not in foodWaterSearched:
            # foodWaterSearched.append((init_water, init_food))
        final,max_final=dp_main(init_water,init_food)
        if max_final>global_max:
            global_max=max_final
            final_pt=final
            max_ij=(init_water,init_food)

    while final_pt!=None:
        print(final_pt.step,final_pt.pt_index,final_pt.money,final_pt.key)
        final_pt=final_pt.prev
    print(max_ij,global_max)

if __name__ == "__main__":
    dp_all(1) # 第一关、第二关求解
