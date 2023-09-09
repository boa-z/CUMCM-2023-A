import p1
import pandas as pd
from datetime import datetime
import csv

# 计算所有镜子的平均效率啥的
# prompt: 下面是一个关于Station的类，其中初始化输入 heli_para、tower_para 为列表，表示x,y,z轴坐标。
# heli_para 的x,y从df中读取，df的格式见附录，z为定值4，tower_para = [0, 0, 84]
# 请写一段python代码，通过.eta_cos可以获取 eta_cos的值，遍历df所有的值，输出到列表 heli_output 中

df = pd.read_excel('/Users/boa/Documents/2023-2024-1暑假/CUMCM2023/data/附件.xlsx')
# 设置起始日期，这里以2023年1月21日为例
start_date = datetime(2023, 1, 21)

# 设置结束日期，这里以2023年12月21日为例
end_date = datetime(2023, 12, 21)

# 定义要打印的时间列表
times = ["9:00", "10:30", "12:00", "13:30", "15:00"]

avg_month_eta_cos = []
avg_month_eta_sb = []
avg_month_eta_trunc = []
avg_month_eta_at = []
avg_month_eta = []

# 循环遍历每个月的21日
current_date = start_date
while current_date <= end_date:
    avg_times_eta_cos = []
    avg_times_eta_sb = []
    avg_times_eta_trunc = []
    avg_times_eta_at = []
    avg_times_eta = []
    # 打印当前日期的每个时间
    for time in times:
        hour, minute = map(int, time.split(':'))
        local_time = datetime(current_date.year, current_date.month, 21, hour, minute, 0)
        heli_eta_cos_output = []
        heli_eta_sb_output = []
        heli_eta_trunc_output = []
        heli_eta_at_output = []
        heli_eta_output = []

        for i, row in df.iterrows():
            heli_para = [row['x坐标 (m)'], row['y坐标 (m)'], 4, 6, 6, []]
            tower_para = [0, 0, 84]
            station = p1.Station(heli_para, tower_para)
            eta_all = p1.each_mirror_output_all(station.heli_para, station.tower_para)
            heli_eta_cos_output.append(eta_all[0])

            print(len(heli_eta_cos_output))
            heli_eta_sb_output.append(eta_all[1])
            heli_eta_trunc_output.append(eta_all[2])
            heli_eta_at_output.append(eta_all[3])
            heli_eta_output.append(eta_all[4])

            # 对应
            min_data = {"heli_eta_cos_output": heli_eta_cos_output,
                    "heli_eta_sb_output": heli_eta_sb_output,
                    "heli_eta_trunc_output": heli_eta_trunc_output,
                    "heli_eta_at_output": heli_eta_at_output,
                    "heli_eta_output": heli_eta_output}
            min_df = pd.DataFrame(min_data)
            csv_name = str(local_time) + '.csv'
            min_df.to_csv(csv_name, mode='w', header=False)

            # 计算平均值并添加到平均时间列表
            avg_time_eta_cos = sum(heli_eta_cos_output) / len(heli_eta_cos_output)
            avg_times_eta_cos.append(avg_time_eta_cos)
            avg_time_eta_sb = sum(heli_eta_sb_output) / len(heli_eta_sb_output)
            avg_times_eta_sb.append(avg_time_eta_sb)
            avg_time_eta_trunc = sum(heli_eta_trunc_output) / len(heli_eta_trunc_output)
            avg_times_eta_trunc.append(avg_time_eta_trunc)
            avg_time_eta_at = sum(heli_eta_at_output) / len(heli_eta_at_output)
            avg_times_eta_at.append(avg_time_eta_at)

            print(local_time, avg_time_eta_cos, avg_time_eta_sb, avg_time_eta_trunc, avg_time_eta_at)

            # 计算5个时间的平均值的平均值并添加到月平均列表
        avg_month_eta_cos.append(sum(avg_times_eta_cos) / len(avg_times_eta_cos))
        avg_month_eta_sb.append(sum(avg_times_eta_sb) / len(avg_times_eta_sb))
        avg_month_eta_trunc.append(sum(avg_times_eta_trunc) / len(avg_times_eta_trunc))
        avg_month_eta_at.append(sum(avg_times_eta_at) / len(avg_times_eta_at))
        print(avg_month_eta_cos, avg_month_eta_sb, avg_month_eta_trunc, avg_month_eta_at)

    # 增加一个月
    if current_date.month == 12:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(month=current_date.month + 1)