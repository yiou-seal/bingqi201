import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("~/Desktop/项目/AT台架实验数据/换挡数据/042211102001换挡数据.csv")

df.drop_duplicates('时间', 'first', inplace=True)     # 去重

x_array = df["时间"]
y_input_rotate_speed_array = df["输入转速(r/min)"]
y_output_rotate_speed_array = df["输出转速"]
y_turbine_speed_array = df["涡轮转速"]
y_target_gear = df['目标挡位']
y_real_gear = df['实际挡位']

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 处理中文乱码
l1 = plt.plot(x_array, y_input_rotate_speed_array, color='blue', label='输入转速(r/min)')
l2 = plt.plot(x_array, y_output_rotate_speed_array, color='red', label='输出转速(r/min)')
l3 = plt.plot(x_array, y_turbine_speed_array, color='green', label='涡轮增压(r/min)')
l4 = plt.plot(x_array, y_target_gear, color='skyblue', label='目标挡位')
l5 = plt.plot(x_array, y_real_gear, color='orange', label='实际挡位')
# plt.plot(x_array, y_input_rotate_speed_array, 'ro-', x_array, y_output_rotate_speed_array, 'g+-',
#          x_array, y_turbine_speed_array, 'b^-')
plt.title('时间与转速折线图')
plt.xlabel('时间')
plt.ylabel('转速(r/min)')
plt.legend()
plt.show()

print("Hello")

