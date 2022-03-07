import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def showwholeplot(curdf):
    x_array = curdf["时间"]
    y_input_rotate_speed_array = curdf["输入转速"]
    y_output_rotate_speed_array = curdf["输出转速"]
    y_turbine_speed_array = curdf["涡轮转速"]
    y_target_gear = curdf['目标挡位']
    y_real_gear = curdf['实际挡位']
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 处理中文乱码
    fig = plt.figure()
    ax1=fig.add_subplot(111)
    l1 = ax1.plot(x_array, y_input_rotate_speed_array, color='blue', label='输入转速(r/min)')
    l2 = ax1.plot(x_array, y_output_rotate_speed_array, color='red', label='输出转速(r/min)')
    l3 = ax1.plot(x_array, y_turbine_speed_array, color='green', label='涡轮增压(r/min)')
    ax2 = ax1.twinx()
    l4 = ax2.plot(x_array, y_target_gear, color='skyblue', label='目标挡位')
    l5 = ax2.plot(x_array, y_real_gear, color='orange', label='实际挡位')
    # plt.plot(x_array, y_input_rotate_speed_array, 'ro-', x_array, y_output_rotate_speed_array, 'g+-',
    #          x_array, y_turbine_speed_array, 'b^-')
    plt.title('时间与转速折线图')
    ax2.set_ylabel(r"挡位")
    ax2.set_ylim(0, 10)
    ax1.set_xlabel('时间')
    ax1.set_ylabel('转速(r/min)')
    ax1.legend()
    ax2.legend()
    # plt.figure(figsize=(900, 480))
    plt.figure(dpi=150)
    plt.show()

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs
    使用了 pandas 提供功能的互相关函数

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

def wholecrosscorrwithplot(d1,d2,name1,name2):
    seconds = 10
    fps = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\n{name1} leads <> {name2} leads', ylim=[.1, .31], xlim=[0, seconds * fps*2], xlabel='Offset',
           ylabel='Pearson r')
    ax.set_xticklabels([int(item - 150) for item in ax.get_xticks()])
    plt.legend()
    plt.show()

if  __name__ == '__main__':
    # wholedf = pd.read_csv("~/Desktop/项目/AT台架实验数据/换挡数据/042211102001换挡数据.csv")
    wholedf = pd.read_csv("./AT台架实验数据/换挡数据/042211102001换挡数据.csv")

    wholedf.drop_duplicates('时间', 'first', inplace=True)     # 去重

    dfwithouthuandang=wholedf.copy(deep=True)


    dfwithouthuandang=dfwithouthuandang[dfwithouthuandang['目标挡位']==dfwithouthuandang['实际挡位']]

    showwholeplot(dfwithouthuandang)

    wholecrosscorrwithplot(dfwithouthuandang['输出转速'],dfwithouthuandang['油底壳温度'],'输出转速','油底壳温度')





