import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from csotre import showwholeplot
from adtk.detector import SeasonalAD, AutoregressionAD, MinClusterDetector, PcaAD, VolatilityShiftAD, RegressionAD, \
    LevelShiftAD
from adtk.visualization import plot

# 定义一个函数，函数名字为get_all_excel，需要传入一个目录
def get_all_excel(dir):
    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘res’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'res' in file:
            if file.endswith('.csv'):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/res.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list

#用于分析换挡部分
def analysishuandang(huandnag34df,show合并各种分析方法的图=True):
    unuselist = \
        ["时间",
         "油底壳温度",
         "目标挡位",
         "实际挡位",
         # "涡轮转速",
         # "输入转速"
         ]

    huandnag34dfUNuse = huandnag34df[unuselist]
    huandnag34dfuse = huandnag34df.drop(unuselist, axis=1)
    huandnag34dfuse["新时间"] = pd.date_range(start='2019-1-09', periods=len(huandnag34dfuse), freq='S')
    huandnag34dfuse = huandnag34dfuse.set_index("新时间")
    # ax = plt.axes(np.array(huandnag34df["时间"]).tolist())

    res = pd.DataFrame({'res0': [0] * len(huandnag34dfuse)})
    # res['res0'] = res['res0'].astype(int)

    # 检测季节模式的异常变化。在内部，它被实现为带有transformer ClassicSeasonal分解的管道网。（对于现有的数据不推荐使用）
    # seasonal_ad = SeasonalAD(c=4.0, side="both")
    # anomalies = seasonal_ad.fit_detect(huandnag34dfuse)
    # plot(huandnag34dfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    # autoregression_ad = AutoregressionAD(n_steps=7 * 2, c=3.0)
    # anomalies = autoregression_ad.fit_detect(huandnag34dfuse)
    # plot(huandnag34dfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=7))
    anomalies1 = min_cluster_detector.fit_detect(huandnag34dfuse)
    plot(huandnag34dfuse, anomaly=anomalies1, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
         curve_group='all')
    plt.show()
    # anomalies1_mask = (anomalies1 == True)
    # anomalies1_mask = anomalies1_mask.reset_index()
    # anomalies1_mask = anomalies1_mask.drop(["新时间"], axis=1)
    anomalies1[anomalies1 == 'False'] = 0
    anomalies1[anomalies1 == 'True'] = 1
    anomalies1 = anomalies1.astype(int)
    anomalies1 = anomalies1.reset_index()
    anomalies1 = anomalies1.drop(["新时间"], axis=1)

    # res['res0'] = res['res0'] + anomalies1


    # 效果好
    pca_ad = PcaAD(k=2, c=8)
    anomalies2 = pca_ad.fit_detect(huandnag34dfuse)
    plot(huandnag34dfuse, anomaly=anomalies2, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
         curve_group='all')
    plt.show()
    anomalies2[anomalies2 == 'False'] = '0'
    anomalies2[anomalies2 == 'True'] = '1'
    anomalies2 = anomalies2.astype(int)
    anomalies2 = anomalies2.reset_index()
    anomalies2 = anomalies2.drop(["新时间"], axis=1)

    # volatility_shift_ad = VolatilityShiftAD(c=3.0, side='positive', window=30)
    # anomalies3 = volatility_shift_ad.fit_detect(huandnag34dfuse)
    # plot(huandnag34dfuse, anomaly=anomalies3, anomaly_color='red')
    # plt.show()

    regression_ad = RegressionAD(regressor=LinearRegression(), target="输入转速", c=3.0)
    anomalies3 = regression_ad.fit_detect(huandnag34dfuse)
    plot(huandnag34dfuse, anomaly=anomalies3, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
         curve_group='all')
    plt.show()
    anomalies3[anomalies3 == 'False'] = '0'
    anomalies3[anomalies3 == 'True'] = '1'
    anomalies3 = anomalies3.astype(int)
    anomalies3 = anomalies3.reset_index()
    anomalies3 = anomalies3.drop(["新时间"], axis=1)

    res['res0'] = res['res0'] + anomalies1 + anomalies2 + anomalies3


    id = huandnag34df.reset_index()
    res = pd.concat([id["index"], res], axis=1)
    res = res.set_index("index")
    res.loc[res['res0']>=1,'res0']=1

    if show合并各种分析方法的图:
        # id['时间']=pd.to_datetime(id["时间"])
        new故障点的标注=res.copy(deep=True)
        new故障点的标注.reset_index(drop=True,inplace=True)
        new故障点的标注["新时间"] = pd.date_range(start='2019-1-09', periods=len(huandnag34dfuse), freq='S')
        new故障点的标注=new故障点的标注.set_index("新时间")
        # new故障点的标注['resbool']='True'
        # new故障点的标注.loc[new故障点的标注['res0'] >= 1, 'resbool'] = 'True'
        # new故障点的标注.loc[new故障点的标注['res0'] < 1, 'resbool'] = 'False'
        # new故障点的标注=new故障点的标注.drop(['res0'], axis=1)
        # new故障点的标注=new故障点的标注['resbool'].astype(bool)
        # huandnag34dfuse.reset_index(drop=True,inplace=True)
        # huandnag34dfuse = pd.concat([id["index"], huandnag34dfuse], axis=1)
        # huandnag34dfuse.set_index("index")
        plot(huandnag34dfuse, anomaly=new故障点的标注, ts_linewidth=1, ts_markersize=3, anomaly_color='red',
             anomaly_alpha=0.3,
             curve_group='all')
        plt.show()

    return res

# 平稳时应该主要检测段之间方差的变化，在一段内部，检测值突变点
def analysispinwen(pingwendf):
    unuselist = \
        ["时间",
         "油底壳温度",
         "目标挡位",
         "实际挡位",
         # "涡轮转速",
         # "输入转速"
         ]

    pingwendfUNuse = pingwendf[unuselist]
    pingwendfuse = pingwendf.drop(unuselist, axis=1)
    pingwendfuse["新时间"] = pd.date_range(start='2019-1-09', periods=len(pingwendfuse), freq='S')
    pingwendfuse = pingwendfuse.set_index("新时间")
    # ax = plt.axes(pingwendf["时间"])

    res = pd.DataFrame({'res0': [0] * len(pingwendfuse)})

    # # 检测季节模式的异常变化。在内部，它被实现为带有transformer ClassicSeasonal分解的管道网。（对于现有的数据不推荐使用）
    # seasonal_ad = SeasonalAD(c=4.0, side="both")
    # anomalies = seasonal_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    # autoregression_ad = AutoregressionAD(n_steps=7 * 2, c=3.0)
    # anomalies = autoregression_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=2))
    anomalies = min_cluster_detector.fit_detect(pingwendfuse)
    plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
         curve_group='all')
    plt.show()


        # pca_ad = PcaAD(k=2, c=8)
        # anomalies = pca_ad.fit_detect(pingwendfuse)
        # plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
        #      curve_group='all')
        # plt.show()
        # anomalies[anomalies == 'False'] = '0'
        # anomalies[anomalies == 'True'] = '1'
        # anomalies = anomalies.astype(int)
        # anomalies = anomalies.reset_index()
        # anomalies = anomalies.drop(["新时间"], axis=1)
        # res['res0'] = res['res0'] + anomalies

    # level_shift_ad = LevelShiftAD(c=7.0, side='both', window=5)
    # anomalies = level_shift_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, anomaly_color='red')
    # plt.show()

    # 不知道为什么检测不出来
    volatility_shift_ad = VolatilityShiftAD(c=6.0, side="both", window=10)
    anomalies = volatility_shift_ad.fit_detect(pingwendfuse)
    plot(pingwendfuse, anomaly=anomalies, anomaly_color='red')
    plt.show()

    # regression_ad = RegressionAD(regressor=LinearRegression(), target="输入转速", c=3.0)
    # anomalies = regression_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
    #      curve_group='all')
    # plt.show()
    anomalies[anomalies == 'False'] = '0'
    anomalies[anomalies == 'True'] = '1'
    anomalies = anomalies.astype(int)
    anomalies = anomalies.reset_index()
    anomalies = anomalies.drop(["新时间"], axis=1)
    res['res0'] = res['res0'] + anomalies

    id = pingwendf.reset_index()
    res = pd.concat([id["index"], res], axis=1)
    res = res.set_index("index")
    return res

def analysispinwen2222(pingwendf,showplot=False):
    unuselist = \
        ["时间",
         "油底壳温度",
         "目标挡位",
         "实际挡位",
         # "涡轮转速",
         # "输入转速"
         ]

    pingwendfUNuse = pingwendf[unuselist]
    pingwendfuse = pingwendf.drop(unuselist, axis=1)
    pingwendfuse["新时间"] = pd.date_range(start='2019-1-09', periods=len(pingwendfuse), freq='S')
    pingwendfuse = pingwendfuse.set_index("新时间")
    # ax = plt.axes(pingwendf["时间"])

    res = pd.DataFrame({'res0': [0] * len(pingwendfuse)})

    # # 检测季节模式的异常变化。在内部，它被实现为带有transformer ClassicSeasonal分解的管道网。（对于现有的数据不推荐使用）
    # seasonal_ad = SeasonalAD(c=4.0, side="both")
    # anomalies = seasonal_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    # autoregression_ad = AutoregressionAD(n_steps=7 * 2, c=3.0)
    # anomalies = autoregression_ad.fit_detect(pingwendfuse)
    # plot(pingwendfuse, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
    # plt.show()

    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=2))
    anomalies = min_cluster_detector.fit_detect(pingwendfuse)

    if showplot:
        plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
             curve_group='all')
        plt.show()


        # pca_ad = PcaAD(k=2, c=8)
        # anomalies = pca_ad.fit_detect(pingwendfuse)
        # plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
        #      curve_group='all')
        # plt.show()
        # anomalies[anomalies == 'False'] = '0'
        # anomalies[anomalies == 'True'] = '1'
        # anomalies = anomalies.astype(int)
        # anomalies = anomalies.reset_index()
        # anomalies = anomalies.drop(["新时间"], axis=1)
        # res['res0'] = res['res0'] + anomalies

    level_shift_ad = LevelShiftAD(c=4.0, side='both', window=10)
    anomalies = level_shift_ad.fit_detect(pingwendfuse)
    plot(pingwendfuse, anomaly=anomalies, anomaly_color='red')
    plt.show()
    if showplot:
        plot(pingwendfuse, anomaly=anomalies, anomaly_color='red')
        plt.show()
    anomalies['axis_1'] = anomalies.loc[:, anomalies.columns.tolist()].apply(lambda x: x.sum(), axis=1)
    anomalies.loc[anomalies['axis_1']>=1,'axis_1']=1
    anomalies.reset_index(inplace=True)
    res['res0'] = res['res0'] + anomalies['axis_1']

    # 不知道为什么检测不出来
    volatility_shift_ad = VolatilityShiftAD(c=6.0, side="both", window=10)
    anomalies = volatility_shift_ad.fit_detect(pingwendfuse)

    if showplot:
        plot(pingwendfuse, anomaly=anomalies, anomaly_color='red')
        plt.show()
    anomalies['axis_1'] = anomalies.loc[:, anomalies.columns.tolist()].apply(lambda x: x.sum(), axis=1)
    anomalies.loc[anomalies['axis_1'] >= 1, 'axis_1'] = 1
    anomalies.reset_index(inplace=True)
    res['res0'] = res['res0'] + anomalies['axis_1']

    regression_ad = RegressionAD(regressor=LinearRegression(), target="输入转速", c=3.0)
    anomalies = regression_ad.fit_detect(pingwendfuse)

    if showplot:
        plot(pingwendfuse, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
             curve_group='all')
        plt.show()
    anomalies[anomalies == 'False'] = '0'
    anomalies[anomalies == 'True'] = '1'
    anomalies = anomalies.astype(int)
    anomalies = anomalies.reset_index()
    anomalies = anomalies.drop(["新时间"], axis=1)
    res['res0'] = res['res0'] + anomalies

    # 检测一下整体标准差,大于10认为不正常
    stddf=pingwendfuse.std().max()
    print(stddf)
    if(stddf>10):
        res['res0']=1

    id = pingwendf.reset_index()
    res = pd.concat([id["index"], res], axis=1)
    res = res.set_index("index")
    return res

def analysispinwenNEW(pingwendf):
    unuselist = \
        ["时间",
         "油底壳温度",
         "目标挡位",
         "实际挡位",
         # "涡轮转速",
         # "输入转速"
         ]

    pingwendfUNuse = pingwendf[unuselist]
    pingwendfuse = pingwendf.drop(unuselist, axis=1)
    pinwendfproces=pingwendf.reset_index()
    lasti=0
    starti=0
    endi=0
    flag=0
    finalres=0
    for index, row in pinwendfproces.iterrows():
        if index==0:
            lasti=row['index']
            starti=row['index']
            continue

        curi=row['index']
        if curi-lasti!=1:
            endi=curi-1
            pinwendfpian=pingwendf.loc[starti:endi,:]# 提取出了单次同一挡位平稳时间的数据
            subres=analysispinwen2222(pinwendfpian)
            if flag==0:
                finalres=subres
                flag=1
            else:
                finalres=pd.concat([finalres, subres], axis=0)
            starti=curi
        lasti=curi

    pinwendfpian = pingwendf.loc[starti:lasti, :]  # 提取出了单次同一挡位平稳时间的数据
    subres = analysispinwen2222(pinwendfpian)
    if flag == 0:
        finalres = subres
        flag = 1
    else:
        finalres = pd.concat([finalres, subres], axis=0)

    return finalres

if  __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 处理中文乱码
    # 获取目录中的指定文件
    filelist=get_all_excel('./AT台架实验数据/换挡数据')
    print(filelist)

    wholedf = pd.read_csv(filelist[0])
    wholedf.drop_duplicates('时间', 'first', inplace=True)     # 去重
    del (filelist[0])
    for fn in filelist:
        print(fn)
        thisdf = pd.read_csv(fn)
        thisdf.drop_duplicates('时间', 'first', inplace=True)     # 去重
        wholedf=pd.concat([wholedf,thisdf],axis=0)
    print(wholedf.isnull().sum())
    wholedf.reset_index(drop=True,inplace=True)

    rawtimeser=wholedf['时间']
    newtimedf=wholedf.drop(['时间'],axis=1)
    newtimedf['时间'] = range(1, len(newtimedf) + 1)
    showwholeplot(newtimedf)


    huandnag34df=wholedf[(wholedf['目标挡位']==5) & (wholedf['实际挡位']==4)]
    res=analysishuandang(huandnag34df)

    wholedfres=wholedf.copy(deep=True)
    wholedfres['res']=0


    # wholedfres=pd.concat([wholedfres,res],axis=1)# 老报错
    # wholedfres=wholedfres.fillna(0)
    # wholedfres['res']=wholedfres['res']+wholedfres['res0']
    # wholedfres = wholedfres.drop(["res0"], axis=1)
    print()

    pingwen3df = wholedf[(wholedf['目标挡位'] == 3) & (wholedf['实际挡位'] == 3)]
    res = analysispinwenNEW(pingwen3df)







