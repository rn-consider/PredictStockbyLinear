#股市行情数据获取和作图 -2
import sys
sys.path.insert(0, sys.path[0]+"/../")

from data.Ashare import *  # 导入父目录下data文件夹中的Ashare模块
from  utils.MyTT import *            #myTT麦语言工具函数指标库  https://github.com/mpquant/MyTT
    
# 证券代码兼容多种格式 通达信，同花顺，聚宽
# sh000001 (000001.XSHG)    sz399006 (399006.XSHE)   sh600519 ( 600519.XSHG ) 

df=get_price('000001.XSHG',frequency='1d',count=120)      #获取今天往前120天的日线实时行情
print('上证指数日线行情\n',df.tail(5))

#-------有数据了，下面开始正题 -------------
CLOSE=df.close.values;         OPEN=df.open.values           #基础数据定义，只要传入的是序列都可以 
HIGH=df.high.values;           LOW=df.low.values             #例如  CLOSE=list(df.close) 都是一样     

MA5=MA(CLOSE,5)                                #获取5日均线序列
MA10=MA(CLOSE,10)                              #获取10日均线序列
up,mid,lower=BOLL(CLOSE)                       #获取布林带指标数据

#-------------------------作图显示-----------------------------------------------------------------
import matplotlib.pyplot as plt ;  from matplotlib.ticker import MultipleLocator
plt.figure(figsize=(15,8))  
plt.plot(CLOSE,label='SHZS');    plt.plot(up,label='UP');           #画图显示 
plt.plot(mid,label='MID');       plt.plot(lower,label='LOW');
plt.plot(MA10,label='MA10',linewidth=0.5,alpha=0.7);
plt.show()