import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import datetime
import glob

plt.ion()
plt.style.use('fivethirtyeight')

# Initialise the subplot function using number of rows and columns
# figure, axis = plt.subplots(2,3)
fig = plt.figure(constrained_layout=True) # Figsize=(8,8)) ?

gs = GridSpec(2, 3, figure=fig)

# Presure, Photo & Battery
ax1 = fig.add_subplot(gs[0,0:])
# temperature
axt = fig.add_subplot(gs[1,0:2])
# Data table
tbl = fig.add_subplot(gs[1,2])
tbl.set_axis_off()

lineTi,=axt.plot([],[], color="red", label="Int T")
#lineThm,=axt.plot([],[], color="blue", label="Thermister")
lineICTmp,=axt.plot([],[], color="green", label="IC Tmp")

linep,=ax1.plot([],[], color="red", label="Pressure")
lineph,=ax1.plot([],[], color="blue", label="PhotoLevel")
lineb,=ax1.plot([],[], color="green", label="Battery")

ax1.legend()
axt.legend()

axt.set_title("Temperature")
ax1.set_title("Misc")
tbl.set_title("Data")

timehead = ["{:X}".format(i) for i in range(4)]
# Time,Thermister,Voltage,ThrCnt,Internal Temp degF,IT_Cnt,Tmp IC degF,PhotoR,PhtCnt,Press,PRcnt,Battery,BtCNt,MemAvailable
listname = ['Thermister','Internal Temp degF','Tmp IC degF','Press','PhotoR','Battery','MemAvailable']
datadisplay = [["" for c in range(4)] for r in range(7)]  # 6? 

tbl.set_axis_off() 
fig.suptitle("Fish")

ii=0

timestamp=datetime.datetime.now()
coolterms = "fromFish%04d%02d%02d_%02d%02d*.csv"%(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute)
listcoolterms=glob.glob(coolterms)
print("Source File search:", coolterms) # listcoolterms[0])
if listcoolterms==[]:
    coolterms = "FromFish%04d%02d%02d_%02d*.cvs"%(timestamp.year,timestamp.month,timestamp.day,timestamp.hour)
    listcoolterms=glob.glob(coolterms)
    if listcoolterms==[]:
        coolterms = "FromFish%04d%02d%02d*.csv"%(timestamp.year,timestamp.month,timestamp.day)
        listcoolterms=glob.glob(coolterms)
        if listcoolterms==[]:
            coolterms = "FromFish%04d%02d*.csv"%(timestamp.year,timestamp.month)
            listcoolterms=glob.glob(coolterms)
            if listcoolterms==[]:
                print("Data file not found in ", coolterms)
                exit()
print(coolterms, listcoolterms[0])


def getdata():
    global data, t,thm,iT,ICTmp,ph,pv,bat,mem
    # Time,Thermister,Voltage,ThrCnt,Internal Temp degF,IT_Cnt,Tmp IC degF,PhotoR,PhtCnt,Press,PRcnt,Battery,BtCNt,MemAvailable
    data = pd.read_csv(listcoolterms[0])
    # t = [str(time.strptime(timestr, "%Y-%m-%d %H:%M:%S")) for timestr in np.asarray(data['Time'])
    # for timestr in data['Time']:
    #    print(timestr)
    #    t = pd.Timestamp(timestr).to_pydatetime()
    t = [pd.Timestamp(timestr).to_pydatetime() for timestr in data['Time']]
    thm = np.asarray(data['Thermister'])
    iT = np.asarray(data['Internal Temp degF'])
    ICTmp = np.asarray(data['Tmp IC degF'])
    ph = np.asarray(data['PhotoR'])
    pv = np.asarray(data['Press'])
    bat = np.asarray(data['Battery'])
    mem = np.asarray(data['MemAvailable'])
    
def animate(ii): # i):
    # print("Animate (t last):", t[0],t[-1], "xm.min/max:", xm.min(),xm.max())
    #------------ Plots -----------------
    # set time limits
    axt.set_xlim(t[0],t[-1])
    ax1.set_xlim(t[0],t[-1])
    
    # -- Temperature Plot -- 
    lineTi.set_data(t,iT)
    lineThm.set_data(t,thm)
    lineICTmp.set_data(t,ICTmp)
    
    # set temperature limits
    xmax = iT.max()
    ymax = thm.max()
    zmax = ICTmp.max()
    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    xmin = iT.min()
    ymin = thm.min()
    zmin = ICTmp.min()
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin
    delta=(current_ymax-current_ymin)/10+1
    axt.set_ylim((current_ymin-delta), (current_ymax+delta))
    
    # -- Misc Temperature -- 
    linep.set_data(t,pv)
    lineph.set_data(t,ph)
    lineb.set_data(t,bat)
    
    # set misc limits
    xmax = pv.max()
    ymax = ph.max()
    zmax = bat.max()
    xmin = pv.min()
    ymin = ph.min()
    zmin = bat.min()

    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin


    delta=(current_ymax-current_ymin)/10
    ax1.set_ylim((current_ymin-delta), (current_ymax+delta))

    datadisplay=[]
    if ph.size<10:
        timehead = ["{:X}".format(i) for i in range(4)]
        # Time,Thermister,Voltage,ThrCnt,Internal Temp degF,IT_Cnt,Tmp IC degF,PhotoR,PhtCnt,Press,PRcnt,Battery,BtCNt,MemAvailable
        listname = ['Thermister','Internal Temp degF','Tmp IC degF','Press','PhotoR','Battery','MemAvailable']
        # old listname = ['imuT','Internal Temp','OutTmp','PhotoR','Pressure Voltage','Battery','xa','ya','za','SG','SG Scaled', 'Accel - ro', 'Mag - ro']
        datadisplay = [["" for c in range(4)] for r in range(7)]
    else:
        timehead = data.loc[ph.size-4:ph.size-1]['Time'].str.split(' ').str[1].values
        listname = ['Thermister','Internal Temp degF','Tmp IC degF','Press','PhotoR','Battery','MemAvailable']
        # old listname = ['imuT','Internal Temp','OutTmp','PhotoR','Pressure Voltage','Battery','xa','ya','za','SG',]
        datax = [data.loc[ph.size-4:ph.size][item].values for item in listname]
        datadisplay = [[np.format_float_positional(i, precision=4) for i in datax[j]] for j in range(7)]
        # print("datadisplay[1]:",type(datadisplay[1][1]), datadisplay[1][1

    if ii>2:
        tbl.clear()
        tbl.set_axis_off()
    table = tbl.table(
        cellText = datadisplay,
        rowLabels = listname,
        colLabels = timehead,
        rowColours =["palegreen"] * 7,
        colColours =["palegreen"] * 4,
        cellLoc ='center',
        loc ='upper left')
    
    fig.canvas.draw()
    # return scale, bee, sg_scaled
  


while True:
    getdata()
    timestamp=datetime.datetime.now()
    if(np.isnan(ICTmp[-1])):
        print("Read data issue, ICTmp is nan", "IC temperature",ICTmp[-1])
    else:
        # scale, bee, sg_scaled = animate(ii)
        animate(ii)
        print("Main(ii):", ii, "time:", timestamp,"RP Time", t[-1],"Size:",ph.size, "Battery:", bat[-1], "RP Mem:", mem[-1])
        ii=ii+1
    # plt.legend()
    # plt.tight_layout()
    plt.pause(2) # Number of seconds you wait to update the plot
