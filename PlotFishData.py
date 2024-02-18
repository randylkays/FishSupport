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
fig = plt.figure(constrained_layout=True)

gs = GridSpec(3, 6, figure=fig)

ax1 = fig.add_subplot(gs[0,:-2])
axt = fig.add_subplot(gs[0,4:])
axa = fig.add_subplot(gs[1,:2])
axg = fig.add_subplot(gs[1,4:])
axm = fig.add_subplot(gs[1,2:4])
spa = fig.add_subplot(gs[2,:2])  # ,polar=True)
# spap = fig.add_subplot(gs[2,1],polar=True)
# spg = fig.add_subplot(gs[2,4])  # ,polar=True)
# spgp = fig.add_subplot(gs[2,3],polar=True)
spm = fig.add_subplot(gs[2,2:4])  # ,polar=True)
# spmp = fig.add_subplot(gs[2,5],polar=True)
tbl = fig.add_subplot(gs[2,4:])
tbl.set_axis_off()

linexa,=axa.plot([],[], color="red", label="X")
lineya,=axa.plot([],[], color="blue", label="Y")
lineza,=axa.plot([],[], color="green", label="Z")
lineroa,= axa.plot([],[], color="purple", label="ro")

spphia, = spa.plot([],[], color="blue", label="phi")
spthea, = spa.plot([],[], color="green", label="theta")

linexg,=axg.plot([],[], color="red", label="X")
lineyg,=axg.plot([],[], color="blue", label="Y")
linezg,=axg.plot([],[], color="green", label="Z")
linerog,=axg.plot([],[], color="purple", label="ro")

# spphig, = spg.plot([],[], color="blue", label="phi")
# sptheg, = spg.plot([],[], color="green", label="theta")

linexm,=axm.plot([],[], color="red", label="X")
lineym,=axm.plot([],[], color="blue", label="Y")
linezm,=axm.plot([],[], color="green", label="Z")
linerom,=axm.plot([],[], color="purple", label="ro")

spphim, = spm.plot([],[], color="blue", label="phi")
spthem, = spm.plot([],[], color="green", label="theta")

lineTi,=axt.plot([],[], color="red", label="Int T")
lineTimu,=axt.plot([],[], color="blue", label="IMU T")
lineOutTmp,=axt.plot([],[], color="green", label="Out T")

linep,=ax1.plot([],[], color="red", label="Pressure")
lineph,=ax1.plot([],[], color="blue", label="PhotoLevel")
lineb,=ax1.plot([],[], color="green", label="Battery")
linesg,=ax1.plot([],[], color="black", label="Strain Gauge")
# sproap = spa.scatter([],[], cmap='hsv',alpha=0.75)
# sprogp = spg.scatter([],[], cmap='hsv',alpha=0.75)
# spromp = spm.scatter([],[], cmap='hsv',alpha=0.75)

ax1.legend()
axa.legend()
spa.legend()
# spg.legend()
spm.legend()
axg.legend()
axm.legend()
axt.legend()

axa.set_title("Accels")
axg.set_title("Gyros")
axm.set_title("Magnetic")
spa.set_label("Spherical Accel")
# spg.set_label("Spherical Gyro")
spm.set_label("Spherical Magnetic")
# sproap.set_label("Spherical Accel")
# sprogp.set_label("Spherical Gyro")
# spromp.set_label("Spherical Magnetic")
axt.set_title("Temperature")
ax1.set_title("Misc")
tbl.set_title("Data")

timehead = ["{:X}".format(i) for i in range(4)]
# val2 = ["{:02X}".format(10 * i) for i in range(10)] 
listname = ['Pressure Voltage','Internal Temp','PhotoR','Battery','imuT','xa','ya','za','SG','SG scale', 'Accel - ro', 'Mag - ro']
datadisplay = [["" for c in range(4)] for r in range(12)]

tbl.set_axis_off() 
# table = tbl.table( 
#     cellText = datadisplay,
#     rowLabels = listname,
#     colLabels = timehead,
#     rowColours =["palegreen"] * 11,
#     colColours =["palegreen"] * 3,
#     cellLoc ='center',
#     loc ='upper left')
# 
fig.suptitle("Fish")

ii=0

timestamp=datetime.datetime.now()
coolterms = "FromFish%04d%02d%02d_%02d%02d*.csv"%(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute)
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

def SphericalPlot(x,y,z,which):
    if x.size<100:
        ro = np.sqrt(x*x+y*y+z*z)
        phi = 90-np.arctan2(z,np.array(ro))*180/np.pi
        theta = 90-np.arctan2(y,x)*180/np.pi
        tt=t
    else:
        xs=x[-100:]
        ys=y[-100:]
        zs=z[-100:]
        tt=t[-100:]
        # j=xa.size-1
        ro = np.sqrt(x*x+y*y+z*z)
        ros=ro[-100:]
        phi = 90-np.arctan2(zs,np.array(ros))*180/np.pi
        theta = 90-np.arctan2(ys,xs)*180/np.pi

    # set angle limits
    phimax = phi.max()
    themax = theta.max()
    current_max = themax if (themax > phimax) else phimax
    phimin = phi.min()
    themin = theta.min()
    current_min = themin if (themin < phimin) else phimin
    delta=(current_max-current_min)/10
    print(current_min, current_max,delta) # romin,romax,phimin,phimax,themin,themax)
    if (which)=="a":
        if (ro.min()!=ro.max()):
            spa.set_xlim(tt[0],tt[-1])
            spa.set_ylim((current_min-delta), (current_max+delta))
            # sproa, =spa.plot(tt,ro, color="red")
            spphia,=spa.plot(tt,phi, color="blue")
            spthea,=spa.plot(tt,theta, color="green")
    elif (which)=="g":
        dummy=phimax
        # if (ro.min()!=ro.max()):
        #     spg.set_xlim(tt[0],tt[-1])
        #     spg.set_ylim((current_min-delta), (current_max+delta))
        #     # sprog, =spg.plot(tt,ro, color="red")
        #     spphig,=spg.plot(tt,phi, color="blue")
        #     sptheg,=spg.plot(tt,theta, color="green")
    elif (which)=="m":
        # print("Theta, Phi:",theta[-1],phi[-1])
        if (ro.min()!=ro.max()):
            spm.set_xlim(tt[0],tt[-1])
            spm.set_ylim((current_min-delta), (current_max+delta))
            # sprom, =spm.plot(tt,ro, color="red")
            spphim,=spm.plot(tt,phi, color="blue")
            spthem,=spm.plot(tt,theta, color="green")
    else:
        print("No Spherical Plot for ", which)
    return ro


def getdata():
    global data, t,xa,ya,za,xg,yg,zg,xm,ym,zm,pv,iT,ph,sg,bat,imuT,OutTmp
    data = pd.read_csv(listcoolterms[0])
    #t = [str(time.strptime(timestr, "%Y-%m-%d %H:%M:%S")) for timestr innp.asarray(data['Time'])
    # for timestr in data['Time']:
    #    print(timestr)
    #    t = pd.Timestamp(timestr).to_pydatetime()
    t = [pd.Timestamp(timestr).to_pydatetime() for timestr in data['Time']]
    xa =np.asarray(data['xa'])
    ya =np.asarray(data['ya'])
    za =np.asarray(data['za'])
    xg =np.asarray(data['xg'])
    yg =np.asarray(data['yg'])
    zg =np.asarray(data['zg'])
    xm =np.asarray(data['xm'])
    ym =np.asarray(data['ym'])
    zm =np.asarray(data['zm'])
    pv =np.asarray(data['Pressure Voltage'])
    iT =np.asarray(data['Internal Temp'])
    ph =np.asarray(data['PhotoR'])
    sg =np.asarray(data['SG'])
    bat =np.asarray(data['Battery'])
    imuT=np.asarray(data['imuT'])
    OutTmp=np.asarray(data['OutTmp'])
    
def animate(ii): # i):
    # print("Animate (t last):", t[0],t[-1], "xm.min/max:", xm.min(),xm.max())
    #------------ Plots -----------------
    # -- accel Plot --
    linexa.set_data(t,xa)
    lineya.set_data(t,ya)
    lineza.set_data(t,za)
    # accel Spherical Plot
    roa = SphericalPlot(xa,ya,za,"a")
    lineroa.set_data(t,roa)

    # set time limits
    axa.set_xlim(t[0],t[-1])
    axg.set_xlim(t[0],t[-1])
    axm.set_xlim(t[0],t[-1])
    axt.set_xlim(t[0],t[-1])
    ax1.set_xlim(t[0],t[-1])
    
    # set accel limits
    xmax = xa.max()
    ymax = ya.max()
    zmax = za.max()
    romax= roa.max()
    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    current_ymax = current_ymax if (current_ymax > romax) else romax
    
    xmin = xa.min()
    ymin = ya.min()
    zmin = za.min()
    romin= roa.min()
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin
    current_ymin = current_ymin if (current_ymin < romin) else romin
    delta=(current_ymax-current_ymin)/10
    # print(current_ymax,current_ymin,delta)
    axa.set_ylim((current_ymin-delta), (current_ymax+delta))
    
    # -- Gyro Plot -- 
    linexg.set_data(t,xg)
    lineyg.set_data(t,yg)
    linezg.set_data(t,zg)
    # Gyro Spherical Plot
    # rog = SphericalPlot(xg,yg,zg,"g")
    # linerog.set_data(t,rog)
    
    # Set gyro limits
    xmax = xg.max()
    ymax = yg.max()
    zmax = zg.max()
    # romax=rog.max()
    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    # current_ymax = current_ymax if (current_ymax > romax) else romax
    
    xmin = xg.min()
    ymin = yg.min()
    zmin = zg.min()
    # romin= rog.min()
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin
    # current_ymin = current_ymin if (current_ymin < zmin) else romin
    delta=(current_ymax-current_ymin)/10
    axg.set_ylim((current_ymin-delta), (current_ymax+delta))

    # -- Magnetic Plot --
    linexm.set_data(t,xm)
    lineym.set_data(t,ym)
    linezm.set_data(t,zm)
    # Magnetic Spherical Plot
    rom = SphericalPlot(xm,ym,zm,"m")
    linerom.set_data(t,rom)
    
    # Set magnetic limits
    xmax = xm.max()
    ymax = ym.max()
    zmax = zm.max()
    romax=rom.max()
    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    current_ymax = current_ymax if (current_ymax > romax) else romax
    
    xmin = xm.min()
    ymin = ym.min()
    zmin = zm.min()
    romin= rom.min()
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin
    current_ymin = current_ymin if (current_ymin < romax) else romax
    delta=(current_ymax-current_ymin)/10
    axm.set_ylim((current_ymin-delta), (current_ymax+delta))
    
    # -- Tmperature Plot -- 
    lineTi.set_data(t,iT)
    lineTimu.set_data(t,imuT)
    lineOutTmp.set_data(t,OutTmp)
    
    # set temperature limits
    xmax = iT.max()
    ymax = imuT.max()
    zmax = OutTmp.max()
    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    xmin = iT.min()
    ymin = imuT.min()
    zmin = OutTmp.min()
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
    smax = sg.max()
    smin = sg.min()

    current_ymax = xmax if (xmax > ymax) else ymax
    current_ymax = current_ymax if (current_ymax > zmax) else zmax
    current_ymin = xmin if (xmin < ymin) else ymin
    current_ymin = current_ymin if (current_ymin < zmin) else zmin

    if smin!=smax:
        scale=(current_ymax-current_ymin)/(smax-smin)*.25
        bee=(current_ymax+current_ymin)/2-scale*(smax+smin)/2
        sg_scaled=sg*scale+bee
    else:
        if smin!=0:
            scale=(current_ymax-current_ymin)/smin*.25
            bee=current_ymax-scale*smax
            sg_scaled=sg*scale+bee
        else:
            scale=1
            bee=0
            sg_scaled=sg

    linesg.set_data(t,sg_scaled)
    # print(sg_scaled.size, smin, smax,current_ymin,current_ymax, scale,bee)
    smax = sg_scaled.max()
    smin = sg_scaled.min()

    current_ymin = current_ymin if (current_ymin < smin) else smin
    current_ymax = current_ymax if (current_ymax > smax) else smax

    delta=(current_ymax-current_ymin)/10
    ax1.set_ylim((current_ymin-delta), (current_ymax+delta))

    datadisplay=[]
    if xa.size<10:
        timehead = ["{:X}".format(i) for i in range(4)]
        listname = ['imuT','Internal Temp','OutTmp','PhotoR','Pressure Voltage','Battery','xa','ya','za','SG','SG Scaled', 'Accel - ro', 'Mag - ro']
        datadisplay = [["" for c in range(4)] for r in range(12)]
    else:
        timehead = data.loc[xa.size-4:xa.size-1]['Time'].str.split(' ').str[1].values
        listname = ['imuT','Internal Temp','OutTmp','PhotoR','Pressure Voltage','Battery','xa','ya','za','SG',]
        datax = [data.loc[xa.size-4:xa.size][item].values for item in listname]
        datadisplay = [[np.format_float_positional(i, precision=4) for i in datax[j]] for j in range(10)]
        # print("datadisplay[1]:",type(datadisplay[1][1]), datadisplay[1][1])
        listname.append('SG Scaled')
        listname.append('Accel-ro')
        listname.append('Mag-ro')
        dummy=sg_scaled[sg_scaled.size-5:sg_scaled.size-1]
        dum = [np.format_float_positional(i, precision=4) for i in dummy]
        datadisplay.append(dum)
        dummy=roa[roa.size-5:roa.size-1]
        dum = [np.format_float_positional(i, precision=4) for i in dummy]
        # print(dum[1],dummy[1])
        datadisplay.append(dum)
        dummy=rom[rom.size-5:rom.size-1]
        dum = [np.format_float_positional(i, precision=4) for i in dummy]
        datadisplay.append(dum)
        # print(datadisplay[0].__len__(),datadisplay[1].__len__(),datadisplay[2].__len__(),datadisplay[3].__len__(),datadisplay[4].__len__(),datadisplay[5].__len__(),datadisplay[6].__len__(),datadisplay[7].__len__(),datadisplay[8].__len__(),datadisplay[9].__len__(),datadisplay[10].__len__())

    if ii>2:
        tbl.clear()
        tbl.set_axis_off()
    table = tbl.table(
        cellText = datadisplay,
        rowLabels = listname,
        colLabels = timehead,
        rowColours =["palegreen"] * 13,
        colColours =["palegreen"] * 4,
        cellLoc ='center',
        loc ='upper left')
    
    fig.canvas.draw()
    return scale, bee, sg_scaled
  


while True:
    getdata()
    timestamp=datetime.datetime.now()
    if(np.isnan(xm[-1])):
        print("read data issue, xm is nan", "Mag x,y,z:",xm[-1],ym[-1],zm[-1])
    else:
        # roac = np.sqrt(xa[-1]*xa[-1]+ya[-1]*ya[-1]+za[-1]*za[-1])
        # theta=90-np.arctan2(ya[-1],xa[-1])*180/np.pi
        # phi = 90-np.arctan2(za[-1],np.array(roac))*180/np.pi
        scale, bee, sg_scaled = animate(ii)
        print("Main:", ii, "time:", timestamp,"RP Time", t[-1],"Size:",xa.size, "Battery:", bat[-1], "imuT", imuT[-1], "SG min, max & diff", sg.min(), sg.max(), sg.max()-sg.min(), sg_scaled.min(), sg_scaled.max(),-sg_scaled.min()+sg_scaled.max())
        ii=ii+1
    # plt.legend()
    # plt.tight_layout()
    plt.pause(2) # Number of seconds you wait to update the plot
