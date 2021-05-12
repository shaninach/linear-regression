###########
# this code takes seeing and meteorological data and creates a linear regression model of sklearn
# lest squares linear model - minimize the residual sum of squares between the observed and predicted

###########
import os
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

########################## input- fill in ##################################
path = r"D:\Master\correlation"
os.chdir(path)
datafile = 'Seeing_Data.txt'
site = 'Neot_Smadar'
metfile = 'ims_data_march-april21.csv'
# functions #

def data(datafile):
    file = open(datafile, "r")
    data = []
    for line in file:               # converts the data .txt file to a list of lists 
        file = open(datafile, "r")
        stripped_line = line. strip()
        line_list = stripped_line. split()
        data.append(line_list)
        file.close() 
    
    Date = []
    LST = []
    Seeing = []
    r0 = []
    row = 0
    while row < len(data):             # extract values of Date, Hour (LST) and seeing from data - to new lists each.
        Date.append(data[row][0])
        LST.append(data[row][4])
        Seeing.append(float(data[row][10]))
        r0.append(float(data[row][12]))      # r0 in mm 
        row += 1

    d = {'Date': Date, 'LST': LST, 'Seeing': Seeing, 'r0': r0}
    df = pd.DataFrame(data=d)
    
    time = pd.to_datetime(df['Date'] + ' ' + df['LST'],format = '%d/%m/%Y %H:%M:%S')
    df = pd.DataFrame({'time': time, 'seeing': df.Seeing, 'r0': df.r0 })
    return df 

def splicing (df,start,end):
    """ input: datafile = .txt file of seeing data ("Seeing_Data.txt"), 
    start, end = date and time of first and last measuremnts wanted, in form dd-mm-yy HH:MM:SS'
    site = observation site. no spaces
        output: spliced table of cyclope data, of the night between start_time and end_time"""
    
    for i in list(range(0,len(df))):
        if df.time.iloc[i].hour == 0:            # fix date for hour 00:-- and 01:--
            df.time.iloc[i] += dt.timedelta(days=1)
        elif df.time.iloc[i].hour == 1:
            df.time.iloc[i] += dt.timedelta(days=1) 

    mask1 = (df['time'] >= start) & (df['time'] <= end) 
    df = df.loc[mask1]
    return df

###### extract the seeing and meteorological data. create a dataframe that contains both (merged)
df = data(datafile)
seeing_med = df.resample('D', on='time').median().dropna()
seeing_std = df.resample('D', on='time').std().dropna()
met = pd.read_csv(metfile,index_col = False, header=0,  names = ["Date","Hour_LST", "temp", "max_temp", "min_temp",\
                                                                       "Nan", "RH","Nan1","Nan2","Nan3","Nan4", "rain(mm)",\
                                                                    "wind_speed","wind_dir","std_wind_dir", "wind_gust",\
                                                                "wind_gust_dir", "max_min", "max_10min", "end_10"]).drop(['Nan','Nan1','Nan2','Nan3','Nan4'],axis=1)
met['time'] = pd.to_datetime(met['Date'] + ' ' + met['Hour_LST'], format = '%d/%m/%Y %H:%M')
met = met.drop(['Date','Hour_LST'],1)
met = met.set_index('time',drop=False)
met = met.between_time('18:00','05:00')  # take only between these hours
met[["temp", "max_temp", "min_temp","RH","rain(mm)","wind_speed","wind_dir","std_wind_dir","wind_gust","wind_gust_dir", "max_min", "max_10min", "end_10"]]\
    = met[["temp", "max_temp", "min_temp","RH","rain(mm)","wind_speed","wind_dir","std_wind_dir", "wind_gust","wind_gust_dir", "max_min", "max_10min", "end_10"]].astype(float) 
 
# create a dataframe with medians per day of met
met_med = met.resample('D', on='time').median().dropna()
# same but with std
met_std = met.resample('D', on='time').std().dropna() 
# merged dataframe of seeing and meteorolgy: 
merged = seeing_med.merge(met_med, on='time', how='outer').dropna()
merged_std = seeing_std.merge(met_std, on='time', how='outer').dropna()

###  linear regression test anf fit 
border = int(0.4*len(merged))
Xtrain = merged[['temp','RH','wind_speed','wind_gust','wind_dir']].iloc[:border,:] # other  'RH','wind_speed','wind_gust'
Ytrain = merged[['seeing']].iloc[:border,:]
Xtest = merged[['temp','RH','wind_speed','wind_gust','wind_dir']].iloc[border:,:] #'RH','wind_speed','wind_gust'
Ytest =  merged[['seeing']].iloc[border:,:]
lm = LinearRegression().fit(Xtrain,Ytrain) # perform linear regression  
r2 = lm.score(Xtrain,Ytrain)  # the R^2 between the prediction on Xtrain (which will give a vector) and Ytrain 
lm.coef_  # list of coefficients for the slope
lm.intercept_ # intercept point on y axis 
Xpred = lm.predict(Xtrain)
Ypred = lm.predict(Xtest)  # make predictions

name = metfile[9:].replace('csv','pdf')
## plot with the measurements vs prediction 
plt.figure()
plt.title(name)
#plt.scatter(Xtest,Ytest, s=12,c='cadetblue')
plt.scatter(Ytest,Ypred,c='lightseagreen',linewidth=1.5)
plt.plot(Ypred,Ypred)
plt.xlabel('seeing (arcsec)',fontsize=14)
plt.ylabel('seeing (arcsec)',fontsize= 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.axis('equal')
#plt.legend(['1:1 line','predictions'],loc='upper right')
plt.savefig(name)

plt.scatter(Ytest, Ypred)
# The coefficients
print('Coefficients:',lm.coef_)
# The mean squared error, and R squared 
print('Mean squared error: %.2f' % mean_squared_error(Ytest, Ypred))
print('R^2: %.2f' % r2_score(Ytest, Ypred))

# visualize data 
(fig), (ax) = plt.subplots(2,2, sharey = True,figsize = (7,7))
ax[0,0].scatter(merged['RH'],merged['seeing'],c='lightseagreen',s=5) 
a = np.poly1d(np.polyfit(merged['RH'], merged['seeing'], 1))
ax[0,0].plot(merged['RH'],a(merged['RH']),"g--")
ax[0,0].legend(['RH trendline','RH'])

ax[0,1].scatter(merged['temp'],merged['seeing'],c='lightgreen',s=5)
b = np.poly1d(np.polyfit(merged['temp'], merged['seeing'], 1))
ax[0,1].plot(merged['temp'],b(merged['temp']),"g--")
ax[0,1].legend(['temp trendline','temp'])

ax[1,0].scatter(merged['wind_speed'],merged['seeing'],c='green',s=5)
c = np.poly1d(np.polyfit(merged['wind_speed'], merged['seeing'], 1))
ax[1,0].plot(merged['wind_speed'],c(merged['wind_speed']),"g--")
ax[1,0].legend(['wind speed trendline','wind_speed'])

ax[1,1].scatter(merged['wind_gust'],merged['seeing'],c='gray',s=5)
d = np.poly1d(np.polyfit(merged['wind_gust'], merged['seeing'], 1))
ax[1,1].plot(merged['wind_gust'],d(merged['wind_gust']),"g--")
ax[1,1].legend(['wind gust trendline','wind_gust'])

# delete of values that are > mean+clip*sigma or lower than mean-clip*sigma:
clip = 2
mergedn = [merged[i][(merged[i] <= merged[i].mean()+clip*merged[i].std()) & \
                         (merged[i] >= merged[i].mean()-clip*merged[i].std())] for i in merged.columns[[0,2,5,7,10]]]
mergedn = pd.DataFrame({'seeing': mergedn[0], 'temp':mergedn[1], 'RH':mergedn[2],'wind_speed':mergedn[3],'wind_gust': mergedn[4]})
mergedn = mergedn.dropna()
    
# normalize the data by maximuצ value of each parameter
norm = [mergedn[i]/mergedn[i].max() for i in mergedn.columns]

# visualize data 
(fig), (ax) = plt.subplots(2,2,figsize = (7,7))
ax[0,0].scatter(norm[1],norm[0],c='lightseagreen',s=5) 
a = np.poly1d(np.polyfit(norm[1],norm[0], 1))
ax[0,0].plot(norm[1],a(norm[1]),"g-")
ax[0,0].legend(['RH trendline','RH'],loc='upper left')

ax[0,1].scatter(merged['temp'],merged['seeing'],c='lightgreen',s=5)
b = np.poly1d(np.polyfit(merged['temp'], merged['seeing'], 1))
ax[0,1].plot(merged['temp'],b(merged['temp']),"g-")
ax[0,1].legend(['temp trendline','temp'],loc='upper left')

ax[1,0].scatter(merged['wind_speed'],merged['seeing'],c='green',s=5)
c = np.poly1d(np.polyfit(merged['wind_speed'], merged['seeing'], 1))
ax[1,0].plot(merged['wind_speed'],c(merged['wind_speed']),"g-")
ax[1,0].legend(['wind speed trendline','wind_speed'],loc='upper left')

ax[1,1].scatter(merged['wind_gust'],merged['seeing'],c='gray',s=5)
d = np.poly1d(np.polyfit(merged['wind_gust'], merged['seeing'], 1))
ax[1,1].plot(merged['wind_gust'],d(merged['wind_gust']),"g-")
ax[1,1].legend(['wind gust trendline','wind_gust'],loc='upper left')
