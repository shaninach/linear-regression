###########
# this code takes seeing and meteorological data and creates a linear regression model of sklearn
# lest squares linear model - minimize the residual sum of squares between the observed and predicted

###########
import os
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import sklearn

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
X = merged[['temp','RH','wind_speed','std_wind_dir','wind_gust','min_temp','max_temp','wind_dir']] # other 
#X = merged[['temp','RH','wind_speed','std_wind_dir','wind_gust','min_temp','max_temp',\
            #'wind_dir','wind_gust_dir','max_min','max_10min','end_10']] # other 
Y = merged['seeing'] 

lm = LinearRegression()  # create object for the class
model = lm.fit(X, Y)  # perform linear regression
Y_pred = lm.predict(X)  # make predictions
sklearn.metrics.r2_score(Y, Y_pred, sample_weight=None, multioutput='uniform_average')

# list of coefficients for the slope
lm.coef_   
# intercept point on y axis 
lm.intercept_

name = metfile[9:].replace('csv','pdf')
## plot with the measurements vs prediction 
plt.figure(figsize=[5,5])
plt.title(name)
plt.scatter(Y, Y_pred, s=12,c='cadetblue')
plt.plot(Y,Y, 'lightseagreen',linewidth=1.5)
plt.xlabel('seeing (arcsec)',fontsize=14)
plt.ylabel('seeing (arcsec)',fontsize= 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axis('equal')
plt.legend(['1:1 line','predictions'])
plt.savefig(name)

