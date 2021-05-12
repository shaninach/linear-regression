#######

#########
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from treeinterpreter import treeinterpreter as ti
#from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

########################## input- fill in ##################################
path = r"D:\Master\correlation"
os.chdir(path)
datafile = 'Seeing_Data.txt'
metfile = 'ims_data_march-april21.csv'
site = 'Neot_Smadar'

# functions #
def data(datafile):
    """
    datafile = .csv file with seeing measurements.
    returns: pandas dataframe with date, seeing, r0 values.
    """
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

# extrcact seeing data 
df = data(datafile)
seeing_med= df.resample('5T', on='time').median().dropna()
seeing_med = seeing_med.resample('10T', axis=0).mean().dropna()
seeing_med['time'] = seeing_med.index
seeing_med.index.name = None
# seeing_std = df.resample('D', on='time').std().dropna()

# extrcact meteorological data 
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
met.index.name = None

# merged dataframe of seeing and meteorolgy: 
merged = seeing_med.merge(met, on='time', how='outer').dropna()
merged.describe()

# call regression model
clf = RandomForestRegressor()
train = merged[['seeing','RH','wind_speed']].iloc[:int(0.7*len(merged)),:]
pred = merged[['seeing','RH','wind_speed']].iloc[int(0.7*len(merged)):,:]

# train the model
clf.fit(train.iloc[:,1:], train['seeing'])

# predict on test data
predict = clf.predict(pred.iloc[:,1:])
clf.n_features_  # number of features (meteorological parameters)
clf.n_outputs_  # number of targets (seeing)
clf.n_estimators  # number of trees used 
sample_weight = pd.DataFrame({'names': train.columns[1:], 'weights': clf.feature_importances_})
#rsq = clf.score(train.iloc[:,1:], train.iloc[:,0]) # r^2 value calculated by (1-u/v), where u = residual sum of squares, v = total sum of squares. 
rsq2 = r2_score(pred.iloc[:,0], predict)

#prediction, bias, contributions = ti.predict(clf, pred.iloc[:,1:])

# summary table of the predicted vs real seeing values 
diff = pd.DataFrame({'predicted': predict})
diff['real'] = list(pred.iloc[:,0])
diff['diff'] = diff['real']-diff['predicted']
diff['diff %'] = (abs(diff['diff'])/diff['real'])*100

d = np.polyfit(diff['real'],diff['predicted'],1)
f = np.poly1d(d)

# plot predicted vs real values 
diff.insert(4,'regression',f(diff['real']))
mean = diff['real'].mean()
std = diff['real'].std()
diff = diff[(diff['real']< mean+std) & (diff['real']> mean-std) ]

#plot
fig = plt.figure()
ax = plt.gca()
X = diff['real']
Y = diff['predicted'] 
ax.scatter(X,Y, c='olive',s=10, marker='o')
ax.plot(X,X)
plt.xlabel('measured seeing (arcsec)')
plt.ylabel('estimated seeing (arcsec)')
#plt.axis('square')

# mean of difference in %:
print(' \nDifference between predicted and real values - \nmean: %.2f' %np.mean(diff['diff %']),'%','\nmedian: %.2f' %np.median(diff['diff %']), '%'\
      '\nR squared value:', '%.3f' %float(rsq2),'\n%i' %len(train), 'measurements were used for training, %i for prediction' %len(pred), '\nnumber of trees: %i' \
          %clf.n_estimators, '\n \n', sample_weight )

