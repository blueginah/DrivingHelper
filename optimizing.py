#!/usr/bin/env python
# coding: utf-8

# # Make each level to each Curve CSV FILE

# In[1]:


import pandas as pd
import numpy as np

# In[2]:


df_begin = []
n_row = []
f_1 = 'beginner_expert_processedData/beginner/beginner_'
f_3 = '.csv'
num_begin = 19
curveList = [[103.9, 209.3], [316.6, 399.6], [425.3, 517.9], [590.5, 756.9], [1048.7, 1110.5], [1212.3, 1437.1]]

df_concat = pd.DataFrame()
# for curve_num in range(0,6):
for curve_num in range(0,1):
    for idx in range(1, num_begin+1):
        tmp_file = f_1+str(idx)+'_new2'+f_3
        df = pd.read_csv(tmp_file)
        df = df.dropna()
        
        tmp = df.astype(float)
        tmp['level'] =0
        
        tmpcorner = tmp[(tmp['Distance'] >= curveList[curve_num][0]) & (tmp['Distance'] <= curveList[curve_num][1])]
        n_row.append(np.size(tmpcorner,0)) 
        
        df_begin.append(tmpcorner)
        df_concat = pd.concat([df_concat,df_begin[idx-1]])      
        
    df_concat.to_csv('cornerData/corner_'+str(curve_num+1)+'_begin'+'.csv')
    df_concat = pd.DataFrame()
    df_begin = []
    


# In[3]:


df_exp = []
f_1 = 'beginner_expert_processedData/expert/expert_'
f_3 = '.csv'
num_exp = 19

df_concat = pd.DataFrame()

# for curve_num in range(0,6):
for curve_num in range(0,1):
    for idx in range(1, num_exp+1):
        tmp_file = f_1+str(idx)+'_new2'+f_3
        df = pd.read_csv(tmp_file)
        df = df.dropna()

        tmp = df.astype(float)
        tmp['level'] =1

        tmpcorner = tmp[(tmp['Distance'] >= curveList[curve_num][0]) & (tmp['Distance'] <= curveList[curve_num][1])]
        n_row.append(np.size(tmpcorner,0)) 

        df_exp.append(tmpcorner)
        df_concat = pd.concat([df_concat,df_exp[idx-1]])
    
    df_concat.to_csv('cornerData/corner_'+str(curve_num+1)+'_expert'+'.csv')
    df_concat = pd.DataFrame()
    df_exp = []

#%% Recursive Feature Selections (RFE) 

import random
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

df_tmp_begin = pd.read_csv('cornerData/corner_1_begin.csv')
df_tmp_exp   = pd.read_csv('cornerData/corner_1_expert.csv')

df_curve1 = pd.concat([df_tmp_begin, df_tmp_exp])

#Hyper-parameters
left_column = [
#'Time',
    'Distance','GPS Latitude','GPS Longitude','Damper Velocity (Calc) FL','Damper Velocity (Calc) FR','Damper Velocity (Calc) RL',
'Damper Velocity (Calc) RR','Corr Dist','Corr Dist (Unstretched)','Corr Speed','Brake Pos',
'CG Accel Lateral','CG Accel Longitudinal','CG Accel Vertical','CG Height','Camber FL','Camber FR','Camber RL','Camber RR','Car Coord X',
'Car Coord Y','Car Coord Z','Car Pos Norm','Chassis Pitch Angle','Chassis Pitch Rate','Chassis Roll Angle','Chassis Roll Rate',
'Chassis Velocity X','Chassis Velocity Y','Chassis Velocity Z','Chassis Yaw Rate','Drive Train Speed','Engine RPM','Ground Speed',
'Ride Height FL','Ride Height FR','Ride Height RL','Ride Height RR','Road Temp','Self Align Torque FL','Self Align Torque FR',
'Self Align Torque RL','Self Align Torque RR','Session Time Left','Steering Angle','Suspension Travel FL','Suspension Travel FR',
'Suspension Travel RL','Suspension Travel RR','Tire Load FL','Tire Load FR','Tire Load RL','Tire Load RR','Tire Loaded Radius FL',
'Tire Loaded Radius FR','Tire Loaded Radius RL','Tire Loaded Radius RR','Tire Pressure FL','Tire Pressure FR','Tire Pressure RL','Tire Pressure RR',
'Tire Rubber Grip FL','Tire Rubber Grip FR','Tire Rubber Grip RL','Tire Rubber Grip RR','Tire Slip Angle FL','Tire Slip Angle FR',
'Tire Slip Angle RL','Tire Slip Angle RR','Tire Slip Ratio FL','Tire Slip Ratio FR','Tire Slip Ratio RL','Tire Slip Ratio RR',
'Tire Temp Core FL','Tire Temp Core FR','Tire Temp Core RL','Tire Temp Core RR','Tire Temp Inner FL','Tire Temp Inner FR',
'Tire Temp Inner RL','Tire Temp Inner RR','Tire Temp Middle FL','Tire Temp Middle FR','Tire Temp Middle RL',
'Tire Temp Middle RR','Tire Temp Outer FL','Tire Temp Outer FR','Tire Temp Outer RL','Tire Temp Outer RR','Toe In FL',
'Toe In FR','Toe In RL','Toe In RR','Wheel Angular Speed FL','Wheel Angular Speed FR','Wheel Angular Speed RL','Wheel Angular Speed RR',
'CG Distance','Lateral Velocity','Longitudinal Velocity','Lateral Acceleration','Longitudinal Acceleration','level']


df_curve1= df_curve1.loc[:,left_column]

df_curve1_saved = df_curve1.loc[:,left_column]

y = df_curve1.pop('level')
X = df_curve1

y = np.array(y)
X = np.array(X)

sc = StandardScaler()
sc.fit(X)

X_std = sc.transform(X)

svc = SVC(kernel='linear', C=0.3)
rfe = RFE(estimator=svc, n_features_to_select=7, step=1)
rfe.fit(X_std, y)
rank = rfe.ranking_
rank_reshape = rfe.ranking_.reshape((6, 17))


plt.matshow(rank_reshape, cmap=plt.cm.rainbow)
plt.colorbar()
plt.title("Ranking of Driving Features")
plt.show()

#%% Appending extracted feature columns in feature list 
rank_list = []
feature_list=[]
for i in range(0, len(rank)):
    if rank[i] == 1:
        rank_list.append(i)
   
for i in rank_list:
    feature_list.append(left_column[i])
print(rank_list)
print(feature_list)


#print(ranking.shape)

#%% Applying Features Extracted

print(feature_list)
##changing data columns
left_column = feature_list




#%% 
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 

df_tmp_begin = pd.read_csv('cornerData/corner_1_begin.csv')
df_tmp_exp   = pd.read_csv('cornerData/corner_1_expert.csv')

df_curve1 = pd.concat([df_tmp_begin, df_tmp_exp], ignore_index=True)
df_curve1= df_curve1.loc[:,left_column]
df_curve1_saved = df_curve1.loc[:,left_column]
df_curve1.to_csv('cornerData/corner_'+'_dfcurve1'+'.csv')
input_x = []
input_y = []
for i in range(0,num_begin + num_exp):
    xx = df_curve1_saved.loc[0:n_row[i]-1]
    df_curve1_saved.drop(range(0,n_row[i]),inplace=True)
    df_curve1_saved.reset_index(drop=True, inplace=True)
    yy = xx.pop('level')
    input_x.append(xx)
    input_y.append(yy)


input_xx = input_x
input_yy = input_y
n_roww = n_row
input_x = []
input_y = []
n_row = []
sequence = np.arange(num_begin + num_exp)
np.random.shuffle(sequence)

print(sequence)
for i in sequence:
    print(i)
    input_x.append(input_xx[i])
    input_y.append(input_yy[i])
    n_row.append(n_roww[i])
    # input_yy = input_y[np.argsort(sequence)]
print(input_x[0],input_y[0],n_row[0])
print(n_row[0])





# In[5]:


num_epochs = 1
batches = 1
learning_rate = 0.001
input_size = len(left_column)-1
output_size = 2
sequence_length = 50 #depend on data sample
hidden_size = 128
num_layers = 5
learning_rate = 0.005 #?
num_begin_train = 16
num_exp_train = 16
num_begin_test = num_begin - num_begin_train
num_exp_test = num_exp - num_exp_train

# def array_to_tensor(input_array):
#     tensor = torch.zeros(input_array.shape[0],input_array.shape[1])
#     torch.from_numpy()


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


gru = GRU(input_size, hidden_size, num_layers, output_size)

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(gru.parameters(), lr=learning_rate)  
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000

df_tmp_begin = pd.read_csv('cornerData/corner_1_begin.csv')
df_tmp_exp   = pd.read_csv('cornerData/corner_1_expert.csv')

df_curve1 = pd.concat([df_tmp_begin, df_tmp_exp], ignore_index=True)
df_curve1= df_curve1.loc[:,left_column]
df_curve1_saved = df_curve1.loc[:,left_column]
df_curve1.to_csv('cornerData/corner_'+'_dfcurve1'+'.csv')
input_x = []
input_y = []
for i in range(0,num_begin + num_exp):
    xx = df_curve1_saved.loc[0:n_row[i]-1]
    df_curve1_saved.drop(range(0,n_row[i]),inplace=True)
    df_curve1_saved.reset_index(drop=True, inplace=True)
    yy = xx.pop('level')
    input_x.append(xx)
    input_y.append(yy)


input_xx = input_x
input_yy = input_y
n_roww = n_row
input_x = []
input_y = []
n_row = []
sequence = np.arange(num_begin + num_exp)
np.random.shuffle(sequence)

print(sequence)
for i in sequence:
    print(i)
    input_x.append(input_xx[i])
    input_y.append(input_yy[i])
    n_row.append(n_roww[i])
    # input_yy = input_y[np.argsort(sequence)]
print(input_x[0],input_y[0],n_row[0])
print(n_row[0])




for epoch in range(num_epochs):
    for i in range(0,num_begin_train + num_exp_train):
        
#         x = df_curve1.loc[0:n_row[i]-1]
#         df_curve1.drop(range(0,n_row[i]),inplace=True)
#         df_curve1.reset_index(drop=True, inplace=True)
#         y = x.pop('level')
#         x.to_csv('cornerData/corner_'+'_asdafds'+str(i+1)+'.csv')
#         y.to_csv('cornerData/Y_'+'_asdafds'+str(i+1)+'.csv')
        
#         X = np.array(x)
#         X = X.reshape(-1,n_row[i],input_size)
#         Y = np.array(y)   

        X = np.array(input_x[i])
        X = X.reshape(-1,n_row[i],input_size)
        Y = np.array(input_y[i])   
        
        XX = torch.from_numpy(X)
        XX = XX.float()
        YY = torch.tensor([Y[0]])

        output = gru(XX)
        loss = criterion(output, YY)
        
#         # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print (f'Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i in range(num_begin_train + num_exp_train, num_begin + num_exp):
#         x = df_curve1.loc[0:n_row[i]-1]
#         df_curve1.drop(range(0,n_row[i]),inplace=True)
#         df_curve1.reset_index(drop=True, inplace=True)
#         y = x.pop('level')
#         x.to_csv('cornerData/corner_'+'_asdafds'+str(i+1)+'.csv')
#         y.to_csv('cornerData/Y_'+'_asdafds'+str(i+1)+'.csv')
        
#         X = np.array(x)
#         X = X.reshape(-1,n_row[i],input_size)
#         Y = np.array(y)   
        abc=1
        X = np.array(input_x[i])
        X = X.reshape(-1,n_row[i],input_size)
        Y = np.array(input_y[i])   

        XX = torch.from_numpy(X)
        XX = XX.float()
        YY = torch.tensor([Y[0]])

        output = gru(XX)
        _, predicted = torch.max(output.data, 1)
        n_samples += YY.size(0)
        n_correct += (predicted == YY).sum().item()        

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {num_begin_test + num_exp_test} test images: {acc} %')










