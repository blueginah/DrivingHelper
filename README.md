# Development of AI model for Racing Driving Level Classification

Classifying high-skilled drivers and low-skilled drivers using data extracted from Assetto-Corsa

## Motivaiton & Social Applications
![image](https://user-images.githubusercontent.com/82494923/122663979-6b1b6d00-d1d9-11eb-96da-f07ebead6209.png)

Firstly, it helps to prevent car accidents. Researchers are trying hard to develop full autonomous driving AI, which does not require drivers' attention, still has a long way to it. Current autonomous driving AI requires drivers' full attention and it just assists drivers. However, if the driving skill of assistance system and that of drivers have a great difference? There is possible danger of accident if the driver does no manage to handle the vehicle. Thus, it is essential to evaluate the driving skill of the driver and provide them with adequate level of driving assistance system. 

Secondly, it helps to provide driver with custrom autonomous driving AI. What if every vehicle on the has the same driving AI? I don't think people are satisfied with it, because every person has different driving habits. For example, some want to get to the destination earlier, while others prefer safe and defensive driving. 
   
Finally, drivers can choose the vehicle that perfectly matches themselves. When buying a new car, people used to mostly consider their financial conditions. However, nowadays they more consider confortness and diverse experience that the car provides. It will be helpful to choose the right car, if it is possible to evaluate the driver's drving skills.


## Problem Formulation and Data

![image](https://user-images.githubusercontent.com/82494923/122663622-f8a98d80-d1d6-11eb-9452-adcf1accfeff.png)
![image](https://user-images.githubusercontent.com/82494923/122663623-019a5f00-d1d7-11eb-9e31-c07042d655b7.png)
![image](https://user-images.githubusercontent.com/82494923/122663636-12e36b80-d1d7-11eb-859a-9fed1929e5de.png)

1. Used the software named "Assetto Corsa" to get all the data. Although it is known as driving game, it provides over 180 features of car condition. 
2. Used Logitech G27 Steering Wheel, Accelerator and Brake.
3. Chose ks_highlands track, which is composed of gentle curves.
4. Chose Pagani Zonda R as a vehicle.
5. Colleted data with external program called ACTI and extracted into csv file.

![image](https://user-images.githubusercontent.com/82494923/122663860-8639ad00-d1d8-11eb-8918-b51b87b8db21.png)
![image](https://user-images.githubusercontent.com/82494923/122663863-88037080-d1d8-11eb-999c-f0f5be1222b5.png)

The columns are list of features of car condition, and the rows are timeseries data recorded every 0.5 seconds. It is possible to control the time step.
There are over 180 features and it looks likely to be able to specify the level of driving skill with them.


## Methods
### Pre-processing Data
![image](https://user-images.githubusercontent.com/82494923/122664009-969e5780-d1d9-11eb-892e-d4fe17ceb36d.png)

1. All Features
2. Selected Features

Firstly, we used data with all features. As we are not expert in cars, we are no sure which feature plays a key role in classifying the level of drivers. So, we used whole bunch of data at once.

Secondly, we used data with selected features. Features, which the program provide with, includes lots of features that is from speicial sensors which professional drivers use. 
So we used selected feature. They are composed of features that can easily get from general sensors which normal drivers can access. Such as, GPS, Accelerometer, Ground speed etc.

### Problem Solving Methods

![image](https://user-images.githubusercontent.com/82494923/122665270-37444580-d1e1-11eb-9a4b-94a7b1f013e5.png)

1. SVM (Support Vector Machine)
2. LSTM (Long Short-Term Memory)
3. GRU (Gated Recurrent Units)

Fistly, we used SVM to create the classification model. SVM is a machine learning model which provides powerful classification. When the data and the labels of the data are given, SVM algoritem creates a model that judges where the new data belong to. It is intuitive if the data is simple and requires linear model. However, if there are lots of lables or features, it require additional technique, such as non-linear modeling and PCA(Principal Component Analysis). PCA is a dimensionality-reduction mehod used to minimize the the dimesion of the dataset. Since there are over 180 feature, it is essention to use PCA to extract major factors that help to create a classification model. However, the SVM does not handle time-series data. Each row of the time-series data is single datum.

Secondly, we used LSTM to create the classification model taking time-series data into account. While the SVM is a machine learning model, LSTM is a deep learning model. There are lots of deep learning model that treats time-series data such as RNN. Many of them fail to make good model with long-term data because the significance of the former data gets smaller exponentially. However, LSTM can handle data with long-term time series data because it has cell state which let data flow along it while unchanged.

Finally, we used GRU to create the classification model taking time-series data into account. Like LSTM, GRU is a deep learning model which take long-term time-series data into account. The advantage over LSTM is that it has simpler layer composition which leads to shorter computational time. It is recently introduced, so we used GRU to check its performance.



## Preliminary Results
### SVM
![image](https://user-images.githubusercontent.com/82494923/122665338-9bffa000-d1e1-11eb-86b6-48145bfc3fa8.png)
![image](https://user-images.githubusercontent.com/82494923/122665373-c5b8c700-d1e1-11eb-9d06-14a73815ed55.png)

From the table, all results are over 90% and it is easy to conclude that the trained classification model is reasonable. The average cross-validation score of all features is higher than selected features. Regardless of the number of curves and type of curve, it showed robush results. Furthermoe, ROC(Receiver Operating Characteristic) curve is used to evaluate the performance and convergence of the machine learning model and it is desirable when the blue curve is goes near to left top corner of the graph. Above ROC curver shows that the trained model has great performance. Morover, Toe in of front tyres came out to be the most significant factor that attributes to create classification model. It is reasonable because Toe in of front tyres is known as indicator of stability while driving of the curve.


### LSTM
![image](https://user-images.githubusercontent.com/82494923/122665671-855a4880-d1e3-11eb-9805-b87b69a017d6.png)

In the case of LSTM, the performance highly depended on curves. There are total seven trials. Six trials from the single curve among of six surves and one trial from all curves. Only the first two curves showed accuracy over 90. Whether using data with all features or selected features did not affected the results a lot.


### GRU
![image](https://user-images.githubusercontent.com/82494923/122665794-4d9fd080-d1e4-11eb-9198-89d7ed09b1b7.png)

In the case of GRU, like LSTM the performance highly depended on curves. There are total seven trials. Six trials from the single curve among of six surves and one trial from all curves. Only the first curve showed accuracy over 90. Whether using data with all features or selected features did not affected the results a lot.


### Discussion

The performance of the SVM was way better than LSTM and GRU. We analyzed the reason why SVM was the best.

1. It is binary classification with lots of features. Thus, it is evident that SVM is more advantageous. If there are more than two lables and less features, the outcome is expected to be opposite.
2. While the SVM treats each time step in time-series data as a single sample data, LSTM and GRU treat whole time-series data of single lap as a single data. So, the number of sample data of SVM is way larger. Thousands of sample data for SVM while 38 sample data for LSTM and GRU. In order to overcome the lack of data, we tried to augment data using interpolation. However it concluded in failure because newly augmented data has too similiar to the original that the model is overfitted.
3. Becuase of the unique features of drving data, it inevitable to have unsmooth data. It is expected to deteriorate the performance of LSTM and GRU which rely on smoothness of the data.


### Future Works
1. Gather more data from more drivers.
2. Develop another model which utilizes not only numeric sensor data but also image sensor data.
3. Improve accuracy of classification and accelerate the code in order to provide the driver with recommendations in real time.



## Code Description
### implementSVM
Modules : pandas, matplolib, numpy, sklearn, boruta

Functions 
1. Generate_data : devide raw data with many laps into data with single lap and extract curve data only
2. Load_data : load generated data
3. Processing_data : normalize data using standard scaler
4. Evaluate_model : evaluate model with average cross-validation score
5. Confusion_matrix : create confusion matrix which contains accuracy, precision, recall and final scores.
6. draw_ROc : create plot of ROC curve
7. grid_searching : grid search in order to find adequate hyper-parameters using auto tune.
8. run_experiment : Main code
9. finding_hyperparameter : get proper hyper-parameters

### implementLSTMandGRU
Modules : pandas, matplolib, numpy, sklearn, boruta
### implementGRU_torch

