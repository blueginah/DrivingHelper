
<!-- ABOUT THE PROJECT -->
# Development of AI model for Racing Driving Level Classification

This is a project to classify high-skilled drivers and low-skilled drivers using data extracted from a steam game named Assetto-Corsa

<!-- GETTING STARTED -->
# Getting Started

## Prerequisites and Usage

1. Clone the repo
   ```sh
   git clone https://github.com/blueginah/DrivingHelper.git
   ```
2. Install Requirements (not needed when using Google Colab)
   ```sh
   pip install -r requirements.txt
   ```
Since this project was implemented using Google Colab, clone this repositiry and you can simply execute cells of each file to see the results. 

<!-- USAGE EXAMPLES -->
# Problem Formulation and Motivation
## Motivation and Societal Applications
1. Racing is not familiar for most people and since it is hard to get trained, we classified experts and beginners to help racers' education so that racers can be trained in more detail according to their driving abilities.
2. We can embed this system to ADAS(Advanced Driver Assistance System) and this will provide different supports according to the driver's ability. Therefore, the overall safety of drivers as well as road safety will be guaranteed.
3. We can customize and recommend automobile systems that match the unique characteristics of each driver.

Therefore, we decided to classify drivers into two groups, experts and beginners.

## Data Collection
![image](https://user-images.githubusercontent.com/30046101/122674317-8608d400-d20f-11eb-9a72-b439d7736936.png)

We collected data by playing a steam game, named "Asseto Corsa" using Logitech G27 Steering Wheel, Accelerator, and Brake(no clutch) and converted data into csv files.

![image](https://user-images.githubusercontent.com/30046101/122668614-e12ccd80-d1f3-11eb-9a4d-5e8f90abd869.png)

![image](https://user-images.githubusercontent.com/30046101/122674342-ac2e7400-d20f-11eb-80d9-fbbc509a8a72.png)

Driving data included about **180 driving features** (e.g. Time, Distance, GPS Latitude, Damper Velocity, Corr Dist..) and each feature contained about 1200 to 2000 rows each.

-> We preprocessed data by extracting actual realistic sensor features(e.g. Steering Angle, Throttle Pos ..) except game settings (e.g. Tire Loaded Radius FR, Brake Pos..).

# Background 
## Model Selection
![image](https://user-images.githubusercontent.com/30046101/122674791-a2a60b80-d211-11eb-87d8-b5564107f08d.png)

* **SVM (Support Vector Machine)**
[From Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine) 
: SVM is a supervised machine learning algorithm which can be used for both classification or regression challenges. 

* **LSTM (Long Short Term Memory networks)**
[From Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory) 
: LSTM is a special kind of RNN, which is capable of learning long-term dependencies and effective for handling time series data. 

* **GRU (Gated Recurrent Unit)**
[From Wikipedia](https://en.wikipedia.org/wiki/Gated_recurrent_unit) 
: GRU is a variation on the LSTM and aims to solve the vanishing gradient problem with a standard RNN. GRU is less complex and more optimized compared to LSTM.


# Implementation
Classifying high-skilled drivers and low-skilled drivers using data extracted from Assetto-Corsa

![image](https://user-images.githubusercontent.com/30046101/122668614-e12ccd80-d1f3-11eb-9a4d-5e8f90abd869.png)

## Description for each Files
<img src="https://user-images.githubusercontent.com/30046101/122676033-0a128a00-d217-11eb-8ec7-268d4e20d78b.png" width="30%" height="30%" align="center" />
<br>

The above tree shows the overall flow of our project. 

There are three sub directories named "beginner_expert_RawData", "beginner_expert_processedData", "cornerData" which includes beginners and experts data csv files. (csv files are not showed in this figure since there are too many) 
Other ".ipynb" files are implementation of SVM, LSTM, GRU and preliminary experiments.

## 1. Directories Description

### * beginner_expert_RawData

    This directory includes two subdirectories named "beginner" and "expert", which includes 35 csv files of beginners' driving data and 19 experts csv files of exeperts' driving data.

### * beginner_expert_processedData

    This directory is made from RawData, which also includes beginners and experts data but modified for ease of implementation. Also, added some new features such as CGDistance, Longitudinal, Lateral Acceleration, Longitudinal Acceleration by calculating in person so that we can classify people more easily.

### * cornerData

    Lap data of beginners and experts are split according to each curve number, using absolute value of distance, and combined to each csv file depending on curve number.

## 2. Description of Files
### * target.csv and group.csv
    Target.csv includes label of each data whether it is beginner's or expert's. 
    Group.csv includes dataset ID, which is 1 to 3, showing whether the specific number of file is train, validation, or test data.
### * dataplotting.ipynb

    This file is for preliminary experiment, to plot experts and beginners data according to different features so that we can compare flow of entire data.

    1) From the processedData directory, convert each csv files into dataframe and append data to df_begin, df_exp each. 

    2) Select some features that we want to plot in featureList and then plot them using matplotlib.

    3) Using curve_num variable, we can select mode whether to specify a certain curve number or plotting the entire curve data.

### * implementGRU-torch_new.ipynb

    This file includes GRU model implementation and experiments.

    1) Getting beginners' and experts' data from processedData directory, converting csv files into dataframe and extracting curve data only and store them in cornerData directory

    2) Initializing hyperparameters such as learning_rate, number of layers, and number of epochs etc.

    3) Concat expert data and beginners data into df_tmp_begin and df_tmp_exp each.
    
    4) Scaling data using MinMaxScaler and randomize sequence using np.random so that each experiment can have different data and truncate each data in 60 rows.
    
    5) Initialize GRU class and using CrossEntropyLoss loss function and Adam Optimizer.
    
    6) Grid Search using optuna module for hyperparameter tuning.
    
    7) Fit and train the GRU model and evaluate using accuracy score.

### * implementLSTMandGRU.ipynb

    This file includes LSTM and GRU model implementation and experiments.

    1) generate_allcorner_data():
    Getting experts and beginners data from processedData directory, and after converting them into dataframe format, scaled them using MinMaxScaler and appended them to sequences list, which is a unit of LSTM and GRU's sequences. 
    Then, padding them according to the last row of max length data and truncate their length into 60. Then put them into train and test dataset.

    2) generate_onecorner_sequences():
    Process data according to each corners, same as generate_allcorner_data()

    3) load_data:
    Split data into trainset, validationset and targetset.
    
    4) model_making:
    Making LSTM and GRU model using keras by adding LSTM and Dropout layers, using sigmoid activation function.
    
    5) Learning:
    Using ADAM optimizer and binary crossentropy loss function, fitting model.
    
    6) loadmodel and evaluate:
    Loading models that are saved in early checkpoints and after compilling them, evaluate using test set. Also, plotting the result. 

### * implementSVM.ipynb

    This file includes SVM model implementation and experiments.

    1) generate_data():
    Read csv file data from processedData directory and add "level" feature to use as a label of classifier and "curve_num" feature to extract data of each corner, which is csv files in cornerData directory.

    2) load_data():
    Loading data generated from generate_data() into X(data to train) and y(label).

    3) processing_data():
    Split loaded data into train, test data and scaling data using StandardScaler.
    
    4) evaluate_model():
    Evaluating model using SVC, which is one of evaluators of Support Vector Machine.
    
    5) confusion_matrix():
    Creating confusion matrix including several evaluation scales including precision score, accuracy score and etc.
    
    6) draw_ROC():
    Drawing ROC curve, which is one of the evaluating metrics of SVM model.
    
    7) grid_searching(X_train, y_train):
    Hyperparameter tuning using grid search and fit the model. 
    
    8) run_experiment(column, corner):
    Running experiment using the above functions, which is the main function.
    
    9) finding_hyperparameter(column, corner):
    Finding the optimal hyperparamters for each train data. 

# Results
## Preliminary Results
We conducted a preliminary experiment to figure out whether data can be well classified according to various features. 
<img src="https://user-images.githubusercontent.com/30046101/122675562-f2d29d00-d214-11eb-916f-f270d1855f8d.png" width="80%" height="80%" align="center" />
<br>

## SVM
![image](https://user-images.githubusercontent.com/30046101/122675632-46dd8180-d215-11eb-8e39-07faf1bb8e12.png)

The table contains average value of Cross-Validation Score. Since all accuracy is above 92%, we can conclude that SVM is a reasonable model which classifies experts and beginners well. Also, according to the ROC curve which is one of the measures to evaluate SVM model, since the blue curve is in the top left, we can see that SVM has classified well. Not only that, as the top two features are the indicator of whether a driver is driving smoothly, the importance of each feature has been expressed well too.

## LSTM
<img src="https://user-images.githubusercontent.com/30046101/122669509-b5601680-d1f8-11eb-9216-2ffa1e3de85e.png" width="50%" height="50%" align="center" />
<br>

The experiment was conducted for every 6 curves, and also the total lane which includes all of the curves including "curvurture" feature. Most of the scores of LSTM showed high accuracy, but some curves including curve 3 and 4 highly were affected to consistency of data.

* Curve Classification with one LSTM layer
![image](https://user-images.githubusercontent.com/30046101/122675753-d7b45d00-d215-11eb-9211-6c06faa90bd0.png)

* Curve Classification with two LSTM layers
![image](https://user-images.githubusercontent.com/30046101/122675758-e1d65b80-d215-11eb-9530-2a43f603499c.png)

Model having two stacks of LSTM layers showed better performance compared to a single layer model. 

## GRU
![image](https://user-images.githubusercontent.com/30046101/122676263-58745880-d218-11eb-8356-d1d741209f8c.png)

GRU was also dependent to the consistency of curve data, which means it showed various performance according to specific curves but showed higher performance than LSTM. 

# Discussion for the Results
## Why did SVM outperform LSTM or GRU?

    1) Since it was Binary Classification with lots of features, it was more advantageous for SVM 

    2) SVM makes decision based on average features but LSTM makes decision based on previous timestamp data

    3) Because of the unique features of driving data, beginners’ driving wasn’t smooth as experts so that  there was less correlation compared to general time series data
    
    4) For same amount of data, LSTM and GRU considers whole time step data as one single data, but SVM considers each as different data so the total amount of data was larger in SVM compared to other models.

# Future Works
1. Gathering more data from people having various driving skills so that our model can classify data into more specific levels

2. Develop other classification models using image and sensor data together

3. Improve Accuracy of Classifications in real time when playing game so that it appropriately recommends directions to users.

<!-- CONTACT -->
# Contact

HwangGiyoung - (https://blueginah97@gmail.com) - email

Project Link: [https://github.com/blueginah/DrivingHelper](https://github.com/blueginah/DrivingHelper)

