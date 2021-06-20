# Development of AI model for Racing Driving Level Classification

Classifying high-skilled drivers and low-skilled drivers using data extracted from Assetto-Corsa

## Motivaiton & Social Applications

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

The columns are list of features of car condition, and the rows are timeseries data recored every 0.5 seconds. It is possible to control the time step.
There are over 180 features and it looks likely to be able to specify the level of driving skill with them.


## Methods

Write a Description
        What your application does,
        Why you used the technologies you used,
        Some of the challenges you faced and features you hope to implement in the future.
Add a Table of Contents (Optional)        
How to Install Your Project
How to Use Your Project
Include Credits
List the License



Why did you build this project?
What problem does it solve?
What did you learn?
What makes your project stand out? If your project has a lot of features, consider adding a "Features" section and listing them here.
