# Working with Time-Series data: Predicting West Nile Virus Incidents In Chicago, by Yap Jun Hong (Nemo), Dennis Chan Zhen Ye, and Adrian Teng

## Section 1: Problem statement

### Part 1.1: The context of the problem

West Nile Virus (WNV) commonly spreads to humans through infected mosquitos. WNV is dangerous because around 20% of infected persons develop symptoms that range from a **persistent fever** to **serious neurological illnesses that can result in death**. In 2002, WNV had spread to the City of Chicago, and the Chicago Department of Public Health (CDPH) had established a comprehensive surveillance and control program that was still in effect as of 2015.

CDPH's methodology is as follows: 

1. The number of mosquitos present in an area can be controlled by airborne pesticides. 

2. To determine whether or not to spray presticide to control WNV, one can count the number of mosquitos with WNV. 

3. To achieve this goal, a city can set up traps across an area to catch mosquitos. 

4. These mosquitos are then tested for WNV. When a certain threshold is crossed, airborne pesticides are released.

Traps have been put up annually, from late spring to the end of fall (autumn).

CDPH has been collecting weather, location, testing, and spraying data across 2007 and 2014 alongside the number of mosquitos with WNV. They have set up this project in the hopes of finding a more accurate way of predicting outbreaks of WNV in mosquitos for more efficient allocation of resources.

### Part 1.2: The goal of this project

The goal of this project is to **predict when and where different speices of mosquitos will test positive for WNV**. The data used will be the aforementioned weather, location, testing, and spraying data.

The training data consists of data from 2007, 2009, 2011, and 2013. The test sets requiring prediction are the test results for 2008, 2010, 2012, and 2014.

### Part 1.3: Evaluation metrics

This project is for a Kaggle competition, linked here:"[West Nile Virus Prediction: Predict West Nile virus in mosquitos across the city of Chicago](https://www.kaggle.com/c/predict-west-nile-virus/overview)."

From the evaluation page: "Submissions are evaluated on area _**under the ROC curve**_ between the predicted probability that WNV is present and the observed outcomes."

Each record in the evaluation file will be a real-valued probability (0-1) that WNV is present.

## Section 2: Notebooks in this project

1. [Exploratory Data Analysis](code/01_weather_and_train_data_analysis.ipynb)
2. [Modelling and Prediction](code/02_Modelling_and_prediction.ipynb)

## Section 3: Datasets and data dictionaries

The dataset is split into 5 files:

1. spray.csv
2. weather.csv
3. mapdata_copyright_openstreetmap_contributors.rds and mapdata_copyright_openstreetmap_contributors.txt (aka map data)
4. train.csv
5. test.csv

This readme will first discuss spray, weather, and map data, before discussing the train and test tests.

**spray.csv**

The City of Chicago does spraying to kill mosquitos. spray.csv contains the GIS data for spray efforts in 2011 and 2013 (i.e. limited data). Spraying reduces the number of mosquitos in the area and might eliminate the appearance of WNV.

spray.csv consists of 4 columns,

- Date
- Time
- Latitude
- Longitude

**weather.csv**

Hot and dry conditions are more favourable for WNV than cold and wet areas. This csv was drawn from the [National Oceanic and Atmospheric Administration (NOAA)](https://www.noaa.gov/) and comprises the weather conditions of 2007 to 2014, during the months of the tests. The data was drawn from two stations.

Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level

|**Index**|**Feature Name**|**Description**|
|---|---|---|
|1|Station|The station that measured that row of data|
|2|Date|Date in DD/M/YYYY|
|---|---|---|
|3|Tmax|Max temperature in Fahrenheit|
|4|Tmin|Minimum temperature in Fahrenheit|
|5|Tavg|Average temperature in Fahrenheit|
|6|Depart|Departure from normal temperature. This is not recorded by station 2|
|7|DewPoint|Average dew point|
|8|WetBulb|Average wet bulb|
|---|---|---|
|---|---|Degree Days: Base of 65 F|
|9|Heat|Heating (Season begins with July), quantifies the demand for energy needed to heat a building|
|10|Cool|Cooling (Seasson begins with January), quantifies demand for air conditioning.|
|11|Sunrise|Calculated sunrise|
|12|Sunset|Calculated sunset|
|---|---|---|
|13|CodeSum|Code of weather phenomena, such as Haze (HZ) or Mist (BR)|
|14|Depth|Depth of snow/ice on ground|
|15|Water1|Depth of water on ground|
|16|SnowFall|Snowfall in inches|
|17|PrecipTotal|Total amount of rainfall (precipitation)|
|---|---|---|
|18|StnPressure|Average station pressure|
|19|SeaLevel|Average sea level pressure|
|---|---|---|
|20|ResultSpeed|Resultant wind speed in miles per hour|
|21|ResultDir|Resultant wind direction in tens of degrees (whole degrees)|
|22|AvgSpeed|Average wind speed in miles per hour|


**map.csv**

The two files, mapdata_copyright_openstreetmap_contributors.rds and mapdata_copyright_openstreetmap_contributors.txt, are from Open Streetmap and are primarily provided for use in visualisations. However, teams have the option of using this data in the models if they so wish.

**train.csv**

This is the training set of the main dataset, consisting of data from 2007, 2009, 2011, and 2013.

It consists of the following columns

|Index|Feature Name|Description|
|---|---|---|
|1|Id|The id of the record|
|2|Date|Date that the WNV test is performed|
|3|Address|Approximate address of the location of the trap. This isused to send to the GeoCoder|
|4|Species|The speices of mosquitos|
|5|Block|Block number of address|
|---|---|---|
|6|Street|Street name|
|7|Trap|Id of the trap|
|8|AddressNumberAndStreet|Approximate address returned from GeoCoder|
|9|Latitude|The Latitude returned from Geocoder|
|10|Longitude|The Longitude returned from Geocoder|
|---|---|---|
|11|AddressAccuracy|Accuracy returned from GeoCoder|
|12|NumMosquitos|Number of mosquitoes caught in that trap|
|13|WnvPresent|Whether WNV was present in these mosquitos. 1 means WNV is present, and 0 means not present. **This is the label**|

**test.csv**

This is the test set of the main dataset, consisting of data from 2008, 2010, 2012, and 2014.

It has all the features from the train.csv **except NumMosquitos and WnvPresent**. 

## Section 4: Production Model and Analysis

### Part 4.1: The final production model

The final production model was an Adaboost model. The model had achieved a score of 73% on the training set and 72% on the test set, meaning that it had some bias, but quite a low variance. This means that it generalised well. This is compared to some other models we tested, including: 

- Logistic Regression
- K-Nearest neighbors
- Random Forest
- Support Vector Machine
- Adaboost (final production model)

The other models obtained really high training scores of more than 90%, but scored around 60-70% on the test set. This means that these models had low bias, but really high variance. This is one reason why we chose the higher biased Adaboost model, because of the generalisability.

### Part 4.2: Kaggle scoring

The kaggle scoring system required us to submit probabilities instead of the classes. i.e. scores of 0.9 and 0.4 (probabilities) were required, rather than the binary 0 and 1 (classifications).

We obtained a Kaggle score of 64%. However, the score was not very meaningful because all of the probabilities predicted by the model was around 0.1 and 0.2. In other words, the final predictions were all 0. So the model guessed the majority class with different probabilities.

### Part 4.3: Why we obtained the score

Adaboost works by combining the outputs of many weak classifiers, in this case decision tree classifiers, and weights the values that were wrong more. Adaboost then runs the classification again with the modified weights on the outputs. The model then converges into a strong learner.

Adaboost is better in situations where there is overfitting by other classifiers. Since the other models had low bias and high variance, it was very easy to overfit on the training data in this case. This is why adaboost generalised the best.

As to why it guessed a low probability of 0.1 and 0.2 each time, it may have been because of the class imbalance in the dataset, where a majority of mosquitos did not have WNV present. The model probably noticed this and simply learned that most mosquitos had a low probability of having WNV.

### Part 4.4: How we could improve in the future

Due to a variety of factors, we did not explore further methods for improving the score. In the future, we could:

1. Do some feature engineering. This includes possibly combining longtitude and latitude into one feature and removing some features so that the model focuses on only a few independent features. 

2. Check and use spatial correlation between the spray, map, and weather data.

## Section 5: Cost-Benefit Analysis

If we want to spray the entire area to get rid of as many mosquitos as possible, it will bring us $2777496 worth of benefits. However, the costs will be around $311404, resulting in more costs than benefits. The city should focus on certain key hotspots to remove the virus.

Another reason why the costs outweighed the benefits was because we calculated the results using home-use pesticides, rather than industrial standard pesticide. Purchasing industrial standard pesticide would be much cheaper overall.

Finally, the benefits of spraying could have been understated.