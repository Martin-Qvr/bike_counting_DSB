# bike_counting_DSB
Academic project

Description of the bikes prediction challenge
This project was carried out during the "Python for Data Science" course of the X-HEC Data Science for Business Master of Science. 

The problem consisted in predicting the number of bikes circulating in Paris in several counters located near cycling lanes. Hence this problem was a regression problem. The original data can be found on the [opendata.paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name) website.

The objective was to build two files : estimator.py containing a scikit-learn pipeline which final step was a predictive model, and external_data.csv, which contained additional data that we had to find ourselves through various sources. They were then sent on the [RAMP](https://ramp.studio/) Platform which we used for the project, which is maintained by the INRIA team.

Considering the fact that the data initially provided was very limited, and after a few attempts to produce predictions with the initial data as only input, we understood how crucial would be the acquisition of external data to increase our RMSE and the overall performance of our models. Hence data acquisition was the first stage of the problem. We focused our attention on weather data.

The second stage of the problem was to find an appropriate Machine Learning model to solve this problem, and even more importantly, to tune the hyperparameters in order to increase its performance. Since the interpretability of the model was actually not a problem in our case, and since the volume of data was not important enough to raise significant issues of computation time, we could use almost any algorithm/machine learning library we saw fit.

Our best model  reached a RMSE of 0.748 on Ramp.

# Data acquisition
The data acquired basically lied among non-temporal and temporal data. Along the way, we built several functions to increase our efficiency to preprocess data and automate the integration of data from new sources.

# Model
Setting a performance criterion
We had to define a criterion. The most common and interpretable one is actually the RMSE. Plus, it was simplified as the ramp-test command on the terminal provided a simple way to produce a quick performance analysis of our model. Nonetheless, we used several other methods to increase the performance of our model, especially to tune the hyperparameters of our models (see below).

## Choosing a model: CatBoost
We have tried several machine learning models and compared them using the ramp-test command of the terminal (which is basically a cross-validation).

- Decision trees : These algorithms did not perform well (compared to the following) on the train dataset, and terribly on the test dataset (problems of overfitting, structural in Decision Trees models).
- LASSO : More powerful than a linear regression since we add a L1 penalty supposed to trade a portion of bias to lower to variance. This did not perform poorly but was not sufficient to compare with the following. This model was nonetheless very economic in terms of computations.
- Random Forests : More powerful than the previous models, and performs better on the test dataset (lower the risk of overfitting compared to Decision Trees).
- Boosted Random Forests (xgboost library) : This was one of the best models we found in terms of RMSE, even with a poor initial choice of hyperparameters, and performed particularly well on the test dataset compared to the other methods. Yet, in order to make it usable in our mixed-data case (both categorical and numerical), we had to encode the categorical features. Plus, this model was very sensitive to the hyperparameters set, and the computations to find optimal ones were costly.

- CatBoost : This model is also a boosted random forest, however it handles the categorical features correctly . Our RMSE increased significantly when we switched to this model, and we managed to optimize it correctly.

## Choosing the features
We mostly used the feature importance attribute returned by the CatBoost model in order to choose the right features : the most significant data was composed of the baseline frequentation for each flight, of the date of departure and of the 3 stock index variables. We tried to remove population departure and holiday, however as we lost in RMSE we figured out that those two variables were still significant.

Here is an output showing the most significant features :


Tuning the hyperparameters
In order to test the hyperparameters, we decided to start with random parameters and to use a GridSearch to tune them. We evaluated the following parameters for the xgboost model. However, we also did it  for the Catboost modell. 

Learning rate and max depth : they control the propensity of the model to overfit. The learning rate is a penalisation used to control the train/test error ratio, while the max depth can be used to stop a tree from making too many decisions on weak predictors;
N estimators and colsample by tree : they control the speed of convergence of the algorithm.


RAMP detailled informations
More informations about RAMP can be obtained on the page of the Airplanes project template here.
