## Proposal
[![Proposal Video](https://img.youtube.com/vi/dMVrwXonqz8/hqdefault.jpg)](https://www.youtube.com/watch?v=dMVrwXonqz8)

### Intro / Background Info
When purchasing a used vehicle, there are many factors one must take into consideration. The value of a car must be determined by much more than the make and model, making price evaluation a critical part of the process. A machine learning model can be developed to optimize the selection process for the benefit of consumers.
There are some literature review examples on car evaluation using machine learning that can shed light on the topic. For example, the comparative analysis done by Chen, Hao, and Xu which compared different models using random forest (RF) and linear regression (LR) methods [1]. They performed a 5-fold cross validation‒4 training sets and 1 testing‒and compared them based on a normalized mean square error. They concluded that for models using different features, LR and RF performance varied and used NMSE to rank model accuracy. Another example is the work of Maddali, who observed California car prices [2]. Tested many regression models including Linear Regression, Decision Tree Regressor, Gradient Boosting, and MLP Regressor. They concluded that decision tree regressor was the best suited option based on the Mean Absolute Error of each model.

### Data Set
This dataset has over 3 million data points with a high fill-rate of important features. This prevents the problem of a lack of data from being encountered when training/testing the model. However there are very few outliers that must be filtered out due to incorrect prices.

[3] https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset

### Problem Definition
In 2021, a global chip shortage caused a major surge in prices for used cars. As a result, car manufacturers were unable to produce enough vehicles as demanded. Since then, it has been increasingly difficult to distinguish whether car listings are too expensive or a good deal. With so much data available, a model can simply extrapolate a car’s value using its features such as registered accidents and body type.

### Methods used
Different supervised machine learning algorithms will be deployed to model the system. Three examples are Neural Network (ReLU Activation), Decision Tree (MARS Algorithm), and Bayesian Linear Regression models. The best approach will be determined based on results and error. An open source library such as Tensorflow will facilitate the implementation of these models.

### Potential Results and Discussion
This model is expected to evaluate the value of a used car by means of an Evaluator Score Method or Scoring Parameters. A default evaluation criteria would allow the model to rank used cars based on their features, such as the price at which it was sold. On the other hand, using a cross-validation approach would implement an internal scoring strategy in which a parameter controls a certain metric that applies to the estimators evaluated.
Model accuracy and performance will be evaluated to justify changes [4]. To do so, error must be analyzed at each epoch to ensure it is decreasing. As changes to the model are implemented, our model’s prices for used cars must be approaching listed prices, and avoid overfitting data.


### References
https://scikit-learn.org/stable/modules/model_evaluation.html
https://aip.scitation.org/doi/abs/10.1063/1.4982530
https://medium.com/odscjournal/predicting-car-prices-using-machine-learning-and-data-science-52ed44abab1b
https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset 
https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

[] “3.3. metrics and scoring: Quantifying the quality of predictions,” scikit. [Online]. Available: https://scikit-learn.org/stable/modules/model_evaluation.html. [Accessed: 20-Feb-2023]. 

[1] C. Chen, L. Hao, and C. Xu, “Comparative analysis of used car price evaluation models,” AIP Publishing, 08-May-2017. [Online]. Available: https://aip.scitation.org/doi/abs/10.1063/1.4982530. [Accessed: 20-Feb-2023]. 

[2] S. Maddali, “Predicting car prices using machine learning and Data Science,” Medium, 24-Aug-2022. [Online]. Available: https://medium.com/odscjournal/predicting-car-prices-using-machine-learning-and-data-science-52ed44abab1b. [Accessed: 20-Feb-2023]. 

[3] AnanayMital, “US used cars dataset,” Kaggle, 21-Sep-2020. [Online]. Available: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset. [Accessed: 22-Feb-2023]. 
[4] A. Bajaj, “Performance metrics in machine learning,” neptune.ai, 17-Feb-2023. [Online]. Available: https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide. [Accessed: 22-Feb-2023].

### Gantt Chart

### Contribution Table 
Member | Responsibilities for Proposal
--- | ---
Ethan | Worked on the Intro and Background info section and came up with the methods and algorithms we plan to use
Matheus | Organizing team meetings, creating the Gantt Chart, creating the contribution table, help on brainstorming the ideas, and created the github page
Cole | Worked with Ethan on the Intro and Background info section and in the problem definition
Donaven | Worked on potential results and discussion section and responsible for the video
Fernando | Worked on potential results and discussion section and responsible for the video