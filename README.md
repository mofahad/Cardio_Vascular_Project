# Cardio_Vascular_Project

Cardiovascular disease is the leading cause of death throughout the United States, with an estimated 840,768 deaths in 2016. However, through simple lifestyle changes and screening, nearly 200,000 deaths per year could be avoided. In this project, we explore several machine learning approaches to detect the presence of cardiovascular disease using only standard health information. The machine learning techniques include a support vector machine (SVM) ,XGBoost (XGB),KNN, and a random forest (RF) classifier. 


#  DataSet Description
The cardiovascular disease dataset is an open-source dataset found on Kaggle [dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset). The data consists of 70,000 patient records (34,979 presenting with cardiovascular disease and 35,021 not presenting with cardiovascular disease) and contains 11 features (4 demographic, 4 examination, and 3 social history):

- Age (demographic)
- Height (demographic)
- Weight (demographic)
- Gender (demographic)
- Systolic blood pressure (examination)
- Diastolic blood pressure (examination)
- Cholesterol (examination)
- Glucose (examination)
- Smoking (social history)
- Alcohol intake (social history)
- Physical activity (social history)

Some features are numerical, others are assigned categorical codes, and others are binary values. The classes are balanced, but there were more female patients observed than male patients. Further, the continuous-valued features are almost normally distributed; however, most categorical-valued features are skewed towards "normal," as opposed to "high" levels of potentially pathological features.


# TODO

Predict on cardiovascular disease (target) based on original data 
>Improving XGB/RF
