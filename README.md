# BootCamp-Mod-17-Credit_Risk_Analysis
Performing analysis on credit risk using machine learning algorithms in Python.


## Overview of Project

### Purpose
The purpose of this analysis is to use Python to predict credit card risk using different supervised machine learning algorithms to then determine which algorithm is best at predicting credit risk when unbalanced classes are used.


## Analysis and Results
### Data
- Credit data was provided.
- The target of the data was the loan status, where each account was classified as either low or high risk.
- We can see that these categories are very unbalanced with many more accounts classified as low risk.
<p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Data_Count.png"width="427" height="112"/></p>

### Algorithms
- To deal with the imbalance in the categories, four different types of algorithms were tested.
  * Oversampling
  * Undersampling
  * Combination (Over and Under) Sampling
  * Ensemble Learners

#### Oversampling
- Oversampling selects more instances from the smaller class to get a new balanced training set.
- Two oversampling algorithms were used.
  * Naive Random Oversampling
  * SMOTE Oversampling

- Naive Random Oversampling
  * The RandomOverSampler function was used to get a balanced dataset.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomOverSampler.png"width="657" height="118"/></p>
  * The logistic regression model was used to predict the credit risk.

- Naive Random Oversampling accuracy
  * Balanced Accuracy Score - 65.34%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomOverSamplier_Accuracy.png"width="505" height="108"/></p>
  * Confusion Matrix - 55 accounts that were actually high risk were labeled by the algorithm as high risk, 32 accounts that were actually high risk were labeled as low risk, 5,570 accounts that were actually low risk were labeled high risk, and 11,548 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomOverSamplier_matrix.png"width="456" height="104"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) or sensitivity of the model is moderate for both high and low risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomOverSamplier_report.png"width="712.5" height="178.5"/></p>
  
- SMOTE Oversampling
  * The SMOTE function was used to get a balanced dataset.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/SMOTE.png"width="501" height="146"/></p>
  * The logistic regression model was used to predict the credit risk.
- SMOTE Oversampling accuracy
  * Balanced Accuracy Score - 65.12%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/SMOTE_Accuracy.png"width="426" height="92"/></p>
  * Confusion Matrix - 56 accounts that were actually high risk were labeled by the algorithm as high risk, 31 accounts that were actually high risk were labeled as low risk, 5,841 accounts that were actually low risk were labeled high risk, and 11,277 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/SMOTE_matrix.png"width="357" height="84"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) of the model is moderate for both high and low risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/SMOTE_report.png"width="718" height="157"/></p>

#### Undersampling
- Undersampling is the opposite of oversampling where fewer instances are selected from the larger class to get a new balanced training set.

- Cluster Centroids
  * The ClusterCentroid function was used to get a balanced dataset.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Undersampling.png"width="735" height="135"/></p>
  * The logistic regression model was used to predict the credit risk.
- Cluster Centroids accuracy
  * Balanced Accuracy Score - 52.93%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Undersampling_Accuracy.png"width="425" height="98"/></p>
  * Confusion Matrix - 53 accounts that were actually high risk were labeled by the algorithm as high risk, 34 accounts that were actually high risk were labeled as low risk, 9,425 accounts that were actually low risk were labeled high risk, and 7,693 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Undersampling_matrix.png"width="362" height="94"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) of the model is moderate for high-risk accounts but low for low-risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Undersampling_report.png"width="714" height="160"/></p>
  
#### Combination (Over and Under) Sampling
- Combination sampling is using oversampling to get a balanced dataset and then using undersampling to clean the data to determine which category each data point belongs to.

- SMOTEENN
  * The SMOTEENN function was used to get a balanced dataset.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Combination.png"width="734" height="145"/></p>
  * The logistic regression model was used to predict the credit risk.
- SMOTEENN accuracy
  * Balanced Accuracy Score - 63.75%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Combination_Accuracy.png"width="427" height="94"/></p>
  * Confusion Matrix - 61 accounts that were actually high risk were labeled by the algorithm as high risk, 26 accounts that were actually high risk were labeled as low risk, 7,294 accounts that were actually low risk were labeled high risk, and 9,824 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Combination_Matrix.png"width="359" height="93"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) of the model is moderate for high-risk accounts but low for low-risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Combination_report.png"width="718" height="160"/></p>

#### Ensemble Learners
- Ensemble learners take multiple models and combine them to try to create a more accurate algorithm.
- Two Ensemble Learners were used.
  * Balanced Random Forest Classifier
  * Easy Ensemble AdaBoost Classifier

- Balanced Random Forest Classifier
  * The RandomOverSampler function was used.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomForest.png"width="644" height="126"/></p>
- Balanced Random Forest Classifier accuracy
  * Balanced Accuracy Score - 78.71%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomForest_Accuracy.png"width="422" height="80"/></p>
  * Confusion Matrix - 58 accounts that were actually high risk were labeled by the algorithm as high risk, 29 accounts that were actually high risk were labeled as low risk, 1,582 accounts that were actually low risk were labeled high risk, and 15,536 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomForest_Matrix.png"width="356" height="87"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) of the model is moderate for high-risk accounts, but good for low-risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/RandomForest_report.png"width="719" height="170"/></p>
- Features most effecting the algorithm
  * Total principle amount
  * Total interest amount
  * Total payment
  * Last payment amount
<p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Features_List.png"width="712" height="615"/></p>
 
- Easy Ensemble AdaBoost Classifier
  * The EasyEnsembleClassifier function was used.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/EasyEnsemble.png"width="597" height="131"/></p>
- Easy Ensemble AdaBoost Classifier accuracy
  * Balanced Accuracy Score - 93.17%
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/EasyEnsemble_Accuracy.png"width="424" height="83"/></p>
  * Confusion Matrix - 93 accounts that were actually high risk were labeled by the algorithm as high risk, 8 accounts that were actually high risk were labeled as low risk, 983 accounts that were actually low risk were labeled high risk, and 16,121 accounts that were actually low risk were labeled low risk.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/EasyEnsemble_Matix.png"width="352" height="94"/></p>
  * Imbalanced Classification Report - Precision (pre) is extremely low for high-risk accounts, however it is 100% accurate for low-risk accounts (which may be just because there are so many more low risk accounts than high risk accounts). Recall (rec) or sensitivity of the model is good for both high and low risk accounts.
  * <p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/EasyEnsemble_report.png"width="718" height="161"/></p>

### Difficulties encountered
- The EasyEnsembleClassfier threw an error.
<p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/EasyEnsemble_Error.png"width="825" height="456"/></p>

- After some googling, I came across a [comment](https://github.com/scikit-learn-contrib/imbalanced-learn/issues/872) that scikit-learn needed to be downgraded because the latest version of sklearn depreciated the attribute n_features_in_. 
<p align="center"><img src="https://github.com/M-Outlaw/BootCamp-Mod-17-Credit_Risk_Analysis/blob/main/Images/Error_fix.png"width="689" height="153"/></p>

- After downgrading the scikit-learn package like it said, the error remained. 
- Therefore, unfortunately I could not get the rest of the code to run because I could not get EasyEnsembleClassifier to fit the data. 
- The values I have in the above analysis were those that were provided with the starter code. However, I did put in the code that I would have run to get those values.

## Summary
### Accuracy
- All of the oversampling and combination sampling algorithms had about the same balanced accuracy score.
  * The two oversampling algorithms had a balanced accuracy score that was less than 0.5% away from each other. Both were 65%.
  * The combination sampling algorithm had a balanced accuracy score of 64%, which was only 1% lower than the oversampling algorithm
- The undersampling algorithm had the smallest balanced accuracy score of 53%.
- The ensemble learners performed better.
  * The Balanced Random Forest Classifier had a balanced accuracy score of 79%
  * The Easy Ensemble AdaBoost Classifier had the best balanced accuracy score of all the algorithms of 93%.
- Easy Ensemble AdaBoost Classifier wins in the balanced accuracy score competition.

### Confusion Matrix
- The oversampling algorithms had about the same confusion matrix.
  * Naive Random Oversampling: 58 true positives and 11,277 true negatives, classifying 11,603 accounts accurately out of the total sample of 17,205.
  * SMOTE Oversampling: 56 true positives and 11,548 true negatives, classifying 11,333 accounts accurately out of the total sample of 17,205.
- The undersampling and the combination sampling algorithms had about the same confusion matrix that were worse than the oversampling algorithms.
  * Undersampling: 53 true positives and 7,693 true negatives, classifying 7,746 accounts accurately out of the total sample of 17,205.
  * Combination Sampling: 61 true positives and 9824 true negatives, classifying 9,885 accounts accurately out of the total sample of 17,205.
- The ensemble Learners had the best confusion matrices.
  * Balanced Random Forest Classifier - 58 true positives and 15,536 true negatives, classifying 15,594 accounts accurately out of the total sample of 17,205.
  * Easy Ensemble AdaBoost Classifier - 93 true positives and 16,121 true negatives, classifying 16,214 accounts accurately out of the total sample of 17,205.
- Easy Ensemble AdaBoost Classifier wins in the confusion matrix competition.

### Imbalanced Classification Report
- The oversampling, undersampling, and combination algorithms had the same precisions values:
  * High risk: 1%
  * Low risk: 100%
- The ensemble learners only had slightly better precision values:
  * Balanced Random Forest Classifier high risk: 4%
  * Balanced Random Forest Classifier low risk: 100%
  * Easy Ensemble AdaBoost Classifier high risk: 9%
  * Easy Ensemble AdaBoost Classifier low risk: 100%

- The oversampling algorithms had about the same sensitivity levels:
  * Naive Random Oversampling high risk: 63%
  * Naive Random Oversampling low risk: 67%
  * SMOTE high risk: 64%
  * SMOTE high risk: 66%
- The undersampling algorithm had the worst sensitivity levels:
  * High risk: 61%
  * Low risk: 45%
- The combination algorithm had sensitivity levels that were slightly higher than the oversampling algorithms:
  * High risk: 70%
  * Low risk: 57%
- The Balanced Random Forest Classifier had a high-risk sensitivity level similar to the oversampling algorithms. However, it had a low-risk sensitivity level that was higher.
  * High risk: 67%
  * Low risk: 91%
- The Easy Ensemble AdaBoost Classifier had high sensitivity levels for both classes of accounts.
  * High risk: 92%
  * Low risk: 94%
- Easy Ensemble AdaBoost Classifier wins in the imbalanced classification report competition.

### Recommendation
- If I had to pick one of these algorithms, I would choose the Easy Ensemble AdaBoost Classifier algorithm.   * It had the best accuracy and sensitivity.
- However, I do have concerns still with this model. 
  * It is okay that precision is low because we care more about sensitivity with this model. We want to make sure we catch all of the high-risk accounts.
  * Both accuracy and precision or over 90%, However, if there are 100 high risk accounts that we are trying to catch, we would miss about 8 accounts which could really negatively impact the institution that is giving the credit.
- I would prefer to see if a different ensemble learner would provide a better algorithm.
