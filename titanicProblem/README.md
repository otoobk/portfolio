# Titanic Survival Data Project
#### By: Brendan O'Toole
***
### Overview
This project is a fairly well-known exercise that utilizes a public data set of Titanic "passengers". The data set contains info about each passenger, ranging from name to age and sex to ticket cost and more. These data points can be used to predict the likelihood of their survival during the sinking of the ship.\
This project serves as an exercise where I had to clean a data set, manipulate the data points to be compatible with different kinds of classifiers, and then apply classifiers through supervised learning and produce conclusions given their results.\
In my approach, I cleaned the data of all data points I found to have a weak link to survival rate. Things like this included the passenger's name and ticket number. I was then left with data points that could have had an impact on a passenger's survival. The data points I chose to focus on consisted of age, sex, the deck at which their room was, how expensive their ticket was, and whether they had other people with them on the ship.\
Once I had declared my approach and cleaned the data of all irrelevant values, I then had to quantify and vectorize the data. In doing so, I took data characteristics such as gender and room number which were previously string or object values, and represented them in an integer form. This allowed me to apply classifiers to the data and predict the survival rate of a test set.\
I chose to use three different classifiers which all produced varying accuracies.\
The Naives Bayes Classifier produced the lowest accuracy, coming in around 77%.\
The next best classifier was a K-Nearest Neighbors Classifier, which had an accuracy of around 85%.\
Lastly, the Random Forest Classifier produced outstanding results, tallying an accuracy of 98% on the testing set.\
As well, in utilizing the Random Forest Classifier, I was able to gain insight on which particular data points had the largest impact on survival rate. The results expressed that age, sex, and the cost of the ticket for the passenger all played a large role in the likelihood of survival for that passenger.

### Technologies
A list of technologies used within the project:
* Python
* Pandas
* Numpy
* Scikit Learn
* Matplotlib
