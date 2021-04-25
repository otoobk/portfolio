""" This file contains code to solve the Titanic Survival Problem
    To ensure the program works, make sure the datasets, 'train.csv' and
    'test.csv' are in the working directory. As well, may need to install/update
    versions of Pandas, Numpy, and/or Sklearn. Code does not produce testing set label
    results but meerly prints findings on accuracy of classifiers used.

    Author: Brendan O'Toole
    Date: 04/10/21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Use this to hide a warning for slicing data frame in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# Import data
training_set = "train.csv"
testing_set = "test.csv"

training_set = pd.read_csv(training_set)
testing_set = pd.read_csv(testing_set)

# Create copies of data before mutating
original_train = training_set.copy()
original_test = testing_set.copy()

# We can drop data characteristics which are irrelevant to
# a passenger's likelihood of survival
training_set = training_set.drop(['PassengerId'], axis=1)
testing_set = testing_set.drop(['PassengerId'], axis=1)

training_set = training_set.drop(['Name'], axis=1)
testing_set = testing_set.drop(['Name'], axis=1)

training_set = training_set.drop(['Ticket'], axis=1)
testing_set = testing_set.drop(['Ticket'], axis=1)

# We also want to simplify the Cabin data points by changing it to
# only represent the level at which the passenger's room was (ie Deck)

# To do so we must change to Cabin object to an int that represents Deck level
data = [training_set, testing_set]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].astype(str)
    for i in range(len(dataset['Deck'])):
        dataset['Deck'][i] = ord(dataset['Deck'][i][0])

training_set = training_set.drop(['Cabin'], axis=1)
testing_set = testing_set.drop(['Cabin'], axis=1)

# Now we must fill in values which are currently null, first we can check
# Which columns have missing values
total = training_set.isnull().sum().sort_values(ascending=False)

# To account for the large amounts of missing age, we can predict an age according
# to the average age of all passangers
# For simplicity, we will predict all missing ages as the average age
avg_age = training_set['Age'].mean()

data = [training_set, testing_set]

for dataset in data:
    for i in range(len(dataset['Age'])):
        if (np.isnan(dataset['Age'][i])):
            dataset['Age'][i] = avg_age.astype(int)

# We can replace the missing embarked values with the most common values accross the set
training_set['Embarked'].mode()
embarked_mode = 'S'

data = [training_set, testing_set]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)

    for i in range(len(dataset['Embarked'])):
        dataset['Embarked'][i] = ord(dataset['Embarked'][i][0])

# We must replace any missing fares, we will do so with the average fare
fare_mean = dataset['Fare'].mean()

data = [training_set, testing_set]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(fare_mean)

# Now we must attempt to convert all data to ints
data = [training_set, testing_set]

for dataset in data:
    for i in range(len(dataset['Sex'])):
        if (dataset['Sex'][i] == 'male'):
            dataset['Sex'][i] = 1
        else:
            dataset['Sex'][i] = 0

# We can create our final training and testing sets

X_train = training_set.drop("Survived", axis=1)
Y_train = training_set["Survived"]
X_test  = testing_set

# We can use a Naives Bayes Classifier to predict survival
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print("The accuracy of the Naives Bayes Classifier:")
print(acc_gaussian)
print("")

# We can use a K-Nearest Neighbors Classifier to predict survival
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print("The accuracy of the K-Nearest Neigherbors Classifier:")
print(acc_knn)
print("")

# We can use a Random Forest Classifier to predict survival
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print("The accuracy of the Random Forest Classifier:")
print(acc_random_forest)
print("")

# With the Random Forest Classifier, we can also view which data points had the most signifigance
# when making predictions
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances.head(15))

classifiers = ('Naives Bayes', 'K-Nearest Neigherbors', 'Random Forest')
y_pos = np.arange(len(classifiers))
accuracy = [acc_gaussian, acc_knn, acc_random_forest]

plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, classifiers)
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy')

plt.show()
