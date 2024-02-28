# Machine Learning. Building a regression model for price of diamonds

import pandas as pd
import sklearn 
from sklearn import svm, preprocessing

df = pd.read_csv("datasets/diamonds.csv", index_col=0)


df["cut"].unique()

cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL": 11}
color_dict = {"J": 1,"I": 2,"H": 3,"G": 4,"F": 5,"E": 6,"D": 7}

df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
# print(df.head())


# It is important to reshuffle the data. This is because as you process the data through the model the model will become more bias.
# Check the data isnt already sorted in a particular way. 
# use sklearn to reshuffle data

df = sklearn.utils.shuffle(df)

x = df.drop("price", axis=1).values
# Scaling data = simplify your model. bringing the range of values to a smaller scale
x = preprocessing.scale(x)

y = df['price'].values 

test_size = 200

# The reason for splitting the test samples and the training samples is to see how accurate the model is.
# This avoids the model being trained on the same data it is being tested on. Therefore, giving it an advantage and skewing the accuracy of the model.
# Similar to being tested on the same question you have practiced in a math test. 

x_train = x[:-test_size]
y_train = y[:-test_size]

x_test = x[-test_size:]
y_test = y[-test_size:]

# clf = svm.SVR(kernel="linear")
# print("This may take some time, please be patient...")
# print(clf.fit(x_train, y_train))

# print(clf.score(x_test, y_test)) # 0 is bad 1 is great

# for x, y in zip(x_test, y_test):
#     print(f"Model: {clf.predict([x])[0]}, Actual: {y}")


clf = svm.SVR(kernel="rbf")
print("This may take some time, please be patient...")
print(clf.fit(x_train, y_train))

print(clf.score(x_test, y_test)) # 0 is bad 1 is great. using R-Squared (0-1 rating) 1 = perfrect fit

for x, y in zip(x_test, y_test):
    print(f"Model: {clf.predict([x])[0]}, Actual: {y}")
