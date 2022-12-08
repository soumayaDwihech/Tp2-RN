# Question1
import pandas
DataFrame = pandas.read_csv("Iris.csv",header=0)
print(DataFrame) 



# Question2
print(DataFrame.head(10))



# Question3
print(DataFrame.shape)




#Question4
import seaborn as sns 
import matplotlib.pyplot as plt
DataFrame = sns.load_dataset('iris')
DataFrame.head()
sns.scatterplot(x='sepal_width', y ='petal_length' ,
data = DataFrame , hue = 'species')
plt.show()






#Question5
DataFrame.loc[DataFrame["Species"] == "Iris-setosa" , "Species"] = 0
DataFrame.loc[DataFrame["Species"] == "Iris-versicolor" , "Species"] = 1
DataFrame.loc[DataFrame["Species"] == "Iris-virginica" , "Species"] = 2



#Question6
print(DataFrame.head(10))




#Question 7
from sklearn.model_selection import train_test_split
X_data = DataFrame.iloc[:, 1:5].values
Y_data = DataFrame.iloc[:, 5].values
X_train,X_test,Y_train,Y_test = train_test_split( X_data, Y_data, train_size=0.7)



#Question 8
print("\n 10 premières données d’apprentissage et celles de test (X):\n------------")
print(X_train[0:10], '\n\n', X_test[0:10])
print("\n 10 premières données d’apprentissage et celles de test (Y):\n------------")
print(Y_train[0:10], '\n\n', Y_test[0:10])




#Question 9
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), 
                           epsilon=0.07, max_iter=150).fit(X_train.astype('int'), Y_train.astype('int'))
predModel=classifier.predict(X_test.astype('int'))
print(predModel)






#Question 10
from sklearn import metrics 
print('The accuracy of the Multi-layer Perceptron is:', metrics.accuracy_score(Y_test.astype('int'),predModel))







#Question 11
from pretty_confusion_matrix import pp_matrix_from_data
cmap= "PuRd"
pp_matrix_from_data(Y_test.astype('int'),prediction)





#Question12 : sur rapport




#Question 13
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), 
                           epsilon=0.07, max_iter=150,learning_rate_init=0.7).fit(X_train.astype('int'), 
                                                                                  Y_train.astype('int'))
predModel=classifier.predict(X_test.astype('int'))
print(predModel)







  
#Question 14
loss_values=[]
listLR=[]
Trainloss_values=[]
for lR in range(1,101):
  classifier=MLPClassifier(hidden_layer_sizes=(3, 3),solver="lbfgs",epsilon=0.07,max_iter=150,
                           learning_rate_init=lR/100).fit(X_train.astype('int'), Y_train.astype('int'))
  #evolution of learning
  TrainpredModel=classifier.predict(X_train.astype('int'))
  Trainloss_values.append(metrics.mean_squared_log_error(Y_train.astype('int'),TrainpredModel))
  #test evolution
  predModel=classifier.predict(X_test.astype('int'))
  loss_values.append(metrics.mean_squared_log_error(Y_test.astype('int'),predModel))
  listLR.append(lR/100)
plt.plot(listLR,Trainloss_values,label="evolution of learning")
plt.plot(listLR,loss_values,label="test evolution")
plt.legend()
plt.title('Evolution according to the variation of the learning rate- % EQM ')
plt.show()

  
  
  
  
  
  
  
  
  
  


#Question 15
classifier=MLPClassifier(hidden_layer_sizes=(3, 3),solver="lbfgs",epsilon=0.07,
                         max_iter=1500).fit(X_train.astype('int'), Y_train.astype('int'))
#fit model on the train dataset
TrainpredModel=classifier.predict(X_train.astype('int'))
print('The accuracy is:', metrics.accuracy_score(Y_train.astype('int'),TrainpredModel))
print(TrainpredModel)

#evaluate on the test dataset
predModel=classifier.predict(X_test.astype('int'))
metrics.accuracy_score(Y_test.astype('int'),predModel)
print(predModel)








#Question 16
from sklearn.neural_network import MLPRegressor
#question5
DataFrame['Species'] = DataFrame['Species'].replace(['Iris-setosa'], '0')
DataFrame['Species'] = DataFrame['Species'].replace(['Iris-versicolor'], '1')
DataFrame['Species'] = DataFrame['Species'].replace(['Iris-virginica'], '2')

columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = DataFrame.loc[:, columns]
y = DataFrame.loc[:, ['Species']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)
model = MLPRegressor().fit(X_train, y_train.values.ravel())
print(model)
prediction= model.predict(X_test)
print( prediction)
print(metrics.r2_score(y_test, prediction))



