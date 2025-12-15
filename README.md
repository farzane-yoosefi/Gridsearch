# Gridsearch
>> These notes serve to demonstrate my machine learning knowledge to potential employers while also helping beginners learn complex concepts in simple terms.

## What Is Gridsearch?
### Before learning gridsearch we have to know parameters and hyperparameters 
**Parameter** : Parameters are adjusted during training, it means you don't adjust them .

Here is how parameters are adjusted :
1. **See the data** : Model sees the data
2. **Makes Guess** : It uses its current patrameters to make a guess.
3. **Check the error** : It measures how wrong the guess was
4. **Adjust parameters** : It tweaks the parameters to less be wrong next time.
By repeating this process billions of times , the parameters are slowly adjusted untill the best patterns for an accurate prediction are learned .

**hyperparameter** :
Basically Hyperparameters are settings you choose before training that controls how the model learns.

Here is an overview of how hyperparameters are adjusted : 
1. You **Choose** the hyperparameter values.
2. Training begins using these fixed settings.
3. The model learns its ***parameters* based on data, guided by hyperparameters.
4. You evaluate model performance.
5. You often tune hyperparameters to find the best learning strategy

## `from sklearn.model_selection import GridSearchCV`

If we don't tune hyperparameters properly, we can get a terrible model. But 
how do we find the right ones? Trying all combinations is too slow, and manually, we might make 
errors. Therefore, we need an automatic tool for this task,which is where `gridsearch` becomes practical.
Gridsearch in a machine learnig tool which applies computations to find the best hyperparameters and tune the model in best possible way.
## GriedsearchCV components 

1. **Estimator** : The model that we want to tune
2. **parameter** : This is the heart of gridsearch. A list of parameters and their values.
3. **Scoring metric** : The score by which the accuracy is measured.
4. **cross-validation** : The methode that gridsearch uses to train and validate the data and prevents overfitting.

## Gridsearch implementation
First we import `gridsearchcv` , your estimator , and acoring metric
```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```
In order to implement the gridsearch we have to first define our estimator and based on that we define parameters and their values.
```python
estimator = DecisionTreeClassifier()
params_grid={
    'criterion' : ['gini','entropy'],
    'max_depth':[1,2,3,4],
    'min_samples_split' : [2,5,10,15],
    'min_samples_leaf':[10,15,20,25]
}
```
Now we prepare the data and define feature / target
```python
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
Y = data.target
```

split the data
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size= 0.2 , random_state=38)
```

Now we create the GidsearchCV
>> Gridsearch fits only on the training data.
### When we fit the gridsearch on training data :
1. The training set devides into 5 folds (Number of folds can vary)
    **For each combination:**
    - The model is trained on 4 folds
    - And validated on one fold 
2. The first combination of hyperparameters is selected
3. it is trained and validated on every fold(Each fold becomes the validation set once)
4. The average validation score is stored
5. The moves to the next combination
6. After trying all possible combinations ,the best hyperparameters are given.
7. By default, after finding the best hyperparameters via cross-validation, the final model is retrained once on all training set


Now let's create our `GridsearchCV` and fit the objects.
```python
# Objects are already  created above
gridsearch = GridSearchCV (estimator=estimator,param_grid=params_grid,  scoring='accuracy',n_jobs= -1 , cv=5)

# Now fit the gridsearch on the training set
gridsearch.fit(X_train,Y_train)
```
Output is a table which is the **cross-validation results** printed by GridsearchCV.It shows the performance for each hyperparameter combination.


![cross-validatiom table](https://github.com/farzane-yoosefi/Gridsearch/blob/main/gridsearch1.PNG)



