# Random Forest Regression

## Definition

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.

It is used whenever we need to fulfil following goals :

  - Random forests can be used to rank the importance of variables in a regression or classification problem in a natural way.
  - Measuring variable importance through permutation.

Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance

## Our Use Case

Here we are using a dataset containing detail about salary demanded by people according to their work experience in years.
We try to find the relations between these variables by performing Simple linear regression to predict whether a new employee with some years of experience is asking
for genuine salary or not.

## Test Our Model

You can test our model by going to our Google Collab session and upload attached dataset in that session and run all cells to visualize the results.
Link to collab : [https://colab.research.google.com/drive/1Q6JZtZMtAs5MifVYkkDXogXaItUKnr4y?usp=sharing](https://colab.research.google.com/drive/1Q6JZtZMtAs5MifVYkkDXogXaItUKnr4y?usp=sharing)

## Results 

On running the Model you will see simillar results as shown below with our given dataset. If you wish to upload your own dataset you can rename and restructure that dataset as ours and upload it to session to view results.

### Test Result

<p align="center"><img src="/docs/img/random_forest.png" alt="slr"></p>

## Contribute

You can directly create a PR through which you can tell us your intrest and contributing area.