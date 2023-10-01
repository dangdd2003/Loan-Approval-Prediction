import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def loan_status_predict(attributes, loan_status):
    """
    Return new predict value for Loan and model
    """
    X_train, X_test, Y_train, Y_test = train_test_split(attributes, loan_status, test_size=0.4, random_state=10)

    lr1 = LogisticRegression()
    lr2 = LogisticRegression()
    lr1.fit(X_train, Y_train)
    lr2.fit(X_train, Y_train)
    Y_train_pred = lr1.predict(X_train)
    Y_test_pred = lr2.predict(X_test)

    if accuracy_score(Y_train, Y_train_pred) > accuracy_score(Y_test, Y_test_pred):
        return Y_train_pred, lr1
    else:
        return Y_test_pred, lr2


def random_loan_status_predict(X, attributes, loan_status):
    """
    Return predict loan status for random input\n
    "X" must be list, array or Dataframe
    """

    # if X is not Dataframe, then convert it to Dataframe
    if type(X) != type(pd.DataFrame()):
        X = pd.DataFrame(X, columns=attributes.columns)


    _, model = loan_status_predict(attributes, loan_status)
    y_pred = model.predict(X)
    if y_pred[0] == 1:
        return 'Yes'
    else:
        return 'No'
    