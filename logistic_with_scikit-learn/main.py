import pandas as pd
import numpy as np
from data_preprocessing import data_filling, numerical_converting
from logistic_regression import loan_status_train, random_loan_status_predict

if __name__ == '__main__':
    dataset = pd.read_csv("LoanApprovalPrediction.csv")
    dataset = dataset.drop('Loan_ID', axis=1)
    
    # data processing
    dataset = data_filling(dataset)

    X = dataset.drop('Loan_Status', axis=1)
    Y = dataset['Loan_Status']

    # predict loan status from random input

    # generate random value for X with Dataframe type
    # generate 11 random value for each column of X_test with binary value
    X_test = pd.DataFrame(np.random.randint(2, size=(1, 11)), columns=X.columns)

    loan_status = random_loan_status_predict(X_test, X, Y)
    print(X_test)
    print(loan_status)

    # when loan status is 'no' ???? so i made this
    # for i in range(100):
    #     X_test = pd.DataFrame(np.random.randint(2, size=(1, 11)), columns=X.columns)
    #     loan_status = random_loan_status_predict(X_test, X, Y)
    #     if loan_status == 'No':
    #         print(X_test)
    #         print(loan_status)
    #         break
