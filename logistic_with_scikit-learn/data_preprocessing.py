from sklearn.preprocessing import LabelEncoder


def numerical_converting(dataset):
    """
    Convert data from string to integer/float
    """
    le = LabelEncoder()
    for i in dataset.columns:
        if dataset[i].dtypes == 'object':
            dataset[i] = le.fit_transform(dataset[i])
        
    return dataset

def data_filling(dataset):
    """
    Fill missing data with int/float of a dataset
    """
    for i in dataset.columns:
        if dataset[i].dtypes == 'object':
            numerical_converting(dataset)
        
    
    for i in dataset.columns:
        dataset[i].fillna(dataset[i].mean(), inplace=True)
    
    return dataset