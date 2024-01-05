from IPython.display import display,HTML
from sklearn.feature_selection import SelectKBest,mutual_info_classif

def inspect_df_na(df, na_row=10):
    df_na_count = df.isna().sum()
    df_na_countAll = df.isna().sum().sum()
    df_na_rowNum = len(df[df.isna().any(axis=1)].index)
    df_na_rowValues = df[df.isna().any(axis=1)]
    data_quality = (1 - (df_na_countAll / df.size)) * 100
    data_quality_byrow = (1 - (df_na_rowNum / df.shape[0])) * 100

    print(f"Null values count by columns : \n")
    [print(i) for i in df_na_count.items()]
    print("--------------------------------------------\n")
    print(f"Total number of null values : {df_na_countAll}")
    print("--------------------------------------------\n")
    print(f'The number of row with null values {df_na_rowNum}')
    print("--------------------------------------------\n")
    display(HTML(df_na_rowValues.head(na_row).to_html()))
    return data_quality, data_quality_byrow

# feature selection
def select_features(X_train, y_train, X_test, k='all'):
    # configure to select a subset of features
    fs = SelectKBest(score_func=mutual_info_classif, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs