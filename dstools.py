
#    Frequently used functions

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of existing pandas DataFrame via convertion data types.
    
    Parameters:
        df (pandas DataFrame): DataFrame.
        verbose (bool): Print % of reduction.
    
    Returns:
        Reduced pandas DataFrame.
    """
    
    import numpy as np
    import pandas as pd
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def get_data_from_hadoop(q, username, host, port):
    """Get data from Hadoop cluster to local Python notebook.
    
    Parameters:
        q (str): SQL query.
        username (str): username.
    
    Returns:
        Pandas DataFrame with results of query.
    """
    
    import puretransport
    from pyhive import hive
    import pandas as pd
    
    transport = puretransport.transport_factory(host=host, port=port, username=username, password=username)
    
    hive_con = hive.connect(thrift_transport=transport)
    cursor = hive_con.cursor()
    cursor.execute(q)
    tbl = pd.DataFrame(cursor.fetchall(), columns = [x[0] for x in cursor.description])
    tbl.columns = [i[i.rfind('.')+1:] for i in tbl.columns.tolist()]
    hive_con.close()
    
    return tbl


def plot_feature_importance(model, feature_names, top=50):
    """Plot feature importance for existing model.
    
    Parameters:
        model (model): sklearn or LightGBM model.
        feature_names (list of str): List with names of features.
        top (int): N of features, which will be shown on feature importances plot.
    
    Returns:
        None.
        
    Note:
        Function makes a plot.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if 'lightgbm' in str(type(model)):
        importances = model.feature_importance()
    else:
        importances = model.feature_importances_
        
    top = len(feature_names) if len(feature_names) < top else top
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 0.25 * top))
    sns.barplot(x=imp[:top], y=imp.index[:top])
    plt.show()
    
    
def plot_roc_curve(test, predict, labels, figsize=(10,8)):
    """
    Plot ROC curve for predicted probabilities and calculate ROC-AUC.
    
    Parameters:
        test (list of int): List with target values.
        predict (list of float): List with predicted probabilities.
        labels (list of str): Labels for curves.
        figsize (tuple of int): Size of plot.
        
    Returns:
        None.
        
    Note:
        Function makes a plot.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    if (str(type(test)) != "<class 'list'>") | \
       (str(type(predict)) != "<class 'list'>") | \
       (str(type(labels)) != "<class 'list'>"):
        return 'Error: test, predict or labels is not a list'
    elif (len(test) != len(predict)) & (len(test) != len(labels)):
        return 'Error: lenghs of test, predict and labels mismatch'
    else:
        plt.figure(figsize=figsize)
        plt.title("ROC curve", fontsize=12)
        
        for i in range(len(test)):
            fpr, tpr, _ = roc_curve(test[i], predict[i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='AUC={}'.format(np.round(roc_auc, 4)) + ', ' + labels[i])

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.legend()
        plt.xlabel('False positive rate', fontsize=12)
        plt.ylabel('True positive rate', fontsize=12)
        plt.show()
        
        
def calculate_feature_importance(
    data, target, fillna=True, fill_val=-1, test_size=0.20, stratify_by_target=True, shuffle=True, random_state=1,
    features=None, n_estimators=500, min_samples_leaf=5, max_features = 0.2, n_jobs=3, max_depth=10, oob_score=True,
    n_samples=-1, plot=True, top=30, save_to_file=True, filename='importances.csv', sep=';', decimal=',', 
    encoding='cp1251', permutation=True):
    """
    Calculate permutation importance for given dataset.
    
    Parameters:
        data (pandas DataFrame): Train dataset.
        target (str): Column name with target values.
        fillna (bool): If True - fill NA values by fill_val (by default).
        fill_val (int, float, or str): Value for fill NA values.
        test_size (float): Fraction of data, which will be used as test set for train model.
        stratify_by_target (bool): If True - data will be splitted by target values.
        shuffle (bool): Shuffle data before splitting.
        random_state (int): Random state for model and splitting data.
        features (list of str): List with feature names. If None - all columns except target will be used as features.
        n_estimators (int): n_estimators parameter for RandomForestClassifier.
        min_samples_leaf (int): min_samples_leaf parameter for RandomForestClassifier.
        max_features (int): max_features parameter for RandomForestClassifier.
        n_jobs (int): n_jobs parameter for RandomForestClassifier.
        max_depth (int): max_depth parameter for RandomForestClassifier.
        oob_score (int): oob_score parameter for RandomForestClassifier.
        n_samples (int): n_samples parameter for rfpimp importance.
        plot (bool): Plot graph with feature importances.
        top (int): N of features, which will be shown on feature importances plot.
        save_to_file (bool): Save feature importances to file.
        filename (str): Filename with feature importances.
        sep (str): Set separator for file with feature importances.
        decimal (str): Set decimal delimiter for file with feature importances.
        encoding (str): Set encoding for file with feature importances.
        permutation (bool): If True - permutation importance will be used, otherwise - standard feature importance.
    
    Returns:
        model (sklearn model): sklearn RandomForestClassifier, used for calculation of feature importances.
        importance (pandas DataFrame): DataFrame with feature importances.
    
    Note:
        If plot is True, function makes a plot with feature importances.
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from rfpimp import importances
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if data.isnull().values.any() & fillna:
        data = data.fillna(fill_val)
    if features is None:
        features = list(set(data.columns.tolist()) - {target})
    
    train, test = train_test_split(data, 
                                   test_size=test_size, 
                                   stratify=data[target] if stratify_by_target else None, 
                                   shuffle=shuffle, 
                                   random_state=random_state)
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   n_jobs=n_jobs, 
                                   max_depth=max_depth,
                                   oob_score=oob_score, 
                                   random_state=random_state)
    model.fit(X_train, y_train)
    
    if permutation:
        importance = importances(model, X_test, y_test, n_samples=n_samples)
    else:
        importance = pd.DataFrame({'Importance': model.feature_importances_, 
                                   'Feature': features}).sort_values(by='Importance', ascending=False).set_index('Feature')
    
    if save_to_file:
        importance.to_csv(filename, sep=sep, decimal=decimal, encoding=encoding)
    
    if plot:
        top = len(features) if len(features) < top else top
        plt.figure(figsize=(8, int(0.25 * top)))
        sns.barplot(x=importance.Importance[:top], y=importance.index[:top])
        plt.show()

    return model, importance


def plot_confusion_matrix(y_true, y_pred, prob=True, threshold=0.5, title='Confusion matrix', figsize=(6, 5)):
    """
    Plot confusion matrix for given threshold value.
    
    Parameters:
        y_true (list of int): True labels.
        y_pred (list of int or float): Predicted labels or probabilities.
        prob (bool): True if y_pred is probabilities.
        threshold (float): Threshold value for predicted probabilities. Need for convertion of probabilities to labels.
        title (str): Title of confusion matrix.
        figsize (tuple of int): Size of plot.
        
    Returns:
        None.
        
    Note:
        Function makes a plot.
    """

    import itertools
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    
    if prob:
        y_pred = (y_pred > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, color='black', fontsize=12)
    plt.xticks([0, 1], ['0', '1'], color='black', fontsize=12)
    plt.yticks([0, 1], ['0', '1'], color='black', fontsize=12)
    plt.colorbar()
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=12)
 
    plt.tight_layout()
    plt.ylabel('True', color='black', fontsize=12)
    plt.xlabel('Predict', color='black', fontsize=12)
    plt.show()
    
    if prob:
        print('Threshold: {}'.format(threshold))
    
    print("Precision: {}%".format(round(precision_score(y_true, y_pred) * 100, 1)))
    print("Recall: {}%".format(round(recall_score(y_true, y_pred) * 100, 1)))
    print("AUC: {}%".format(round(roc_auc_score(y_true, y_pred) * 100, 1)))
    
    
def plot_predicted_probability(
    data, label_col='label', pred_col='pred', frac=1.0, title='Predicted probabilities, distributed by label', 
    figsize=(9,8), kde=True):
    """Plot distributions of predicted probabilities, divided by true labels.
    
    Parameters:
        data (pandas DataFrame): Pandas DataFrame with true labels and predicted probabilities.
        label_col (str): Column name with true labels.
        pred_col (str): Column name with predicted probabilities.
        frac (float): Fraction of data, which will be used for plot (for large datasets).
        title (str): Title of plot.
        figsize (tuple of int): Size of plot.
        
    Returns:
        None.
        
    Note:
        Function makes a plot.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    sns.distplot(data[data[label_col] == 0][pred_col], kde=kde, bins=np.linspace(0, 1, 51), 
                 label='{} = 0'.format(label_col))
    sns.distplot(data[data[label_col] == 1][pred_col], kde=kde, bins=np.linspace(0, 1, 51), 
                 label='{} = 1'.format(label_col))
    plt.xlabel('Probability', fontsize=12)
    plt.legend()
    plt.show()
    
    
def plot_precision_recall_curve(test, predict, figsize=(10,8)):
    """Plot Precision-Recall curve for predicted probabilities.
    
    Parameters:
        test (list of int): True labels.
        predict (list of float): Predicted probabilities.
        figsize (tuple of int): Size of plot.
        
    Returns:
        None.
        
    Note:
        Function makes a plot.
    """
    
    from sklearn.metrics import average_precision_score, precision_recall_curve
    import matplotlib.pyplot as plt
    
    average_precision = average_precision_score(test, predict)
    precision, recall, _ = precision_recall_curve(test, predict, pos_label=1)
    plt.figure(figsize=figsize)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision), fontsize=12)
    plt.show()