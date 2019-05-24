
# ==============================================
#    Frequently used functions by Frolov A.S.
# ==============================================

def get_data_from_hadoop(q, username='andrey.frolov'):
    "Get data from Hadoop cluster to local Python notebook"
    
    import puretransport
    from pyhive import hive
    import pandas as pd
    
    transport = puretransport.transport_factory(host='t2ru-bda-mnp-001.corp.tele2.ru', port='10000', username=username, password=username)
    hive_con = hive.connect(thrift_transport=transport)
    cursor = hive_con.cursor()
    cursor.execute(q)
    tbl = pd.DataFrame(cursor.fetchall(), columns = [x[0] for x in cursor.description])
    tbl.columns = [i[i.rfind('.')+1:] for i in tbl.columns.tolist()]
    hive_con.close()
    
    return tbl


def plot_feature_importance(model, feature_names, top=50):
    "Plot feature importance for existing model"
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    importances = model.feature_importance() if 'lightgbm' in str(type(model)) else model.feature_importances_
    top = len(feature_names) if len(feature_names) < top else top
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 0.25 * top))
    sns.barplot(x=imp[:top], y=imp.index[:top])
    plt.show()
    
    
def plot_roc_curve(test, predict, labels, figsize=(10,8)):
    "Plot ROC curve for predicted probabilities and calculate ROC-AUC"
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    if (str(type(test)) != "<class 'list'>") | (str(type(predict)) != "<class 'list'>") | (str(type(labels)) != "<class 'list'>"):
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
        
        
def calculate_feature_importance(data, target, fillna=True, fill_val=-1, test_size=0.20, stratify_by_target=True, shuffle=True, random_state=1,
                                 features=None, n_estimators=500, min_samples_leaf=5, max_features = 0.2, n_jobs=3, max_depth=10, oob_score=True,
                                 n_samples=-1, plot=True, top=30, save_to_file=True, filename='importances.csv', sep=';', decimal=',', 
                                 encoding='cp1251'):
    "Calculate permutation importance for given dataset"
    
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
    importance = importances(model, X_test, y_test, n_samples=n_samples)
    
    if save_to_file:
        importance.to_csv(filename, sep=sep, decimal=decimal, encoding=encoding)
    
    if plot:
        top = len(features) if len(features) < top else top
        plt.figure(figsize=(8, int(0.25 * top)))
        sns.barplot(x=importance.Importance[:top], y=importance.index[:top])
        plt.show()

    return model, importance


def plot_confusion_matrix(y_true, y_pred, prob=True, threshold=0.5, title='Confusion matrix', figsize=(6, 5)):
    "Plot confusion matrix for given threshold value"
    
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
    
    print('Threshold: {}'.format(threshold))
    print("Precision: {}%".format(round(precision_score(y_true, y_pred) * 100, 1)))
    print("Recall: {}%".format(round(recall_score(y_true, y_pred) * 100, 1)))
    print("AUC: {}%".format(round(roc_auc_score(y_true, y_pred) * 100, 1)))
    
    
def plot_predicted_probability(data, label_col='label', pred_col='pred', frac=1.0, title='Predicted probabilities, distributed by label', figsize=(9,8)):
    "Plot distributions of predicted probability, divided by true label"
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    sns.distplot(data[data[label_col] == 0][pred_col], kde=True, bins=np.linspace(0, 1, 51), label='{} = 0'.format(label_col))
    sns.distplot(data[data[label_col] == 1][pred_col], kde=True, bins=np.linspace(0, 1, 51), label='{} = 1'.format(label_col))
    plt.xlabel('Probability', fontsize=12)
    plt.legend()
    plt.show()
    
    
def plot_precision_recall_curve(test, predict, figsize=(10,8)):
    "Plot Precision-Recall curve for predicted probabilities"
    
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