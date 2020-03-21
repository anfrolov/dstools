#
# Data Science tools.
#
# Module includes tools for loading data, feature selection, 
# training model and scoring data, and useful functions
# for model metrics visualisation.
#

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rfpimp import importances
import matplotlib.pyplot as plt
import seaborn as sns
import json
import lightgbm as lgb
import joblib
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)


class DataKeeper:
    """Class with default features and functions for dumping, 
    loading, merging, and converting feature sets.
    """
    def __init__(self):
        """Default feature set.
        """
        self.feature_sets = []
        self.feature_sets.append({
            'name': 'set1',
            'class': 'main',
            'type': 'numerical',
            'source': 'features_set1',
            'features': [
                'set1_feature_0', 'set1_feature_1', 'set1_feature_2', 'set1_feature_3', 'set1_feature_4'
            ]            
        })
        self.feature_sets.append({
            'name': 'set2',
            'class': 'extra',
            'type': 'binary',
            'source': 'features_set2',
            'features': [
                'set2_feature_0', 'set2_feature_1', 'set2_feature_2', 'set2_feature_3', 'set2_feature_4'
            ]            
        })
        self.feature_sets.append({
            'name': 'set3',
            'class': 'extra',
            'type': 'numerical',
            'source': 'features_set3',
            'features': [
                'set3_feature_0', 'set3_feature_1', 'set3_feature_2', 'set3_feature_3', 'set3_feature_4',
                'set3_feature_5', 'set3_feature_6', 'set3_feature_7', 'set3_feature_8'
            ]            
        })
        self.feature_sets.append({
            'name': 'set4',
            'class': 'main',
            'type': 'numerical',
            'source': 'features_set4',
            'features': [
                'set4_feature_0', 'set4_feature_1', 'set4_feature_2', 'set4_feature_3', 'set4_feature_4',
                'set4_feature_5', 'set4_feature_6', 'set4_feature_7', 'set4_feature_8', 'set4_feature_9'
            ]            
        })
        self.feature_sets.append({
            'name': 'set5',
            'class': 'main',
            'type': 'numerical',
            'source': 'features_set5',
            'features': [
                'set5_feature_0', 'set5_feature_1', 'set5_feature_2', 'set5_feature_3', 'set5_feature_4',
                'set5_feature_5', 'set5_feature_6', 'set5_feature_7', 'set5_feature_8', 'set5_feature_9'
            ]            
        })
        
    def dump_features(self, filename='features.json'):
        """Dump features to JSON file.
        
        Parameters:
            filename (str): filename
        """
        with open(filename, 'w') as f:
            json.dump(self.feature_sets, f)
            
    def load_features(self, filename='features.json'):
        """Load features from JSON file.
        
        Parameters:
            filename (str): filename
        """
        with open(filename, 'r') as f:
            self.feature_sets = json.load(f)
            
    def merge_features(self, input_files, output_file='merged_features.json'):
        """Merge several JSON files with feature sets.
        
        Parameters:
            input_files (list of str): list of JSON files to be merged
            output_file (str): name of merged file
        """
        
        all_features = []
        for file in input_files:
            with open(file, 'r') as f:
                temp_json = json.load(f)
            for subset in temp_json:
                all_features = all_features + subset['features']
        all_features = list(set(all_features))
        
        merged_json = []
        for subset in self.feature_sets:
            features = [i for i in subset['features'] if i in all_features]
            if len(features) > 0:
                merged_json.append({
                    'name': subset['name'],
                    'class': subset['class'],
                    'type': subset['type'],
                    'source': subset['source'],
                    'features': features
                })
        with open(output_file, 'w') as f:
            json.dump(merged_json, f)
        
    def convert_features(self, features, filename):
        """Convert list of features to JSON file.
        
        Parameters:
            features (list of str): list of features
            filename (str): JSON filename
        """
        json_file = []
        for subset in self.feature_sets:
            fts = [i for i in subset['features'] if i in features]
            if len(fts) > 0:
                json_file.append({
                    'name': subset['name'],
                    'class': subset['class'],
                    'type': subset['type'],
                    'source': subset['source'],
                    'features': fts
                })
        with open(filename, 'w') as f:
            json.dump(json_file, f)


class Loader(DataKeeper):
    def __init__(self, db):
        super().__init__()
        self.db = db
        
    def load(self, source=None, target_column=None, id_column = 'client_id', feature_source=None,
             feature_sets='all', except_sets=None, except_features=None, silent=False):
        """Load data.
        
        Parameters:
            target_source (str): table name with target
            target_column (str): column name with target
            feature_source (str): path to file with feature names
            feature_sets (str or list of str): 
                'all' - load all features, 
                'main' - load only main features, 
                if list - names of feature sets for loading, available names: 
                        'set1', 'set2', 'set3', 'set4', 'set5'
            except_sets (None or str or list of str): if not None, this feature set(s) won't be loaded
            except_features (None or str or list of str): if not None, this feature(s) won't be loaded
            silent (bool): don't output any information
            
        Returns:
            pandas.DataFrame with loaded data
        """
        
        self.source = source
        self.target_column = target_column
        self.id_column = id_column
        
        if feature_source:
            self.load_features(feature_source)
        
        if feature_sets == 'all':
            subsets = [i['name'] for i in self.feature_sets]
        elif feature_sets == 'main':
            subsets = [i['name'] for i in self.feature_sets if i['class'] == 'main']
        else:
            subsets = feature_sets
        
        if except_sets:
            subsets = [i for i in subsets if i not in except_sets]
        
        if target_column:
            query_load_ids = f"select {self.id_column}, {self.target_column} from {self.target_source}"
        else:
            query_load_ids = f"select {self.id_column} from {self.target_source}"
            
        df = self.reduce_mem_usage(self.db.read(query_load_ids))
            
        if not silent:
            print(f'IDs loaded, {df.shape[0]} rows')
        
        for i in [j for j in self.feature_sets if j['name'] in subsets]:
            
            features = i['features']
            if except_features:
                features = [i for i in features if i not in except_features]
                        
            add_query = f"select {self.id_column}, " + ", ".join(features) + \
                        f" from {self.source} as t join {i['source']} as d on " + \
                        f"d.{self.id_column} = t.{self.id_column}"
                             
            add_df = self.reduce_mem_usage(self.db.read(add_query))
            df = df.merge(add_df, on='id', how='left')
            
            if not silent:
                print(f"Loaded {len(features)} features from {i['name']}")
            
        return df
        
    def reduce_mem_usage(self, df, silent=True):
        """Reduce memory usage via conversion of data types.
        
        Parameters:
            df (pandas.DataFrame): initial DataFrame
            silent (bool): don't output any information
            
        Returns:
            df (pandas.DataFrame): reduced DataFrame
        """
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
        if not silent: 
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            ))
        return df   
    
    
class FeatureSelector(General):
    def __init__(self):
        super().__init__()
        
    def load_data(self, report_date, target_source, target_column='label', feature_source=None,
                  teradata_dsn='dsn', feature_sets='all', except_sets=None, except_features=None, 
                  silent=False):
        """Load data for feature selection.
        
        Parameters:
            report_date (str): report date, format: '2019-01-01'
            target_source (str): table name with target
            target_column (str): column name with target
            feature_source (str): path to file with feature names
            teradata_dsn (str): name of ODBC connection
            feature_sets (str or list of str): 
                'all' - load all features, 
                'set1' - load only set1 features, 
                if list - names of feature sets for loading, available names: 
                        'set1', 'set2', 'set3'
            except_sets (None or str or list of str): if not None, this feature set(s) won't be loaded
            except_features (None or str or list of str): if not None, this feature(s) won't be loaded
            silent (bool): don't output any information
        """
        self.target_column = target_column
        self.useless = {}
        
        loader = Loader()
        self.data = loader.load(report_date=report_date, target_source=target_source, target_column=target_column, 
                                feature_source=feature_source, teradata_dsn=teradata_dsn, feature_sets=feature_sets, 
                                except_sets=except_sets, except_features=except_features, silent=silent)
        
        self.features = [col for col in self.data.columns if col not in ['id', self.target_column]]
        
        if not silent:
            print('All data loaded successfully, shape: {}'.format(self.data.shape))
        
    def identify_missing(self, threshold, silent=False):
        """Identify features with high fraction of missing values. 
        Features with missing fraction greather than threshold value will be removed later.
        
        Parameters:
            threshold (float): threshold fraction
            silent (bool): don't output any information
        """
        missing_counts = self.data[self.features].isnull().sum() / self.data[self.features].shape[0]
        for_removal = missing_counts[missing_counts > threshold].index.tolist()
        self.useless['missing'] = for_removal
        if not silent:
            print('Found {} features with missing fraction greater than {}'.format(len(for_removal), threshold))        
    
    def identify_disbalance(self, threshold, silent=False):
        """Identify features with high fraction of most frequent value.
        Features with disbalance fraction greather than threshold value will be removed later.
        
        Parameters:
            threshold (float): threshold fraction
            silent (bool): don't output any information
        """
        for_removal = []
        all_useless = []
        for k in self.useless:
            all_useless += self.useless[k]
        for col in self.features:
            if col not in all_useless:
                if sum(self.data[col] == self.data[col].value_counts().index[0]) / self.data.shape[0] > threshold:
                    for_removal.append(col)
        self.useless['disbalance'] = for_removal
        if not silent:
            print('Found {} features with disbalance fraction greater than {}'.format(len(for_removal), threshold))

    def identify_collinear(self, threshold, silent=False):
        """Identify collinear features. One feature from collinear pair will be removed later.
        
        Parameters:
            threshold (float): threshold for corellation coefficient
            silent (bool): don't output any information
        """
        
        corr_matrix = self.data[self.features].corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        for_removal = upper.where(upper > threshold).dropna(how='all', axis=1).columns.tolist()
        self.useless['collinear'] = for_removal
        if not silent:
            print('Found {} features with correlation coefficient greater than {}'.format(len(for_removal), threshold))

    def drop_useless(self, silent=False):
        """Drop useless features (missing, disbalance, collinear) from data.
        
        Parameters:
            silent (bool): don't output any information
        """
        all_useless = []
        for k in self.useless:
            self.data.drop([i for i in self.useless[k] if i in self.data.columns], axis=1, inplace=True)
            all_useless += self.useless[k]
        self.features = [i for i in self.data.columns.tolist() if i not in ['subs_id', self.target_column]]
        if not silent:
            print('Deleted {} useless features, shape of data: {}'.format(len(list(set(all_useless))), self.data.shape))
            
    def fillna(self, value=-1, silent=False):
        """Fill NA values in data.
        
        Parameters:
            value (int or float): fill value
            silent (bool): don't output any information
        """
        if self.data.isnull().values.any():
            self.data = self.data.fillna(value)
            if not silent:
                print('NA values replaced by {}'.format(value))
            
    def encode_cats(self, silent=False):
        """Encode categorical features.
        
        Parameters:
            silent (bool): don't output any information
        """
        if np.sum(self.data.dtypes == 'object') > 0:
            self.cat_dict = {}
            for var in self.data.dtypes[self.data.dtypes == 'object'].index.tolist():
                self.data[var] = self.data[var].astype('category')
                self.cat_dict[var] = self.data[var].cat.categories
                self.data[var] = self.data[var].cat.codes
            if not silent:
                print('Categorical features encoded')
    
    def calculate_feature_importances(self, permutation=False, silent=False):
        """Calculate feature importance.
        
        Parameters:
            permutation (bool): use permutation importance
            silent (bool): don't output any information
        """
        self.fillna()
        self.encode_cats()
        train, test = train_test_split(self.data, 
                                       test_size=0.2, 
                                       stratify=self.data[self.target_column], 
                                       shuffle=True, 
                                       random_state=1)

        X_train, y_train = train[self.features], train[self.target_column]
        X_test, y_test = test[self.features], test[self.target_column]

        model = RandomForestClassifier(n_estimators=500,
                                       min_samples_leaf=5,
                                       max_features=0.2,
                                       n_jobs=-1, 
                                       max_depth=10,
                                       oob_score=True, 
                                       random_state=1)
        model.fit(X_train, y_train)

        if permutation:
            self.feature_importances = importances(model, X_test, y_test, n_samples=-1)
        else:
            self.feature_importances = pd.DataFrame({
                'Importance': model.feature_importances_,
                'Feature': self.features}).sort_values(by='Importance', ascending=False).set_index('Feature')

        score = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:, 1])
        if not silent:
            print('Feature importances calculated, ROC-AUC score {}'.format(np.round(score, 4)))

    def plot_feature_importances(self, top=30):
        """Plot calculated feature importances.
        
        Parameters:
            top (int): top N features, which will be shown.
        """
        top = len(self.features) if len(self.features) < top else top
        plt.figure(figsize=(8, 0.3 * top))
        sns.barplot(x=self.feature_importances['Importance'][:top], 
                    y=self.feature_importances.index[:top])
        plt.show()

    def export_top_features(self, filename='top_features.json', top=30, silent=False):
        """Export top features for model creation.
        
        Parameters:
            filename (str): filename for saving top features
            top (int): top N features, which will be saved
            silent (bool): don't output any information
        """
        top_features = self.feature_importances.index.tolist()[:top]
        top_features_json = []
        for subset in self.feature_sets:
            features = [i for i in subset['features'] if i in top_features]
            if len(features) > 0:
                top_features_json.append({
                    'name': subset['name'],
                    'class': subset['class'],
                    'type': subset['type'],
                    'source': subset['source'],
                    'features': features
                })
        with open(filename, 'w') as f:
            json.dump(top_features_json, f)
        if not silent:
            print('Top features exported to file {}'.format(filename))
    
    def run(self, report_date, target_source, filename, target_column='label', feature_source=None,
            teradata_dsn='dsn', feature_sets='all', except_sets=None, except_features=None, 
            missing_threshold=0.7, disbalance_threshold=0.7, collinear_threshold=0.9,    
            permutation=False, top=30, silent=False):
        """Run feature selection procedure step-by-step.
        
        Parameters:
            report_date (str): report date, format: '2019-01-01'
            target_source (str): table name with target
            target_column (str): column name with target
            feature_source (str): path to file with feature names
            teradata_dsn (str): name of ODBC connection
            feature_sets (str or list of str): 
                'all' - load all features, 
                'dmsc' - load only dmsc features, 
                if list - names of feature sets for loading, available names: 
                        'dmsc_numerical', 'dmsc_binary', 'dmsc_categorical', 
                        'dmx_be', 'dmx_chrgd', 'dmx_dsuite', 'dmx_mc', 'dmx_scd2'
            except_sets (None or str or list of str): if not None, this feature set(s) won't be loaded
            except_features (None or str or list of str): if not None, this feature(s) won't be loaded
            missing_threshold (float): threshold missing fraction
            disbalance_threshold (float): threshold disbalance fraction
            collinear_threshold (float): threshold corellation coefficient
            permutation (bool): use permutation importance
            filename (str): filename for saving top features
            top (int): top N features, which will be saved
            silent (bool): don't output any information
        """
        self.load_data(report_date=report_date, target_source=target_source, 
                       target_column=target_column, feature_source=feature_source,
                       teradata_dsn=teradata_dsn, feature_sets=feature_sets, 
                       except_sets=except_sets, except_features=except_features, 
                       silent=silent)
        self.identify_missing(threshold=missing_threshold, silent=silent)
        self.identify_disbalance(threshold=disbalance_threshold, silent=silent)
        self.identify_collinear(threshold=collinear_threshold, silent=silent)
        self.drop_useless(silent=silent)
        self.calculate_feature_importances(permutation=permutation, silent=silent)
        self.export_top_features(filename=filename, top=top, silent=silent)
        self.plot_feature_importances(top=top)
        
        
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