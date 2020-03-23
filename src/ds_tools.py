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
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score
from bayes_opt import BayesianOptimization
import rfpimp
import matplotlib.pyplot as plt
import seaborn as sns
import json
import lightgbm as lgb
import joblib
import itertools
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)


def plot_roc_curve(sets, figsize=(10,8)):
    """Plot ROC curves and calculate ROC-AUC.
    
    Parameters:
        sets (list of tuples): tuples with true labels, predicted probabilities and labels.
            tuple=(y_true, y_pred, label), where:
                y_true (list or pandas.Series of int): true labels
                y_pred (list or pandas.Series of float): predicted probabilities
                label (str): label for legend
        figsize (tuple of int): figure size
    """
    
    plt.figure(figsize=figsize)
    plt.title("ROC curves", fontsize=12)
    
    for i in sets:
        fpr, tpr, _ = roc_curve(i[0], i[1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC={np.round(roc_auc, 4)}, {i[2]}')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True positive rate', fontsize=12)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, prob=True, threshold=0.5, title='Confusion matrix', figsize=(6, 5)):
    """
    Plot confusion matrix for given threshold value.
    
    Parameters:
        y_true (list of int): true labels
        y_pred (list of int or float): predicted labels or probabilities
        prob (bool): True if y_pred is probabilities
        threshold (float): threshold value for predicted probabilities
        title (str): title of confusion matrix
        figsize (tuple of int): figure size
    """

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
    
    
def plot_predicted_probability(y_true, y_pred, title='Predicted probabilities, distributed by label', 
                               figsize=(9,8), kde=True):
    """Plot distributions of predicted probabilities, divided by true labels.
    
    Parameters:
        y_true (list of int): true labels
        y_pred (list of float): predicted probabilities
        title (str): figure title
        figsize (tuple of int): figure size
        kde (bool): show kernel density estimation
    """
    
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    sns.distplot(y_pred[y_true == 0], kde=kde, bins=np.linspace(0, 1, 51), label="target = 0")
    sns.distplot(y_pred[y_true == 1], kde=kde, bins=np.linspace(0, 1, 51), label="target = 1")
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
        """Loader initialization.
        
        Parameters:
            db (db_tools): DB class from db_tools
        """
        super().__init__()
        self.db = db
        
    def load(self, source=None, target_column=None, id_column = 'client_id', feature_source=None,
             feature_sets='all', except_sets=None, except_features=None, silent=False):
        """Load data.
        
        Parameters:
            source (str): table name with target
            target_column (str): column name with target
            id_column (str): column name with id
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
            query_load_ids = f"select {self.id_column}, {self.target_column} from {self.source}"
        else:
            query_load_ids = f"select {self.id_column} from {self.source}"
            
        df = self.reduce_mem_usage(self.db.read(query_load_ids))
            
        if not silent:
            print(f'IDs loaded, {df.shape[0]} rows')
        
        for i in [j for j in self.feature_sets if j['name'] in subsets]:
            
            features = i['features']
            if except_features:
                features = [i for i in features if i not in except_features]
                        
            add_query = f"select t.{self.id_column}, " + ", ".join(features) + \
                        f" from {self.source} as t join {i['source']} as d on " + \
                        f"d.{self.id_column} = t.{self.id_column}"
                             
            add_df = self.reduce_mem_usage(self.db.read(add_query))
            df = df.merge(add_df, on=self.id_column, how='left')
            
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
    
    
class FeatureSelector(DataKeeper):
    def __init__(self, db):
        """Feature selector initialization.
        
        Parameters:
            db (db_tools): DB class from db_tools
        """
        super().__init__()
        self.db = db
        
    def load_data(self, source, target_column='target', id_column = 'client_id',
                  feature_source=None, feature_sets='all', except_sets=None, 
                  except_features=None, silent=False):
        """Load data for feature selection.
        
        Parameters:
            source (str): table name with target
            target_column (str): column name with target
            id_column (str): column name with id
            feature_source (str): path to file with feature names
            feature_sets (str or list of str): 
                'all' - load all features, 
                'main' - load only main features, 
                if list - names of feature sets for loading, available names: 
                        'set1', 'set2', 'set3', 'set4', 'set5'
            except_sets (None or str or list of str): if not None, this feature set(s) won't be loaded
            except_features (None or str or list of str): if not None, this feature(s) won't be loaded
            silent (bool): don't output any information
        """
        self.target_column = target_column
        self.id_column = id_column
        self.useless = {}
        
        if not silent:
            print("Loading data ...")
        
        loader = Loader(self.db)
        self.data = loader.load(source=source, target_column=target_column, id_column=id_column,
                                feature_source=feature_source, feature_sets=feature_sets, 
                                except_sets=except_sets, except_features=except_features, silent=silent)
              
        self.features = [col for col in self.data.columns if col not in [self.id_column, self.target_column]]
        
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
            self.data.drop([col for col in self.useless[k] if col in self.data.columns], axis=1, inplace=True)
            all_useless += self.useless[k]
        self.features = [col for col in self.data.columns if col not in [self.id_column, self.target_column]]
        if not silent:
            print('Deleted {} useless features, shape: {}'.format(len(list(set(all_useless))), self.data.shape))
            
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
        if np.sum(self.data[self.features].dtypes == 'object') > 0:
            self.cat_dict = {}
            for var in self.data.dtypes[self.data.dtypes == 'object'].index.tolist():
                self.data[var] = self.data[var].astype('category')
                self.cat_dict[var] = self.data[var].cat.categories
                self.data[var] = self.data[var].cat.codes
            if not silent:
                print('Categorical features encoded')
    
    def calculate_feature_importances(self, test_size=0.2, n_estimators=500, min_samples_leaf=5, 
                                      max_features=0.2, n_jobs=-1, oob_score=True, random_state=1,
                                      permutation=True, silent=False):
        """Calculate feature importance.
        
        Parameters:
            test_size (float): test data size
            n_estimators (int): n_estimators for RandomForestClassifier
            min_samples_leaf (int): min_samples_leaf for RandomForestClassifier
            max_features (float): max_features for RandomForestClassifier
            n_jobs (int): n_jobs for RandomForestClassifier
            oob_score (bool): oob_score for RandomForestClassifier
            random_state (int): random_state
            permutation (bool): use permutation importance
            silent (bool): don't output any information
        """
        
        if not silent:
            print(f"Calculating feature importances, permutation={permutation} ...")
        
        self.permutation = permutation
        self.fillna()
        self.encode_cats()
        train, test = train_test_split(self.data, 
                                       test_size=test_size, 
                                       stratify=self.data[self.target_column], 
                                       shuffle=True, 
                                       random_state=random_state)

        X_train, y_train = train[self.features], train[self.target_column]
        X_test, y_test = test[self.features], test[self.target_column]

        model = RandomForestClassifier(n_estimators=n_estimators,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       n_jobs=n_jobs,
                                       oob_score=oob_score, 
                                       random_state=random_state)
        model.fit(X_train, y_train)

        if permutation:
            self.feature_importances = rfpimp.importances(model, X_test, y_test, n_samples=-1)
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

    def export_top_features(self, filename='top_features.json', top=30, 
                            permutation_threshold=0.0005, silent=False):
        """Export top features for model creation.
        
        Parameters:
            filename (str): filename for saving top features
            top (int): top N features, which will be saved
            permutation_threshold (float): permutation importance threshold for selecting top features,
                                           used only if self.permutation=True, otherwise, top will be used
            silent (bool): don't output any information
        """
        top = len(self.features) if len(self.features) < top else top
        
        if self.permutation:
            top_features = self.feature_importances[self.feature_importances.Importance > permutation_threshold].index.tolist()
        else:
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
            print(f'Top {len(top_features)} features exported to file {filename}')
    
    def run(self, source, filename, target_column='label', id_column='client_id', feature_source=None,
            feature_sets='all', except_sets=None, except_features=None, 
            missing_threshold=0.7, disbalance_threshold=0.7, collinear_threshold=0.9,    
            top=30, permutation=True, permutation_threshold=0.0005, silent=False):
        """Run feature selection procedure step-by-step.
        
        Parameters:
            source (str): table name with target
            filename (str): filename for saving top features
            target_column (str): column name with target
            id_column (str): column name with id
            feature_source (str): path to file with feature names
            feature_sets (str or list of str): 
                'all' - load all features, 
                'main' - load only main features, 
                if list - names of feature sets for loading, available names: 
                        'set1', 'set2', 'set3', 'set4', 'set5'
            except_sets (None or str or list of str): if not None, this feature set(s) won't be loaded
            except_features (None or str or list of str): if not None, this feature(s) won't be loaded
            missing_threshold (float): threshold missing fraction
            disbalance_threshold (float): threshold disbalance fraction
            collinear_threshold (float): threshold corellation coefficient
            top (int): top N features, which will be saved
            permutation (bool): use permutation importance
            permutation_threshold (float): permutation importance threshold for selecting top features,
                                           used only if permutation=True, otherwise, top will be used
            silent (bool): don't output any information
        """
        self.load_data(source=source, target_column=target_column, id_column=id_column,
                       feature_source=feature_source, feature_sets=feature_sets, 
                       except_sets=except_sets, except_features=except_features, 
                       silent=silent)    
        self.identify_missing(threshold=missing_threshold, silent=silent)
        self.identify_disbalance(threshold=disbalance_threshold, silent=silent)
        self.identify_collinear(threshold=collinear_threshold, silent=silent)
        self.drop_useless(silent=silent)
        self.calculate_feature_importances(permutation=permutation, silent=silent)
        self.export_top_features(filename=filename, top=top, permutation_threshold=permutation_threshold, silent=silent)
        self.plot_feature_importances(top=top)


class ModelBuilder(DataKeeper):
    def __init__(self, db, random_state=1):
        """Model Creator initialization.
        
        Parameters:
            db (db_tools): DB class from db_tools
        """
        super().__init__()
        self.db = db
        self.random_state=random_state

    def load_data(self, source, target_column='target', id_column = 'client_id',
                  feature_source=None, silent=False):
        """Load data for feature selection.
        
        Parameters:
            source (str): table name with target
            target_column (str): column name with target
            id_column (str): column name with id
            feature_source (str): path to file with feature names
            silent (bool): don't output any information
        """
        
        if not silent:
            print("Loading data ...")
        
        self.target_column = target_column
        self.id_column = id_column
        
        loader = Loader(self.db)
        self.data = loader.load(source=source, target_column=target_column, id_column=id_column,
                                feature_source=feature_source, silent=silent)
              
        self.features = [col for col in self.data.columns if col not in [self.id_column, self.target_column]]
        
        if not silent:
            print('All data loaded successfully, shape: {}'.format(self.data.shape))
    
    def encode_cats(self, filename='cat_dict.npy', silent=False):
        """Encode categorical features.
        
        Parameters:
            filename (str): filename for saving dictionary with categories ids for categorical features
            silent (bool): don't output any information
        """
        if np.sum(self.data[self.features].dtypes == 'object') > 0:
            
            self.cat_dict = {}
            for var in self.data.dtypes[self.data.dtypes == 'object'].index.tolist():
                self.data[var] = self.data[var].astype('category')
                self.cat_dict[var] = self.data[var].cat.categories
                self.data[var] = self.data[var].cat.codes
            
            np.save(filename, self.cat_dict)
            if not silent:
                print(f'Categorical features encoded, dictionary saved to {filename}')
        else:
            if not silent:
                print('Categorical features not found')

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
            
    def create_holdout(self, test_size=0.2, shuffle=True, silent=False):
        """Split data to train and holdout datasets.
        
        Parameters:
            test_size (float): holdout fraction
            shuffle (bool): shuffle data before split
            silent (bool): don't output any information
        """
        self.train, self.holdout = train_test_split(
            self.data, test_size=test_size, stratify=self.data[self.target_column], shuffle=True, random_state=self.random_state
        )
        if not silent:
            print(f"Holdout created, train shape: {self.train.shape}, holdout shape: {self.holdout.shape}")
    
    def optimize_params(self, params, cv_splits=4, cv_shuffle=True, init_points=5, n_iter=100, silent=False):
        """Optimize model params using Bayesian optimization.
        
        Parameters:
            params (dict): parameter search space
            cv_splits (int): number of folds
            cv_shuffle (bool): use shuffle before splitting to folds
            init_points (int) initial points for searching
            n_iter (int): search iterations
            silent (bool): don't output any information
        """
        if not silent:
            print("Optimizing params ...")
        
        self.params = params
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=self.random_state)
        self.bo = BayesianOptimization(self.model_evaluate, params, verbose=(1 if not silent else 0))
        self.bo.maximize(init_points=init_points, n_iter=n_iter)
        
        if not silent:
            print("Params optimized, train final model ...") 
        
        self.model = lgb.LGBMClassifier(
            random_state=self.random_state, 
            objective='binary', 
            metric='auc', 
            is_unbalance=True,
            num_leaves=int(self.bo.max['params']['num_leaves']),
            num_iterations=int(self.bo.max['params']['num_iterations']),
            min_data_in_leaf=int(self.bo.max['params']['min_data_in_leaf']),
            max_depth=int(self.bo.max['params']['max_depth']),
            bagging_fraction=self.bo.max['params']['bagging_fraction'],
            feature_fraction=self.bo.max['params']['feature_fraction'],
            max_bin=int(self.bo.max['params']['max_bin']),
            colsample_bytree=self.bo.max['params']['colsample_bytree'],
            learning_rate=self.bo.max['params']['learning_rate'],
            subsample=self.bo.max['params']['subsample'],
            min_gain_to_split=self.bo.max['params']['min_gain_to_split']
        )
        self.model.fit(self.train[self.features], self.train[self.target_column])
        
    def model_evaluate(self, **params):
        """Evaluation function for Bayesian optimization.

        Parameters:
            params (dict): dictionary with model parameters
        """
        fold_auc = []
        for train_idx, test_idx in self.cv.split(self.train[self.features], self.train[self.target_column]):
            
            train_set = self.train.iloc[train_idx]
            test_set = self.train.iloc[test_idx]
            
            model = lgb.LGBMClassifier(
                random_state=1, 
                objective='binary', 
                metric='auc', 
                is_unbalance=True,
                num_leaves=int(params['num_leaves']),
                num_iterations=int(params['num_iterations']),
                min_data_in_leaf=int(params['min_data_in_leaf']),
                max_depth=int(params['max_depth']),
                bagging_fraction=params['bagging_fraction'],
                feature_fraction=params['feature_fraction'],
                max_bin=int(params['max_bin']),
                colsample_bytree=params['colsample_bytree'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                min_gain_to_split=params['min_gain_to_split']
            )
            
            model.fit(train_set[self.features], train_set[self.target_column])
            y_pred = model.predict_proba(test_set[self.features])[:, 1]
            fold_auc.append(roc_auc_score(y_true=test_set[self.target_column], y_score=y_pred))

        return np.mean(fold_auc)
        
    def show_metrics(self, metrics=['roc_curve', 'pr_curve', 'prob_hist', 'conf_matrix'], 
                     prob_kde=True, cm_threshold=0.5, figsize=(10,8)):
        """Show metrics of model, calculated on holdout dataset.
        
        Parameters:
            metrics (list of str): which metrics to show, available values:
                    'roc_curve' - plot roc curves, for train and holdout
                    'pr_curve' - plot precision-recall curve for holdout
                    'prob_hist' - plot histogram of probabilities for holdout
                    'conf_matrix' - plot confusion matrix for holdout
            prob_kde (bool): use kernel density estimation in prob_hist
            cm_threshold (float): threshold value for holdout probabilities in conf_matrix
            figsize (tuple of int): figure size
        """
        
        if 'roc_curve' in metrics:
            plot_roc_curve(sets=[
                (self.train[self.target_column], self.model.predict_proba(self.train[self.features])[:, 1], 'train'),
                (self.holdout[self.target_column], self.model.predict_proba(self.holdout[self.features])[:, 1], 'holdout')
            ], figsize=figsize)
        
        if 'pr_curve' in metrics:
            plot_precision_recall_curve(
                self.holdout[self.target_column], 
                self.model.predict_proba(self.holdout[self.features])[:, 1], 
                figsize=figsize
            )
        
        if 'prob_hist' in metrics:
            plot_predicted_probability(
                self.holdout[self.target_column],  
                self.model.predict_proba(self.holdout[self.features])[:, 1],
                figsize=figsize,
                kde=prob_kde
            )
            
        if 'conf_matrix' in metrics:
            plot_confusion_matrix(
                self.holdout[self.target_column],  
                self.model.predict_proba(self.holdout[self.features])[:, 1], 
                threshold=cm_threshold
            )
        
    def save_model(self, filename, silent=False, plot_hist=True, figsize=(10,8)):
        joblib.dump(self.model, filename)
        
        if not silent:
            print(f"Model saved to file {filename}")
        
        if plot_hist:
            plt.figure(figsize=figsize)
            plt.title("Histogram of probabilities", fontsize=12)
            sns.distplot(self.model.predict_proba(self.holdout[self.features])[:, 1])
            plt.xlabel('Probability', fontsize=12)
            plt.show()
            
    def run(self, source, params, target_column='target', id_column = 'client_id',
            feature_source=None, encode_filename='cat_dict.npy', na_value=-1,
            holdout_size=0.2, holdout_shuffle=True, cv_splits=4, cv_shuffle=True,
            init_points=5, n_iter=100, prob_kde=True, cm_threshold=0.5, 
            metrics=['roc_curve', 'pr_curve', 'prob_hist', 'conf_matrix'], 
            figsize=(10,8), model_filename='model.sav', plot_hist=True, silent=False):
        """Run model builder step-by-step.
        
        Parameters:
            source (str): table name with target
            params (dict): parameter search space for params optimizer
            target_column (str): column name with target
            id_column (str): column name with id
            feature_source (str): path to file with feature names
            encode_filename (str): filename for saving dictionary with categories for categorical features
            na_value (int or float): value for filling NA values
            holdout_size (float): holdout fraction
            holdout_shuffle (bool): shuffle data before split holdout
            cv_splits (int): number of folds for params optimizer
            cv_shuffle (bool): use shuffle before splitting to folds for params optimizer
            init_points (int) initial points for searching for params optimizer
            n_iter (int): search iterations for params optimizer
            prob_kde (bool): use kernel density estimation in prob_hist
            cm_threshold (float): threshold value for holdout probabilities in conf_matrix
            metrics (list of str): which metrics to show, available values:
                    'roc_curve' - plot roc curves, for train and holdout
                    'pr_curve' - plot precision-recall curve for holdout
                    'prob_hist' - plot histogram of probabilities for holdout
                    'conf_matrix' - plot confusion matrix for holdout
            figsize (tuple of int): figure sizes
            plot_hist (bool): plot histogram of probabilities for holdout
            silent (bool): don't output any information
        """
        self.load_data(source=source, target_column=target_column, id_column=id_column,
                       feature_source=feature_source, silent=silent)
        self.encode_cats(filename=encode_filename, silent=silent)
        self.fillna(value=na_value, silent=silent)
        self.create_holdout(test_size=holdout_size, shuffle=holdout_shuffle, silent=silent)
        self.optimize_params(cv_splits=cv_splits, cv_shuffle=cv_shuffle, params=params, n_iter=n_iter, silent=silent)
        self.save_model(filename=model_filename, silent=silent, plot_hist=plot_hist, figsize=figsize)
        self.show_metrics(metrics=metrics, prob_kde=prob_kde, cm_threshold=cm_threshold, figsize=figsize)


class Scorer(DataKeeper):
    def __init__(self, db):
        """Scorer init. 
        
        Parameters:
            db (db_tools): DB class from db_tools
        """
        super().__init__()
        self.db = db
        
    def load_data(self, source, feature_source, id_column = 'client_id', silent=False):
        """Load data for feature selection.
        
        Parameters:
            source (str): table name with target
            feature_source (str): path to file with feature names
            id_column (str): column name with id
            silent (bool): don't output any information
        """
        
        if not silent:
            print("Loading data ...")
        
        self.id_column = id_column
        
        loader = Loader(self.db)
        self.data = loader.load(source=source, feature_source=feature_source, id_column=id_column, silent=silent)
        self.features = [col for col in self.data.columns if col not in [self.id_column]]
        
        if not silent:
            print('All data loaded successfully, shape: {}'.format(self.data.shape))

    def load_model(self, model, cat_dict=None, silent=False):
        """Load model filename with artefacts.
        
        Parameters:
            model (str): path to file with model
            cat_dict (str) path to file with dict for categorial features
            silent (bool): don't output any information
        """
        self.model_filename = model
        self.cat_dict_filename = cat_dict
        
        self.model = joblib.load(self.model_filename)
        if not silent:
            print(f"Model loaded from: {self.model_filename}")
        
        if cat_dict:
            self.cat_dict = np.load(self.cat_dict_filename, allow_pickle=True).item() 
            if not silent:
                print(f"Categorical dictionary loaded from: {self.cat_dict_filename}")
            
    def encode_cats(self, silent=False):
        """Encode categorical features.
        
        Parameters:
            silent (bool): don't output any information
        """
        
        if self.cat_dict_filename:
            cats = df.dtypes[df.dtypes == 'object'].index.tolist()
            for var in cats:
                self.data[var] = pd.Categorical(self.data[var].values, categories=self.cat_dict[var])
                self.data[var] = self.data[var].cat.codes
    
            if not silent:
                print(f'Categorical features encoded')
            
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
                
    def scoring(self, pred_col='pred', silent=False):
        """Scoring data. 
        
        Parameters:
            pred_col (str): column name with predicted probability
            silent (bool): don't output any information
        """
        self.pred_col = pred_col
        
        if not silent:
            print("Scoring data ...")
        
        self.data[self.pred_col] = self.model.predict_proba(self.data[self.features])[:, 1]
        
        if not silent:
            print("Scoring completed")
            
    def save_scoring(self, filename, table=None, silent=False):
        """Save scoring. 
        
        Parameters:
            filename (str): path to file with scoring
            table (str): table name with scoring
            silent (bool): don't output any information
        """
        if not silent:
            print("Saving scoring ...")
        
        self.data[[self.id_column, self.pred_col]].to_csv(filename, decimal=".", sep=",", index=False)
        if not silent:
            print(f"Scoring saved to file: {filename}")
        
        if table:
            self.db.load(filename=filename, table=table, primary_key=self.id_column)
            
            if not silent:
                print(f"Scoring loaded to table: {table}")

    def plot_hist(self, frac=1.0, figsize=(10,8)):
        """Plot histogram of predicted probabilities.
        
        Parameters:
            frac (float): fraction of scoring data
            figsize (tuple of int): figure size
        """
        plt.figure(figsize=figsize)
        plt.title("Histogram of probabilities", fontsize=12)
        sns.distplot(self.data.sample(frac=frac)[self.pred_col])
        plt.xlabel('Probability', fontsize=12)
        plt.show()

    def run(self, source, feature_source, model, cat_dict=None, id_column = 'client_id', 
            na_value=-1, pred_col='pred', scoring_filename='scoring.csv', scoring_table='scoring',
            plot=True, hist_frac=1.0, figsize=(10,8), silent=False):
        """Run scoring step-by-step.
        
        Parameters:
            source (str): table name with target
            feature_source (str): path to file with feature names
            model (str): path to file with model
            cat_dict (str) path to file with dict for categorial features
            id_column (str): column name with id
            na_value (int or float): fill NA values
            pred_col (str): column name with predicted probability
            scoring_filename (str): path to file with scoring
            scoring_table (str): table name with scoring
            silent (bool): don't output any information
        """
        self.load_data(source=source, feature_source=feature_source, id_column=id_column, silent=silent)
        self.load_model(model=model, cat_dict=cat_dict, silent=silent)
        self.encode_cats(silent=silent)
        self.fillna(value=na_value, silent=silent)
        self.scoring(pred_col=pred_col, silent=silent)
        self.save_scoring(filename=scoring_filename, table=scoring_table, silent=silent)
        if plot:
            self.plot_hist(frac=hist_frac, figsize=figsize)



