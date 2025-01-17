# Пример ML проекта

Структура:
```
config - конфигурационные файлы
|- settings.toml - публичные настройки проекта (файлы, таблицы, признаки)
|- .secrets.toml - секретные настройки проекта (логины, пароли)
data - данные
|- train.csv - обучающая выборка
|- test.scv - выборка для скоринга
|- features_set1.csv - набор с признаками номер 1
|- features_set2.csv - набор с признаками номер 2
|- features_set3.csv - набор с признаками номер 3
|- features_set4.csv - набор с признаками номер 4
|- features_set5.csv - набор с признаками номер 5
model - модель и артефакты
|- model.sav - сохранённая модель
|- top_features.json - список признаков для модели
notebooks - jupyter ноутбуки
|- loading_data.ipynb - загрузка данных в базу
|- create_model.ipynb - создание модели
|- scoring_data.ipynb - скоринг данных
scoring - результаты скоринга
|- scoring_20200323.csv - результаты скоринга
src - Python код
|- db_tools.py - классы для работы с базами данных
|- ds_tools.py - классы для построения модели
.gitignore - исключения для Git
README.md - описание
```

### config

В конфигах лежат все настройки проекта: источники данных, названия колонок, параметры для отбора признаков и обучения модели.
Для работы с конфигами используется библиотека `dynaconf`, конфиги оформлены в формате `TOML`.

Для проекта обычно требуется два конфига:
* `settings.toml` содержит общедоступную информацию
* `.secrets.toml` содержит чувствительную информацию.

В этом проекте никаких паролей нет, поэтому файл `.secrets.toml` не используется, но для примера там записан вымышленный логин и пароль. 

### data 

Для построения модели использовались данные из соревнования X5 Retail Hero: https://retailhero.ai/

По этим данным были построены различные признаки и залиты как разные наборы данных, чтобы показать как работает пайплайн 
с несколькими источниками. 

### model

В данную папку сохраняется `JSON`-файл с отобранными признаками и их источниками, а также построенная модель.

### notebooks

Здесь лежат `Jupyter`-ноутбуки:
* `loading_data.ipynb` - содержит код для создания базы данных `SQLite` и заливки данных из папки `data` в базу
* `create_model.ipynb` - содержит код для запуска отбора признаков и построения модели
* `scoring_data.ipynb` - содержит код для запуска скоринга

### scoring

В данную папку сохраняется скоринг в формате `CSV`, скоринг также заливается в базу.

### src

Основная папка с `Python` кодом. Содержит 2 модуля.

#### db_tools.py 

Данный модуль содержит классы для подключения к различным базам данных. Сейчас там лежат 3 класса:
* `SQLiteDB` - класс для работы с локальной базой данных `SQLite` (которая используется в этом проекте)
* `TeradataDB` - класс для работы с базой данных `Teradata`
* `HiveDB` - класс для работы с базой данных `Hive`

У каждого класса должно быть 3 метода (но не везде есть):
* `load` - заливка данных в базу
* `read` - чтение данных из базы
* `execute` - выполнение SQL запроса в базе

#### ds_tools.py

Данный модуль содержит классы и функции для построения моделей.

Функции:
* `plot_roc_curve` - строит ROC кривые
* `plot_confusion_matrix` - строит confusion matrix
* `plot_predicted_probability` - строит гистограммы полученных вероятностей в разбивке по таргет переменной
* `plot_precision_recall_curve` - строит precision-recall кривую

Классы:
* `DataKeeper` - хранит список всех доступных признаков для построения моделей и их источники, а также функции для работы с ними
* `Loader` - загружает данных из базы данных, и сжимает их для экономии места
* `FeatureSelector` - отбирает топ признаков из всех доступных. Удаляет признаки с высокой долей пустых значений или одного значения, а также один из пары коррелирующих признаков. Дальше реализовано два способа:
    - плохой, но быстрый - выгрузка feature_importance_ от модели Random Forest. Нужно указать кол-во топ признаков.
    - хороший, но долгий - вычисление permutation importance по модели Random Forest (используется библиотека rfpimp). Нужно либо указать кол-во топ признаков, либо указать threshold значение для permutation importance.
* `ModelBuilder` - создает и обучает модель. В качестве модели используется `LightGBM.LGBMClassifier`, гиперпараметры подбираются с помощью Байесовской оптимизации (используется библиотка `bayes_opt`)
* `Scorer` - скорит тестовые данные
















