## Содержание
- [Into](#DS_course)
- [Проекты](#projects)
- [Используемые библиотеки](#libs)
- [Описание проектов](#description)
- [1. Анализ продаж видео игр](#games_analyz)
- [2. Выбор наиболее прибыльной скважины](#wells)
- [3. Обогащение золота](#gold)
- [4. Машинное обучение для текстов](#texts)

# Into <a name="DS_course"></a>
Здесь представлены несколько моих ноутбуков из курса Data Science который я прошел в 2021 году на Yandex Практикуме. Каждая из этих работ которых была итоговой работой того или иного спринта. Все они выполнены в формате Jupyter Notebooks. Для их запуска можно воспользоваться [google colab](https://research.google.com/colaboratory/) или другими ресурсами. Также можно развернуть jupyter lab на своем ноутбуке. Подробнее [здесь](https://jupyter.org/install)

## Проекты <a name="projects"></a>
1. Аналзи продаж видео игр
2. Выбор наиболее прибыльной скважины
3. Обогащение золота
4. Машинное обучение для текстов

## Используемые библиотеки <a name="libs"></a>
* numpy
* sklearn
* matplotlib
* nltk
* pandas
* tqdm
* seaborn
* wordcloud
* math


## Описание проектов <a name="description"></a>
### 1. Анализ продаж видео игр <a name="games_analyz"></a>
Данная работа была аналитического характера.\
В нашем распоряжении были данных по прадажам видео игр до 2016 года 


**_Цель проекта:_** выявить влияющие на продажи игры закономерности. \
_Задачи:_
1. определить время жизни игровой консоли
2. узнать жанровые предпочтения разных регионов. 

Основные инструменты в данной работы - это pivot_table и query. 

_Ключевые моменты:_ 
* проведение статистических тестов  (ttest_ind) для проверки равенства средних значений двух выборок
* сравнение дисперсий двух выборок по критерию Левена

```
def chk_var_leven(samp1, samp2):
    cnt = min(samp1.shape[0],samp2.shape[0] )

    _, pvalue = st.levene(samp1.head(cnt), samp2.head(cnt))
    if pvalue < alpha:
        print('По критерию Левена, дисперсия двух выборок отличаются Pvalue =', round(pvalue,6))
        return False
    else:
        print('По критерию Левена, дисперсия двух выборок схожи Pvalue =', round(pvalue,6))
        return True
        
equal = chk_var_leven(pc['user_score'], xone['user_score'])

results = st.ttest_ind(pc['user_score'], xone['user_score'], equal_var=equal)
print('результаты проверки:')
if results.pvalue < alpha:
    print('\tОтвергаем нулевую гипотезу о равенстве средних двух выборок. P_value=', round(results.pvalue,6))
else:
    print('\tНет оснований отвергать нулевую гипотезу о равенстве средних двух выборок. P_value=', round(results.pvalue,2))
    
```

### 2. Выбор наиболее прибыльной скважины <a name="wells"></a>
В данном проекте у нас были пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов.


**_Цель проекта_**: построить модель машинного обучения для определения наиболее прибыльного региона. \

_Задачи_:
1. Найти коррелирующие признаки
2. расчитать показатели для безубыточной разработки скважины
3. спрогнозировать прибыль от каждого региона и расчиать риски убытков

_Ключевые моменты_: \
Применил технику Bootstrap, т.к target значений было мало и данные не были распределены нормально
```
    for _ in range(1000):
        target_valid_sampled = target_valid[i].sample(n=500, replace = True, random_state=state)
        prediction_sampled = predicted_series[i][target_valid_sampled.index]
        income_values[i].append(revenue(target_valid_sampled, prediction_sampled, 200))
    
    income_values[i] = pd.Series(income_values[i])
    lower = income_values[i].quantile(q=0.025)
    upper = income_values[i].quantile(q=0.975)
```



![risks of nonprofit investments](https://github.com/ilart/DS_course/blob/main/media/density_of_income.png?raw=true)

### 3. Обогащение золота <a name="gold"></a>
Проект по расчету коэфициента обогащения золота. В нашем распоряжении был датасет с большим количеством признаков полученных в процессе обогащения руды. В котором присутствовало много ненужных признаков. 


**_Цель проекта_**: Построить модель для предсказания коэфициент восстановления золота из руды. \
_Задачи_:
1. Разобраться в технологическом процессе
2. Определить нужные для модели признаки
3. Решить проблему мультиколлинеарности.
4. По формуле проверить полученные данные о коэффициенте.


Сложность проекта в том что надо разобраться в процессе обогащения. Понять суть всех этапов (флотация, несколько стадий фильтрации ), отсеять лишние признаки. А также необходимо было применить кастомную скор-функцию для оценки качества прогноза коэфициента. 

```
scorer = make_scorer(smape_scorer, greater_is_better=False)

parametrs = { 'n_estimators': [10, 12 ],
              'min_samples_leaf': [2,4]
             }

model = RandomForestRegressor(random_state=12345, min_samples_split=2, max_depth=18)

grid = GridSearchCV(model, parametrs, scoring = scorer ,  cv=5)
grid.fit(features_train, target_train)
```

![Anomalies in concentrates of elements](https://github.com/ilart/DS_course/blob/main/media/Density%20of%20elements%20concentrates.png)
![heatmap](https://github.com/ilart/DS_course/blob/main/media/gold_recovery_heatmap.png?raw=true)


### 4. Машинное обучение для текстов <a name="texts"></a>
Проект по классификации текстов по признаку токсичности. 


**_Цель проекта_**: Создать модель для классификации комментариев по признаку токсичности

Основные этапы:
- очистка текста от лишних символов + лемматизация + удаление стоп-слов
- векторизация текста с помощью TfIdf
- борьба с дизбалансом
- Тюнинг моделей
- обучение различных моделей и сравнение результатов



```
parametrs = { 'max_depth': range (150, 200, 50)}
              'min_samples_leaf': range (2,14, 2)}
              'min_samples_split': range (2,8,2) }
              
model = DecisionTreeClassifier(random_state=12345)
grid = GridSearchCV(model, parametrs, scoring="f1", n_jobs=-1, verbose = 3, cv=3)
grid.fit(train_features, train_target)
```

![Зависимость F1 от threshold](https://github.com/ilart/DS_course/blob/main/media/setup_threashold.png?raw=true)
![](https://github.com/ilart/DS_course/blob/main/media/cloud_of_word.png?raw=true)

![Сравнение моделей](https://github.com/ilart/DS_course/blob/main/media/models_comparing.png?raw=true)
