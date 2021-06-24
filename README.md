# DS_course
Здесь представлены несколько моих ноутбуков из курса Data Science. Каждый из которых был итоговой работой того или иного спринта.  

## Проекты
1. Аналзи продаж видео игр
2. Выбор наиболее прибыльной скважины
3. Обогащение золота
4. Машинное обучение для текстов

## Используемые библиотеки
* numpy
* sklearn
* matplotlib
* nltk
* pandas
* tqdm
* seaborn
* wordcloud
* math


## Описание проектово
### Анализ продаж видео игр
Данная работа была аналитического характера. Много работы с pivot_table, query. 
Так же были выдвинуты гипотезы и проведены статистические тесты (ttest_ind) для проверки равенства средних значений двух выборок.
Так же было проведено сравнение дисперсий двух выборок
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

### Выбор наиболее прибыльной скважины
В данном проекте у нас были пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Необходимо было построить модель машинного обучения для определения наиболее прибыльного региона. 
![Anomalies in concentrates of elements]https://github.com/ilart/DS_course/blob/main/media/Density%20of%20elements%20concentrates.png

Здесь я применил Bootstrap. Т.к target значений было мало и данные не были распределены нормально
```
    for _ in range(1000):
        target_valid_sampled = target_valid[i].sample(n=500, replace = True, random_state=state)
        prediction_sampled = predicted_series[i][target_valid_sampled.index]
        income_values[i].append(revenue(target_valid_sampled, prediction_sampled, 200))
    
    income_values[i] = pd.Series(income_values[i])
    lower = income_values[i].quantile(q=0.025)
    upper = income_values[i].quantile(q=0.975)
```
![risks of nonprofit investments]https://github.com/ilart/DS_course/blob/main/media/Density%20of%20incomes%20with%20qantiles.png?raw=true

### Обогащение золота
В данном проекте необходимо было предсказать коэфициент восстановления золота из руды. Сложный проект тем что надо разобраться в процессе обогащения. Понять суть всех этапов (флотация, несколько стадий фильтрации ), отсеять лишние признаки. А также необходимо было применить кастомную скор-функцию для оценки качества прогноза коэфициента. 
![heatmap]https://github.com/ilart/DS_course/blob/main/media/gold_recovery_heatmap.png?raw=true


### Машинное обучение для текстов
Проект по классификации текстов по признаку токсичности. 
Основные этапы:
- очистка текста от лишних символов + лемматизация + удаление стоп-слов
- векторизация текста с помощью TfIdf
- борьба с дизбалансом
- Тюнинг моделей
```
parametrs = { 'max_depth': range (150, 200, 50)}
              'min_samples_leaf': range (2,14, 2)}
              'min_samples_split': range (2,8,2) }
              
model = DecisionTreeClassifier(random_state=12345)
grid = GridSearchCV(model, parametrs, scoring="f1", n_jobs=-1, verbose = 3, cv=3)
grid.fit(train_features, train_target)
```
- обучение различных моделей и сравнение результатов
![Итоговое сравнение моделей]https://github.com/ilart/DS_course/blob/main/media/models_comparing.png?raw=true
