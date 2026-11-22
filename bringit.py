import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from scipy import stats

    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.compose import make_column_transformer, ColumnTransformer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
    from catboost import CatBoostRegressor
    from sklearn.kernel_ridge import KernelRidge


    return np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv('hp/train.csv')
    test_df = pd.read_csv('hp/test.csv')
    df.shape, test_df.shape, df.head()
    return df, test_df


@app.cell
def _(df, plt):

    df['SalePrice'].hist(bins = 50)
    plt.title('распределение saleprice')
    plt.show()

    print(f"min: {df['SalePrice'].min()}")
    print(f"median: {df['SalePrice'].median()}")
    print(f"max: {df['SalePrice'].max()}")
    return


@app.cell
def _(df, np, plt):
    df['SalePrice'].apply(np.log1p).hist(bins = 50)
    plt.title('saleprice after log(1+x)')
    plt.show()
    return


@app.cell
def _(df, test_df):
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id']
    df_clean = df.drop(columns=cols_to_drop)
    test_clean = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    print(f"колонок train до: {df.shape[1]}")

    print(f"колонок train осталось: {df_clean.shape[1]}")
    print(f"колонок test осталось: {test_clean.shape[1]}")
    return df_clean, test_clean


@app.cell
def _(df_clean):
    missing = df_clean.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing
    return


@app.cell
def _(df_clean, np, test_clean):
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include = 'str').columns.tolist()


    df_processed = df_clean.copy()
    test_processed = test_clean.copy()

    df_processed['LotFrontage'] = df_processed['LotFrontage'].fillna(df_processed['LotFrontage'].median())
    test_processed['LotFrontage'] = test_processed['LotFrontage'].fillna(df_processed['LotFrontage'].median())

    df_processed['MasVnrArea'] = df_processed['MasVnrArea'].fillna(0)
    test_processed['MasVnrArea'] = test_processed['MasVnrArea'].fillna(0)

    df_processed['GarageYrBlt'] = df_processed['GarageYrBlt'].fillna(0)
    test_processed['GarageYrBlt'] = test_processed['GarageYrBlt'].fillna(0)

    electrical_mode = df_processed['Electrical'].mode()[0]
    df_processed['Electrical']=df_processed['Electrical'].fillna(electrical_mode)
    test_processed['Electrical']=test_processed['Electrical'].fillna(electrical_mode)

    for col in cat_cols:
        df_processed[col] = df_processed[col].fillna('Missing')
        test_processed[col] = test_processed[col].fillna('Missing')

    # Заполняем оставшиеся числовые пропуски в test нулём
    # Логика та же: нет подвала/гаража = площадь и количество равны 0
    for col in num_cols:
        if col == 'SalePrice':
            continue
        if test_processed[col].isnull().sum() > 0:
            test_processed[col] = test_processed[col].fillna(0)

    print(f"Пропусков в test: {test_processed.isnull().sum().sum()}")
    print(f"Пропусков в train: {df_processed.isnull().sum().sum()}")
    print(f"Пропусков в test: {test_processed.isnull().sum().sum()}")
    return df_processed, test_processed


@app.cell
def _(df_processed, pd, test_processed):
    train_size = df_processed.shape[0]

    target = df_processed['SalePrice']
    df_no_target = df_processed.drop(columns=['SalePrice'])

    combined = pd.concat([df_no_target, test_processed], axis=0)

    combined_encoded = pd.get_dummies(combined)
    print(f"размер после кодирования: {combined_encoded.shape}")

    train_encoded = combined_encoded.iloc[:train_size, :]
    test_encoded = combined_encoded.iloc[train_size:, :]

    print(f"Train: {train_encoded.shape}")
    print(f"Test: {test_encoded.shape}")
    return target, test_encoded, train_encoded


@app.cell
def _(target, train_encoded):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X = train_encoded
    y = target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print(f"Обучающая выборка: {X_train.shape}")
    print(f"валидационная выборка: {X_val.shape}")
    return X_train, X_val, mean_squared_error, y_train, y_val


@app.cell
def _(X_train, X_val, mean_squared_error, np, y_train, y_val):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, prediction))
    print(f"rmse: {rmse:.0f}")
    return (RandomForestRegressor,)


@app.cell
def _(RandomForestRegressor, np, target, train_encoded):
    from sklearn.model_selection import cross_val_score
    y_log = np.log1p(target)

    model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model_cv, train_encoded, y_log,
                             cv = 5,
                             scoring = 'neg_root_mean_squared_error')

    rmse_log = np.abs(scores).mean()
    print(f"cv rmse (log): {rmse_log:.4f}")
    return cross_val_score, y_log


@app.cell
def _(RandomForestRegressor, np, pd, test_encoded, train_encoded, y_log):
    # обучаем финальную модель на ВСЕХ тренировочных данных

    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(train_encoded, y_log)  # y_log это наш логарифмированный таргет

    # Предсказываем цены для test
    predictions_log = final_model.predict(test_encoded)

    # обратное преобразование
    # np.expm1 это обратная операция к np.log1p
    predictions_final = np.expm1(predictions_log)

    # cоздаём файл для сабмита в формате который ожидает Kaggle
    submission = pd.DataFrame({
        'Id': pd.read_csv('hp/test.csv')['Id'],  # берём Id из оригинального test
        'SalePrice': predictions_final
    })

    submission.to_csv('submission.csv', index=False)
    print(submission.head())
    print(f"Строк в сабмите: {submission.shape[0]}")
    return


@app.cell
def _(cross_val_score, np, train_encoded, y_log):
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators = 1000,
        learning_rate = 0.05,
        max_depth = 4,
        random_state = 42,
        n_jobs = -1
    )
    scores_xgb = cross_val_score(xgb_model, train_encoded, y_log, cv = 5,
                            scoring='neg_root_mean_squared_error')

    rmse_xgb = np.abs(scores_xgb).mean()
    print(f"xgb cv rmse: {rmse_xgb:.4f}")
    print(f"random forest cv rmse: 0.1436")
    print(f"разница: {0.1436 - rmse_xgb:.4f}")
    return (XGBRegressor,)


@app.cell
def _(XGBRegressor, np, pd, test_encoded, train_encoded, y_log):
    final_xgb = XGBRegressor(
        n_estimators = 1000,
        learning_rate = 0.05,
        max_depth = 4,
        random_state = 42,
        n_jobs = -1
    )
    final_xgb.fit(train_encoded, y_log)
    preds_log = final_xgb.predict(test_encoded)
    preds_final = np.expm1(preds_log)
    submission_xgb = pd.DataFrame({
        'Id': pd.read_csv('hp/test.csv')['Id'],
        'SalePrice': preds_final
    })

    submission_xgb.to_csv('submission_xgb.csv', index=False)
    print("Готово!")
    return


if __name__ == "__main__":
    app.run()
