import pandas as pd
from scripts.preprocess import machine_makers, most_similar, df_concat_clean
from scripts.config import *

brand_res = {}


def most_similar_brand(x):
    if x in brand_res:
        return brand_res[x]
    brand_res[x] = most_similar(machine_makers, x)
    return brand_res[x]


def feature_enginering(df):
    print("similar")
    df = df.copy()
    df[COL_Car_Brand] = df[COL_Car_Name].apply(most_similar_brand)
    print("drop")
    df = df.drop(COL_Car_Name, axis=1)
    df[COL_diff_year] = 2026 - df[COL_Year]
    return df


def split_train_test(df, test_size=0.2, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def standardize_numeric(train_df, test_df, columns):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for column in columns:
        mean = train_df[column].mean()
        std = train_df[column].std(ddof=0)
        if std == 0 or pd.isna(std):
            std = 1.0
        train_df[column] = (train_df[column] - mean) / std
        test_df[column] = (test_df[column] - mean) / std

    return train_df, test_df


def one_hot_encode(train_df, test_df, columns):
    train_encoded = pd.get_dummies(train_df, columns=columns, drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=columns, drop_first=True)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, test_encoded


print("go")
df_feat = feature_enginering(df_concat_clean)

X = df_feat.drop(COL_Sel_Price, axis=1)
y = df_feat[COL_Sel_Price]

X_train, X_test = split_train_test(X, test_size=0.2, random_state=42)
y_train, y_test = split_train_test(y.to_frame(name=COL_Sel_Price), test_size=0.2, random_state=42)
y_train = y_train[COL_Sel_Price]
y_test = y_test[COL_Sel_Price]

numerical_cols = [COL_Year, COL_Pres_Price, COL_Kms_Driven, COL_Owner, COL_diff_year]
categorical_cols = [COL_Fuel_Type, COL_Trans, COL_Sell_Type, COL_Car_Brand]

X_train_transformed, X_test_transformed = standardize_numeric(X_train, X_test, numerical_cols)
X_train_transformed, X_test_transformed = one_hot_encode(
    X_train_transformed,
    X_test_transformed,
    categorical_cols,
)

df_train_transformed = X_train_transformed
df_test_transformed = X_test_transformed

print(df_train_transformed.head(10))
print("Train shape:", df_train_transformed.shape)
print("Test shape:", df_test_transformed.shape)
