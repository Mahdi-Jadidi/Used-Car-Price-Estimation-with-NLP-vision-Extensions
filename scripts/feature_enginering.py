import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scripts.preprocess import machine_makers, most_similar, df_concat_clean
from scripts.config import *
brand_res = {}
def most_similar_brand(x):
    if(x in brand_res):
        return brand_res[x]
    brand_res[x] = most_similar(machine_makers, x)
    return brand_res[x]

def feature_enginering(df):
    print('similar')
    df[COL_Car_Brand] = df[COL_Car_Name].apply(most_similar_brand)
    print('drop')
    df = df.drop(COL_Car_Name, axis=1)
    df[COL_diff_year] = 2026 - df[COL_Year]
    return df
print('go')
df_feat = feature_enginering(df_concat_clean)

X = df_feat.drop(COL_Sel_Price, axis=1)
y = df_feat[COL_Sel_Price]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_cols = [COL_Year, COL_Pres_Price, COL_Kms_Driven, COL_Owner, COL_diff_year]
categorical_cols = [COL_Fuel_Type, COL_Trans, COL_Sell_Type, COL_Car_Brand]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
])

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = numerical_cols + list(cat_names)

df_train_transformed = pd.DataFrame(X_train_transformed, columns=all_features)
df_test_transformed = pd.DataFrame(X_test_transformed, columns=all_features)
print(df_train_transformed.head(10))
print("Train shape:", df_train_transformed.shape)
print("Test shape:", df_test_transformed.shape)
