import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Data Ingestion (قراءة البيانات)
df = pd.read_csv('netflix.csv')

# 2. Data Cleaning (تنقية البيانات)
# كنعمرو البلايص الخاوين (Missing Values)
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('UR')

# 3. Feature Engineering (تحويل البيانات لأرقام)
le_type = LabelEncoder()
le_rating = LabelEncoder()
le_country = LabelEncoder()

# تحويل 'type' لـ Target (Movie=0, TV Show=1)
df['target'] = le_type.fit_transform(df['type'])

# تحويل البيانات النصية لأرقام باش الموديل يفهم
df['rating_encoded'] = le_rating.fit_transform(df['rating'])
df['country_encoded'] = le_country.fit_transform(df['country'])

# اختيار الـ Features لي غانخدمو بيهم
X = df[['release_year', 'rating_encoded', 'country_encoded']]
y = df['target']

# 4. Machine Learning (تدريب الموديل)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print(f"L'accuracy du modèle est: {accuracy * 100:.2f}%")

# 5. Saving (حفظ الموديل والمحولون باش نخدمو بيهم فـ App)
joblib.dump(model, 'netflix_model.pkl')
joblib.dump(le_rating, 'le_rating.pkl')
joblib.dump(le_country, 'le_country.pkl')
joblib.dump(le_type, 'le_type.pkl')

print("Le modèle a été entraîné et sauvegardé avec succès!")