# === Imports === 
import pandas as pd 
import re 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC 
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import accuracy_score, classification_report 
 
# === Load Dataset === 
file_path = r"C:\\Users\\user\\Desktop\\pt\\data_cleaned2.xlsx" 
data = pd.read_excel(file_path) 
 
print(" Data uploaded successfully!") 
print("First five rows:\n", data.head(), "\n") 
 
# === Validate Required Columns === 
expected_cols = ['ingredients', 'cuisine'] 
for col in expected_cols: 
    if col not in data.columns: 
        raise ValueError(f"Column '{col}' not found. Found: {list(data.columns)}") 
 
# === Clean Text Function === 
def clean_text(text): 
    text = str(text).lower() 
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # remove numbers and punctuation 
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces 
    return text 
 
data['ingredients'] = data['ingredients'].apply(clean_text) 
 
# === Drop Missing Values === 
data = data.dropna(subset=['ingredients', 'cuisine']) 
 
# === Remove Rare Cuisines === 
class_counts = data['cuisine'].value_counts() 
data = data[data['cuisine'].isin(class_counts[class_counts >= 2].index)] 
 
print(f"Cuisines after filtering: {data['cuisine'].nunique()}") 
 
# === Features & Labels === 
X = data['ingredients'] 
y = data['cuisine'] 
 
# === Train-Test Split === 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y 
) 
 
# === Pipeline: TF-IDF + SVM === 
model = make_pipeline( 
    TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2), 
    LinearSVC() 
) 
 
# === Train Model === 
model.fit(X_train, y_train) 
 
# === Evaluate Model === 
y_pred = model.predict(X_test) 
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2) 
print(f"\n Improved Accuracy: {accuracy}%\n") 
 
print("Classification Report:\n") 
print(classification_report(y_test, y_pred)) 
 
# === Example Prediction === 
example = "egg, whole wheat bread crumbs, onion flakes, italian seasoning, seasoning salt, garlic sea salt, chili pepper flakes, kosher salt 26 freshly ground black pepper, ground turkey, mozzarella cheese, basil olive oil, butter, yellow onion, mushrooms, garlic cloves, tomato paste, tomato sauce, merlot, kosher salt 26 freshly ground black pepper" 
prediction = model.predict([example])[0] 
print(f"\nExample ingredients: {example}") 
print(f"Predicted cuisine: {prediction}")
# === User Query: Search by Dish Name ===
# === Loop: User can search multiple times ===
dish_column = "name"  # عدّل الاسم حسب عمود الوجبة في ملفك

if dish_column not in data.columns:
    raise ValueError(f"Column '{dish_column}' not found in dataset.")

while True:
    dish_name = input("\nEnter dish name (or type 'exit' to quit): ").strip()
    
    if dish_name.lower() == "exit":
        print("Exiting...")
        break

    search_value = clean_text(dish_name)

    result = data[data[dish_column].str.lower().apply(clean_text) == search_value]

    if result.empty:
        print("Dish not found. Try again.")
    else:
        row = result.iloc[0]
        print("\n=== Dish Information ===")
        print("Dish Name:", row[dish_column])
        print("Ingredients:", row['ingredients'])
        print("Country (Cuisine):", row['cuisine'])