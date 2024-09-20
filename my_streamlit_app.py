import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st

# Load and preprocess data
df = pd.read_csv("/content/Financial_inclusion_dataset.csv")
df.dropna(inplace=True)

# Ensure the uniqueid column is dropped
if 'uniqueid' in df.columns:
    df.drop(columns=['uniqueid'], inplace=True)

# Define features and target
target = df['bank_account']
features = df.drop(columns=['bank_account'])

# Encode categorical features
categorical_columns = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                       'relationship_with_head', 'marital_status', 'education_level', 'job_type']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])

X = features
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model and label encoders
with open('financial_inclusion_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Streamlit app
st.set_page_config(page_title="Financial Inclusion Prediction App", layout="wide")

st.title('Financial Inclusion Prediction App')
st.markdown(
    """
    This application predicts whether an individual is likely to have a bank account based on various demographic features. Enter individual details below to get a prediction.
    """
)

st.sidebar.header('Input Individual Data')

# Sidebar inputs
country = st.sidebar.selectbox('Country', df['country'].unique())
year = st.sidebar.selectbox('Year', df['year'].unique())
location_type = st.sidebar.selectbox('Location Type', df['location_type'].unique())
cellphone_access = st.sidebar.selectbox('Cellphone Access', df['cellphone_access'].unique())
gender_of_respondent = st.sidebar.selectbox('Gender of Respondent', df['gender_of_respondent'].unique())
relationship_with_head = st.sidebar.selectbox('Relationship with Head', df['relationship_with_head'].unique())
marital_status = st.sidebar.selectbox('Marital Status', df['marital_status'].unique())
education_level = st.sidebar.selectbox('Education Level', df['education_level'].unique())
job_type = st.sidebar.selectbox('Job Type', df['job_type'].unique())
age_of_respondent = st.sidebar.slider('Age of Respondent', int(df['age_of_respondent'].min()), int(df['age_of_respondent'].max()), int(df['age_of_respondent'].mean()))
household_size = st.sidebar.slider('Household Size', int(df['household_size'].min()), int(df['household_size'].max()), int(df['household_size'].mean()))

if st.sidebar.button('Predict Financial Inclusion'):
    # Create the input_data DataFrame with the same columns and order as the training data
    input_data = pd.DataFrame([{
        'country': country,
        'year': year,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'gender_of_respondent': gender_of_respondent,
        'relationship_with_head': relationship_with_head,
        'marital_status': marital_status,
        'education_level': education_level,
        'job_type': job_type,
        'age_of_respondent': age_of_respondent,
        'household_size': household_size
    }])

    # Reorder input_data to match the order of the training features
    input_data = input_data[features.columns]

    # Apply the same label encoding to the input data
    for column in categorical_columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Load the model and make a prediction
    with open('financial_inclusion_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(input_data)
    
    st.write(f'### Prediction: {"Has a bank account" if prediction[0] == 1 else "Does not have a bank account"}')
