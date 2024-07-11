import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tkinter as tk
from tkinter import ttk, messagebox

# Load dataset with the correct delimiter and quote character
df = pd.read_csv('bank.csv', delimiter=';', quotechar='"')

# Identify numerical and categorical columns
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Define preprocessing for numerical columns (impute missing values and scale)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns (impute missing values and one-hot encode)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to the feature columns and create the target variable
X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert target to binary format

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)


# Function to validate and convert input to integer
def validate_int(input_str):
    try:
        return int(input_str.replace(',', ''))
    except ValueError:
        return None


# Function to clear placeholder text on focus
def clear_placeholder(event, entry, placeholder):
    if entry.get() == placeholder:
        entry.delete(0, tk.END)
        entry.config(foreground='black')


# Function to add placeholder text on focus out
def add_placeholder(event, entry, placeholder):
    if entry.get() == '':
        entry.insert(0, placeholder)
        entry.config(foreground='grey')


# Function to predict credit score based on user input
def predict():
    # Get user input and validate
    input_data = {
        'age': validate_int(age_entry.get()),
        'balance': validate_int(balance_entry.get()),
        'day': validate_int(day_entry.get()),
        'duration': validate_int(duration_entry.get()),
        'campaign': validate_int(campaign_entry.get()),
        'pdays': validate_int(pdays_entry.get()),
        'previous': validate_int(previous_entry.get()),
        'job': job_combobox.get(),
        'marital': marital_combobox.get(),
        'education': education_combobox.get(),
        'default': default_combobox.get(),
        'housing': housing_combobox.get(),
        'loan': loan_combobox.get(),
        'contact': contact_combobox.get(),
        'month': month_combobox.get(),
        'poutcome': poutcome_combobox.get()
    }

    # Check for any invalid inputs
    if None in input_data.values():
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Display the result
    result = "Creditworthy" if prediction == 1 else "Not Creditworthy"
    messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {prediction_proba:.2f}")


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x = event.x_root + 20
        y = event.y_root + 10
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, background="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# Create the GUI
root = tk.Tk()
root.title("Credit Scoring Model")

# Create input fields with placeholders
placeholders = ["Enter Age", "Enter Balance (e.g., 30,000)", "Enter Day", "Enter Duration",
                "Enter Campaign", "Enter Pdays", "Enter Previous"]

entries = []

tooltips = [
    "Age of the individual",
    "Balance amount in the account",
    "Day of the month",
    "Duration of the last contact",
    "Number of contacts performed during this campaign",
    "Number of days since the client was last contacted",
    "Number of contacts performed before this campaign"
]

for i, placeholder in enumerate(placeholders):
    label = ttk.Label(root, text=f"{placeholder.split()[1]}:")
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = ttk.Entry(root, foreground='grey')
    entry.insert(0, placeholder)
    entry.bind("<FocusIn>", lambda event, e=entry, p=placeholder: clear_placeholder(event, e, p))
    entry.bind("<FocusOut>", lambda event, e=entry, p=placeholder: add_placeholder(event, e, p))
    entry.grid(row=i, column=1, padx=10, pady=5)
    Tooltip(label, tooltips[i])
    entries.append(entry)

age_entry, balance_entry, day_entry, duration_entry, campaign_entry, pdays_entry, previous_entry = entries

ttk.Label(root, text="Job:").grid(row=7, column=0, padx=10, pady=5)
job_combobox = ttk.Combobox(root, values=df['job'].unique())
job_combobox.grid(row=7, column=1, padx=10, pady=5)

ttk.Label(root, text="Marital:").grid(row=8, column=0, padx=10, pady=5)
marital_combobox = ttk.Combobox(root, values=df['marital'].unique())
marital_combobox.grid(row=8, column=1, padx=10, pady=5)

ttk.Label(root, text="Education:").grid(row=9, column=0, padx=10, pady=5)
education_combobox = ttk.Combobox(root, values=df['education'].unique())
education_combobox.grid(row=9, column=1, padx=10, pady=5)

ttk.Label(root, text="Default:").grid(row=10, column=0, padx=10, pady=5)
default_combobox = ttk.Combobox(root, values=df['default'].unique())
default_combobox.grid(row=10, column=1, padx=10, pady=5)

ttk.Label(root, text="Housing:").grid(row=11, column=0, padx=10, pady=5)
housing_combobox = ttk.Combobox(root, values=df['housing'].unique())
housing_combobox.grid(row=11, column=1, padx=10, pady=5)

ttk.Label(root, text="Loan:").grid(row=12, column=0, padx=10, pady=5)
loan_combobox = ttk.Combobox(root, values=df['loan'].unique())
loan_combobox.grid(row=12, column=1, padx=10, pady=5)

ttk.Label(root, text="Contact:").grid(row=13, column=0, padx=10, pady=5)
contact_combobox = ttk.Combobox(root, values=df['contact'].unique())
contact_combobox.grid(row=13, column=1, padx=10, pady=5)

ttk.Label(root, text="Month:").grid(row=14, column=0, padx=10, pady=5)
month_combobox = ttk.Combobox(root, values=df['month'].unique())
month_combobox.grid(row=14, column=1, padx=10, pady=5)

ttk.Label(root, text="Poutcome:").grid(row=15, column=0, padx=10, pady=5)
poutcome_combobox = ttk.Combobox(root, values=df['poutcome'].unique())
poutcome_combobox.grid(row=15, column=1, padx=10, pady=5)

# Create predict button
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=16, columnspan=2, pady=10)

root.mainloop()
