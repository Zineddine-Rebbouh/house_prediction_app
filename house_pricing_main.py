import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

#  load data function 
def load_data(data_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded from {data_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {data_path} not found.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
def preprocess_data(df):
 
    """
    Preprocess the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
    """
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Create interaction terms for highly correlated features
    df['rooms_per_dwelling_squared'] = df['avg_rooms_per_dwelling'] ** 2
    df['lower_status_squared'] = df['lower_status_pct'] ** 2

    # Create distance-based feature
    df['dist_to_employment_scaled'] = np.log1p(df['distance_to_employment'])

    # Create ratio features
    df['price_to_tax_ratio'] = df['property_tax_rate'] / df["median_home_value_k"]
    df['crime_to_distance_ratio'] = df['crime_rate'] / df['distance_to_employment']
    
    top_features = [
        'avg_rooms_per_dwelling', 'lower_status_pct', 'distance_to_employment',
        'property_tax_rate', 'median_home_value_k', 'crime_rate'
    ]
    
    # Drop the non-top features
    df.drop(columns=[col for col in df.columns if col not in top_features], inplace=True, errors='ignore')

    
    return df

# Function to preprocess the data
def scale_features(X_train , X_test):
    """
    Scale numerical features using StandardScaler.
    """
    
    # Initialize the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
      
    # Log the scaler
    mlflow.sklearn.log_model(scaler, "scaler")
    return X_train_scaled, X_test_scaled, scaler    

# Train the model 
def train_model(X_train, y_train):
    
    """
    Train a machine learning model using the training data.
    """
    # Initialize the model
    model = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Log model parameters
    mlflow.log_param("n_estimators", 180)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 3)
    
    return model
    

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Log metrics to MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("root_mean_squared_error", rmse)
    
    # Print the evaluation metrics
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    



def main(args):    
    
    # Main function to run the Boston Housing Price Prediction pipeline.
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("boston_housing_experiment")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    with mlflow.start_run(run_name="boston_housing_model_training"):
        # Load the dataset
        df = load_data( "./datasets/boston_housing.csv")

        # Preprocess the data
        df = preprocess_data(df)
        
        print(df.columns)
        # Split features and target
        X = df.drop(columns=['median_home_value_k'])
        y = df['median_home_value_k']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)


        # Train the model
        model = train_model(X_train_scaled, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test_scaled, y_test)
        
        # Print success message
        print("Model training and evaluation completed successfully.")

        # Save the model
        joblib.dump(model, "./models/boston_housing_model.pkl")    

        # Log the model to MLflow
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.sklearn.log_model(model, name="boston_housing_model" , registered_model_name="BostonHousingModel" )
        mlflow.log_artifact(local_path="./models/boston_housing_model.pkl", artifact_path="models")

        # Print success message
        print("Model saved and logged to MLflow.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing Price Prediction Pipeline")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    args = parser.parse_args()
    
    main(args)