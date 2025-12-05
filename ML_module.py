import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import sys
import time

class Global_functions:
    @staticmethod
    def separate(num=30, style="-"):
        print()
        for _ in range(num):
            sys.stdout.write(style)
            sys.stdout.flush()
            time.sleep(0.01)
        print("\n")

class ModelReadyPreprocessor:
    def __init__(self):
        self.gf = Global_functions()
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def clean_data(self, df):
        print("... [1/5] Cleaning duplicates and missing values")
        df = df.copy()

        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_rows:
            print(f"    - Removed {initial_rows - len(df)} duplicate rows.")

        df.dropna(how='all', axis=1, inplace=True)

        return df

    def process_for_model(self, df, target_col):
        print(f"... [2/5] Processing features for Target: '{target_col}'")
        df = df.copy()

        # --- STEP A: SEPARATE TARGET ---
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # --- STEP B: PROCESS TARGET (y) ---
        is_regression = pd.api.types.is_numeric_dtype(y)
        if not is_regression:
            print(f"    - Target '{target_col}' is categorical. Label Encoding...")
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            y = pd.Series(y, name=target_col)
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"    - Target Mapping: {mapping}")

        # --- STEP C: PROCESS FEATURES (X) ---

        # 1. Identify types
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        bool_cols = X.select_dtypes(include=['bool']).columns.tolist()

        # 2. Handle Numeric Features
        if num_cols:
            print(f"    - Scaling {len(num_cols)} numeric features...")
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            X[num_cols] = self.scaler.fit_transform(X[num_cols])

        # 3. Handle Boolean Features (True/False -> 1/0)
        if bool_cols:
            print(f"    - Converting {len(bool_cols)} boolean features to 1/0...")
            for col in bool_cols:
                X[col] = X[col].astype(int)

        # 4. Handle Categorical Features (One-Hot Encoding)
        if cat_cols:
            print(f"    - One-Hot Encoding {len(cat_cols)} categorical features...")
            for col in cat_cols:
                if X[col].isnull().any():
                    mode_val = X[col].mode()[0]
                    X[col] = X[col].fillna(mode_val)

            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

        # --- STEP D: RECOMBINE ---
        X.reset_index(drop=True, inplace=True)
        y = pd.Series(y, name=target_col).reset_index(drop=True)

        final_df = pd.concat([X, y], axis=1)

        return final_df, is_regression

    def train_and_evaluate_model(self, df_model_ready, target_col):
        print("... [4/5] Training simple Linear Regression Model")

        # A. Split Data
        X = df_model_ready.drop(columns=[target_col])
        y = df_model_ready[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"    - Data split: Train={len(X_train)} rows, Test={len(X_test)} rows.")

        # B. Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # C. Predict and Evaluate
        y_pred = model.predict(X_test)

        print("\n--- Model Performance Metrics (Regression) ---")
        print(f"R-squared (R2): {r2_score(y_test, y_pred):.4f}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
        print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print("---------------------------------------------")

    def CLI(self):
            self.gf.separate(40, "=")
            print("  UNIVERSAL ML DATA PREPARER & REGRESSION TEST  ")
            self.gf.separate(40, "=")
            
            current_dir = os.getcwd()
            print(f"Working Directory: {current_dir}\n")
    
            # 1. Load File (REVISED for exit option)
            raw_df = None
            while raw_df is None:
                f_in = input(">> [1/5] Enter input CSV filename (or 'q' to quit): ").strip()
                if f_in.lower() == 'q':
                    print("Exiting application.")
                    return
    
                path = os.path.join(current_dir, f_in)
                if os.path.exists(path) and f_in.endswith('.csv'):
                    try:
                        raw_df = pd.read_csv(path)
                        print(f"Loaded successfully: {raw_df.shape}\n")
                    except Exception as e:
                        print(f"Error loading file: {e}")
                else:
                    print("File not found or is not a CSV. Try again.")
    
            # --- STEP 2: SELECT TARGET (REVISED for data info and validation) ---
            self.gf.separate(30, ".")
            print("Data Overview:")
            
            # Display key info about the dataframe for target selection
            info_df = pd.DataFrame({
                'Dtype': raw_df.dtypes.astype(str),
                'Unique': raw_df.nunique(),
                'Nulls': raw_df.isnull().sum()
            })
            print(info_df)
            self.gf.separate(30, ".")
    
            target = ""
            while target not in raw_df.columns:
                print("\nAvailable Columns:")
                for i, col in enumerate(raw_df.columns, 1):
                    print(f"  {i}. {col} ({info_df.loc[col, 'Dtype']})")
    
                target = input("\n>> [2/5] Enter TARGET column name (prediction goal): ").strip()
                
                if target not in raw_df.columns:
                    print("Column not found in data! Please choose from the list above.")
                else:
                    # Basic validation: ensure target is not a constant value
                    if raw_df[target].nunique() <= 1:
                        print(f"Target '{target}' has only {raw_df[target].nunique()} unique value(s). Please select a variable target.")
                        target = "" # Reset target to continue the loop
    
            # 3. Process
            try:
                self.gf.separate(30, ".")
                df_clean = self.clean_data(raw_df)
                df_model_ready, is_regression = self.process_for_model(df_clean, target)
                self.gf.separate(30, ".")
    
                # 4. Preview
                target_type = "Regression (Numeric)" if is_regression else "Classification (Categorical)"
                print(f"\nTarget Type Detected: **{target_type}**")
                print("Preview of Model-Ready Data (Features Scaled/Encoded):")
                print(df_model_ready.head())
                print(f"\nFinal Shape: {df_model_ready.shape}")
                print(f"Total Columns Created: {len(df_model_ready.columns)}")
    
                # 5. Run Model (Conditional on Target Type and User Consent)
                if is_regression:
                    model_consent = input(f"\n>> [4/5] Run simple Linear Regression test on '{target}'? (y/n): ").strip().lower()
                    if model_consent == 'y':
                        self.gf.separate(30, ".")
                        self.train_and_evaluate_model(df_model_ready, target)
                        self.gf.separate(30, ".")
                    else:
                        print("Skipping model evaluation.")
                else:
                    print(f"\nSkipping regression model: Target '{target}' is categorical.")
                    
                # 6. Save
                default_name = f"model_ready_{target}.csv"
                save_name = input(f"\n>> [5/5] Save preprocessed data as (Press Enter for '{default_name}'): ").strip()
                if not save_name:
                    save_name = default_name
                
                if not save_name.endswith('.csv'):
                    save_name += ".csv"
    
                save_path = os.path.join(current_dir, save_name)
                df_model_ready.to_csv(save_path, index=False)
                
                self.gf.separate(40, "=")
                print(" **SUCCESS!** ")
                print(f"Model-Ready File Saved: {save_path}")
                print(f"Ready for `model.fit(X, y)` using this file.")
                self.gf.separate(40, "=")
    
            except Exception as e:
                self.gf.separate(40, "!")
                print(f"\n **CRITICAL ERROR** DURING PROCESSING: {e}")
                self.gf.separate(40, "!")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    app = ModelReadyPreprocessor()
    app.CLI()