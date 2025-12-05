import os
import pandas as pd

class Global_functions:
    def __init__(self):
        pass

    def seperate(self,num):
        import time
        for i in range (0,num+1):
            print("-",end="",flush=False)
            time.sleep(0.01)



class ML:
    def __init__(self):
        pass
    def auto_clean_df(self,df, to_predict):
    
        import pandas as pd 
        import numpy as np
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        import os
    
    
        # Step 0: Preserve target column FIRST (before any modifications)
        if to_predict not in df.columns:
            raise ValueError(f"Target column '{to_predict}' not found")
        
        target_col = df[to_predict].copy()  # Safe copy of original target
        df = df.drop(columns=[to_predict])  # Remove target from cleaning
        
        pd.set_option('future.no_silent_downcasting', True)
        
        # 1. Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # 2. Handle Missing Values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # 3. Detect categorical / numeric (after cleaning)
        categorical_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # 4. Encode categorical columns
        for col in categorical_cols:
            if df[col].nunique() <= 10:
                # One-hot encode
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            else:
                # Label encode
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        # 5. Convert booleans to int (after get_dummies)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].replace({True: 1, False: 0})
        
        # 6. Scale numerical columns (update numeric_cols first)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 7. Restore original target column
        df[to_predict] = target_col
        
        return df
    
    def Linear_Regression(self,DATA,to_predict):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        df = pd.read_csv(DATA)
        X = df.drop(to_predict,axis=1)
        Y = df[to_predict]
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


        model = LinearRegression()
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_test)
        
        r2 = r2_score(Y_test,Y_pred)

        n = X_test.shape[0]         #Rows
        p = X_test.shape[1]         #Coluns

        adjusted_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))          #fixed formula
        adjusted_r2



    def CLI(self):
        # Data processing
        ML_inst = ML()
        CURRENT_DIR = os.getcwd()
        print(f"\n\ncurrently in dir : {CURRENT_DIR}\n")
        i=True
        while i == True:
            try:
                LOCATION = str(input("Enter your data file path : "))
                raw_df = pd.read_csv(os.path.join(CURRENT_DIR, LOCATION))
                i = False
            except FileNotFoundError as e:
                i = True

        print(f"\nColumns: {list(raw_df.columns)}\n")

        predict_value = str(input("Predict value : "))
        df = ML_inst.auto_clean_df(raw_df, to_predict=predict_value)
        print(f"Cleaned df : \n{df.head()}")

        save_to = str(input(f"\n\nYou are in {CURRENT_DIR}\n\nSave it to (ENTER A VALID DIR LOCATION WITH NAME OF THE FILE): "))
        df.to_csv(os.path.join(CURRENT_DIR, save_to), index=False)
        print(f"Saved to: {os.path.join(CURRENT_DIR, save_to)}")

def main():
    ML_instance = ML()
    ML_instance.CLI()
    return 0

if __name__ == "__main__":
    main()
