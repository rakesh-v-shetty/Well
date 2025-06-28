import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Removed PolynomialFeatures as not actively used in pipeline, but keep if you plan to use it
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC # SVC can be computationally expensive without specific tuning; commented out for faster execution.
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.feature_selection import SelectKBest, f_classif # Not actively used for numerical in final preprocessor but kept if needed for other analysis
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pickle # Import the pickle library

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('Synthetic_Water_Well_Prediction_Data.csv')
    print("Dataset loaded successfully.")
    print(df.head())
    print(df.info())
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'Synthetic_Water_Well_Prediction_Data.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- 2. Feature Engineering and Data Analysis ---

# Target variable for Well Suitability
y = df['Well_Suitability_Binary']

# Create additional engineered features for better prediction
df_engineered = df.copy()

# Feature 1: Depth to Discharge Ratio (efficiency indicator)
df_engineered['Depth_to_Discharge_Ratio'] = df_engineered['Well_Depth_Actual_m'] / (df_engineered['Well_Discharge_LPM'] + 1)

# Feature 2: Water Quality Category (based on TDS levels)
# Ensure consistent binning as this will be used during prediction
df_engineered['Water_Quality_Category'] = pd.cut(df_engineered['Water_Quality_TDS_mg_L'],
                                                  bins=[0, 500, 1000, 1500, float('inf')],
                                                  labels=['Excellent', 'Good', 'Fair', 'Poor'],
                                                  right=True,
                                                  include_lowest=True) # include_lowest for 0-500 range

# Feature 3: Groundwater Stress Index
df_engineered['Groundwater_Stress'] = df_engineered['Stage of Ground Water Extraction (%)'] / 100

# Feature 4: Recharge to Availability Ratio
df_engineered['Recharge_Availability_Ratio'] = (df_engineered['Total Annual Ground Water Recharge'] /
                                                  (df_engineered['Net Ground Water Availability for future use'] + 1))

# Feature 5: Regional Well Density (simplified approximation)
# Calculate location_counts which will be saved for the Flask app
location_counts = df_engineered.groupby(['Name of State', 'Name of District']).size().reset_index(name='Regional_Well_Density_Count')
df_engineered = df_engineered.merge(location_counts, on=['Name of State', 'Name of District'], how='left')
df_engineered['Regional_Well_Density'] = df_engineered['Regional_Well_Density_Count'] # Use the merged column
df_engineered.drop(columns=['Regional_Well_Density_Count'], inplace=True) # Drop the temporary column

# Update target and features
y = df_engineered['Well_Suitability_Binary']
X = df_engineered.drop(columns=['Well_ID', 'Well_Suitability_Binary'])

print(f"Original features: {len(df.columns) - 2}")
print(f"Engineered features: {len(X.columns)}")
print("New features added:")
print("- Depth_to_Discharge_Ratio")
print("- Water_Quality_Category")
print("- Groundwater_Stress")
print("- Recharge_Availability_Ratio")
print("- Regional_Well_Density")

# Identify categorical and numerical features for preprocessing
categorical_features = [
    'Name of State',
    'Name of District',
    'Lithology_Top_Layer',
    'Drilling_Method_Used',
    'Water_Quality_Category'  # New engineered feature
]

numerical_features = [
    'Latitude',
    'Longitude',
    'Well_Depth_Actual_m',
    'Well_Discharge_LPM',
    'Water_Quality_TDS_mg_L',
    'Total Annual Ground Water Recharge',
    'Net Ground Water Availability for future use',
    'Stage of Ground Water Extraction (%)',
    'Depth_to_Discharge_Ratio',    # New engineered feature
    'Groundwater_Stress',          # New engineered feature
    'Recharge_Availability_Ratio', # New engineered feature
    'Regional_Well_Density'        # New engineered feature
]

# --- 3. Enhanced Pre-processing Pipeline ---

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' avoids multicollinearity
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\nAdvanced preprocessing pipeline created.")

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 5. Advanced Model Pipeline with Hyperparameter Tuning ---

# Define multiple models for ensemble (simplified for speed)
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

# Simplified hyperparameter grids for faster execution
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__max_features': ['sqrt'],
        'classifier__class_weight': ['balanced']
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.1, 0.15],
        'classifier__max_depth': [3, 5]
    },
    'LogisticRegression': {
        'classifier__C': [1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear'],
        'classifier__class_weight': ['balanced']
    }
}

best_models = {}
best_scores = {}

print("\nStarting hyperparameter tuning for multiple models...")

# Train and tune each model
for name, model in models.items():
    print(f"\nTuning {name}...")

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Grid search with reduced CV for speed
    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,   # Reduced from 5 to 3 for faster execution
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

    print(f"{name} - Best CV Score: {grid_search.best_score_:.4f}")
    print(f"{name} - Best Parameters: {grid_search.best_params_}")

print("\n" + "="*50)
print("MODEL COMPARISON:")
for name, score in best_scores.items():
    print(f"{name}: {score:.4f}")
print("="*50)

# --- 6. Create Ensemble Model ---

# Select best performing models for ensemble
print("\nCreating ensemble model...")

# Get the top 3 performing models
top_models = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:3]
ensemble_models = [(name, best_models[name]) for name, score in top_models]

print(f"Top models for ensemble: {[name for name, _ in top_models]}")

# Create voting classifier
ensemble_model = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',   # Use probability averages
    weights=[score for name, score in top_models] # Weight by CV score
)

print("\nTraining ensemble model...")
ensemble_model.fit(X_train, y_train)
print("Ensemble model training complete.")

# Also keep the best individual model
best_individual_name = max(best_scores, key=best_scores.get)
best_individual_model = best_models[best_individual_name]

print(f"\nBest individual model: {best_individual_name} (CV Score: {best_scores[best_individual_name]:.4f})")

# --- 7. Comprehensive Model Evaluation ---

print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Evaluate ensemble model
print("\n1. ENSEMBLE MODEL PERFORMANCE:")
y_pred_ensemble = ensemble_model.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")

print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Evaluate best individual model
print(f"\n2. BEST INDIVIDUAL MODEL ({best_individual_name}) PERFORMANCE:")
y_pred_individual = best_individual_model.predict(X_test)
accuracy_individual = accuracy_score(y_test, y_pred_individual)
print(f"Individual Model Accuracy: {accuracy_individual:.4f}")

print(f"\n{best_individual_name} Classification Report:")
print(classification_report(y_test, y_pred_individual))

# Choose the better performing model
if accuracy_ensemble >= accuracy_individual:
    final_model = ensemble_model
    y_pred_final = y_pred_ensemble
    accuracy_final = accuracy_ensemble
    model_name = "Ensemble"
    print(f"\n🏆 WINNER: Ensemble Model (Accuracy: {accuracy_final:.4f})")
else:
    final_model = best_individual_model
    y_pred_final = y_pred_individual
    accuracy_final = accuracy_individual
    model_name = best_individual_name
    print(f"\n🏆 WINNER: {best_individual_name} (Accuracy: {accuracy_final:.4f})")

# Feature importance (for tree-based models)
if hasattr(final_model, 'feature_importances_') or (hasattr(final_model, 'estimators_') and hasattr(final_model.estimators_[0], 'feature_importances_')):
    print(f"\n3. FEATURE IMPORTANCE ANALYSIS:")
    try:
        # Get feature names after preprocessing
        # Get one-hot encoded feature names first
        onehot_feature_names = final_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        # Combine numerical and one-hot encoded feature names
        all_feature_names = numerical_features + list(onehot_feature_names)

        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
        elif isinstance(final_model, VotingClassifier):
            individual_importances = []
            for name, estimator in final_model.estimators_:
                if hasattr(estimator.named_steps['classifier'], 'feature_importances_'):
                    individual_importances.append(estimator.named_steps['classifier'].feature_importances_)
            if individual_importances:
                importances = np.mean(individual_importances, axis=0)
            else:
                importances = None
        else: # For single models, if it's a pipeline, get from the classifier step
            if hasattr(final_model.named_steps['classifier'], 'feature_importances_'):
                importances = final_model.named_steps['classifier'].feature_importances_
            else:
                importances = None


        if importances is not None and len(importances) == len(all_feature_names):
            # Sort features by importance
            feature_importance = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)

            print("Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"{i+1:2d}. {feature[:40]:<40} {importance:.4f}")
        else:
             print("Could not retrieve feature importances or mismatch in feature count.")

    except Exception as e:
        print(f"Could not extract feature importance: {e}")

print(f"\n4. ACCURACY IMPROVEMENT:")
baseline_accuracy = 0.5   # Random baseline
improvement = ((accuracy_final - baseline_accuracy) / baseline_accuracy) * 100
print(f"Improvement over random baseline: {improvement:.1f}%")

# Enhanced Confusion Matrix for better insight
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Not Suitable (0)', 'Suitable (1)'],
            yticklabels=['Not Suitable (0)', 'Suitable (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - {model_name} Model\nAccuracy: {accuracy_final:.4f}')
plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved as 'confusion_matrix_optimized.png'")
plt.close()

# Performance summary
print("\n" + "="*60)
print("🎯 FINAL PERFORMANCE SUMMARY")
print("="*60)
print(f"✅ Model Type: {model_name}")
print(f"✅ Final Accuracy: {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
print(f"✅ Features Used: {len(X.columns)} (including {len(numerical_features)-8} engineered features)") # Adjusted count
print(f"✅ Training Samples: {X_train.shape[0]}")
print(f"✅ Test Samples: {X_test.shape[0]}")
print("="*60)

print(f"\n🚀 OPTIMIZED water well suitability prediction model ready!")
print(f"This {model_name.lower()} model can predict well suitability with {accuracy_final:.1%} accuracy.")


# --- 8. Save the Final Model and Auxiliary Data for Deployment ---
model_filename = 'well_suitability_model.pkl'
features_filename = 'model_features.pkl'
categorical_options_filename = 'categorical_options.pkl'
location_counts_filename = 'location_counts_lookup.pkl' # To handle Regional_Well_Density

try:
    with open(model_filename, 'wb') as file:
        pickle.dump(final_model, file)
    print(f"\nModel successfully saved as '{model_filename}'")

    with open(features_filename, 'wb') as file:
        pickle.dump(X.columns.tolist(), file) # Save the list of original features expected by the model
    print(f"Original feature list saved as '{features_filename}'")

    # Save unique categorical options for the Flask app dropdowns
    unique_categorical_values = {col: df[col].unique().tolist() for col in categorical_features if col in df.columns}
    # For 'Water_Quality_Category', ensure specific order of labels
    if 'Water_Quality_Category' in unique_categorical_values:
        unique_categorical_values['Water_Quality_Category'] = ['Excellent', 'Good', 'Fair', 'Poor'] # Enforce order
    with open(categorical_options_filename, 'wb') as file:
        pickle.dump(unique_categorical_values, file)
    print(f"Categorical options saved as '{categorical_options_filename}'")

    # Save the location_counts DataFrame
    with open(location_counts_filename, 'wb') as file:
        pickle.dump(location_counts, file)
    print(f"Location counts lookup saved as '{location_counts_filename}'")

except Exception as e:
    print(f"Error saving model or auxiliary data: {e}")