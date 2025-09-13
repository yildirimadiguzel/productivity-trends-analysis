"""
Contributor Predictive Models

This script implements predictive models to analyze the relationship between 
contributor experience and impact metrics. It includes both regression and 
classification models to predict contributor impact and identify key factors 
that influence productivity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# For classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set visualization style
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_and_preprocess_data():
    """Load and preprocess the contributor impact dataset."""
    # Load the contributor impact dataset
    df = pd.read_csv('data/contributor_impact_dataset.csv')
    
    # Convert date columns to datetime
    df['first_contribution'] = pd.to_datetime(df['first_contribution'])
    df['last_contribution'] = pd.to_datetime(df['last_contribution'])
    
    # Handle infinite values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate values (median for numeric columns)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Create additional features that might be useful for modeling
    df['contribution_duration'] = (df['last_contribution'] - df['first_contribution']).dt.days
    df['avg_changes_per_commit'] = df['total_changes'] / df['total_commits']
    df['active_ratio'] = df['active_years'] / df['years_since_first_commit']
    
    # Replace any new infinite values
    for col in ['avg_changes_per_commit', 'active_ratio']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_features_and_target(df, target_col='impact_score', classification=False):
    """Prepare features and target variables for modeling."""
    # Select features for modeling
    feature_cols = [
        'total_commits', 'total_files_changed', 'total_additions', 'total_deletions',
        'years_since_first_commit', 'active_years', 'total_changes',
        'avg_additions_per_commit', 'avg_deletions_per_commit', 'avg_files_per_commit',
        'recency_score', 'consistency_score', 'contribution_duration', 'avg_changes_per_commit',
        'active_ratio'
    ]
    
    # Remove any columns that don't exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # For classification, create categorical target
    if classification:
        # Create impact categories based on quartiles
        q_25, q_50, q_75 = df[target_col].quantile([0.25, 0.5, 0.75])
        
        def categorize_impact(value):
            if value <= q_25:
                return 'Low'
            elif value <= q_50:
                return 'Medium-Low'
            elif value <= q_75:
                return 'Medium-High'
            else:
                return 'High'
        
        df['impact_category'] = df[target_col].apply(categorize_impact)
        target = df['impact_category']
        
        # Encode the target variable
        le = LabelEncoder()
        target_encoded = le.fit_transform(target)
        
        # Store the mapping for interpretation
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Target mapping: {target_mapping}")
        
        X = df[feature_cols]
        y = target_encoded
    else:
        # For regression, use the numeric target
        X = df[feature_cols]
        y = df[target_col]
    
    return X, y

def train_regression_models(X, y):
    """Train and evaluate regression models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a preprocessing pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae
        }
        
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            feature_importances[name] = pd.Series(
                np.abs(model.coef_),
                index=X.columns
            ).sort_values(ascending=False)
    
    return results, feature_importances, X_train, X_test, y_train, y_test

def train_classification_models(X, y):
    """Train and evaluate classification models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a preprocessing pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix
        }
        
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            # For multi-class, take the mean of absolute coefficients across classes
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                feature_importances[name] = pd.Series(
                    np.mean(np.abs(model.coef_), axis=0),
                    index=X.columns
                ).sort_values(ascending=False)
            else:
                feature_importances[name] = pd.Series(
                    np.abs(model.coef_[0]),
                    index=X.columns
                ).sort_values(ascending=False)
    
    return results, feature_importances, X_train, X_test, y_train, y_test

def plot_regression_results(results, feature_importances, X_test, y_test, best_model_name, best_model_pipeline):
    """Plot regression model results and feature importances."""
    # Plot model performance comparison
    metrics = ['RMSE', 'R2', 'MAE']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results.keys()]
        axes[i].bar(results.keys(), values)
        axes[i].set_title(f'Model Comparison - {metric}')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/regression_model_comparison.png')
    
    # Plot feature importances for the best model
    if best_model_name in feature_importances:
        plt.figure(figsize=(10, 8))
        feature_importances[best_model_name].head(10).plot(kind='barh')
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('plots/regression_feature_importance.png')
    
    # Plot actual vs predicted values
    y_pred = best_model_pipeline.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {best_model_name}')
    plt.tight_layout()
    plt.savefig('plots/regression_actual_vs_predicted.png')

def plot_classification_results(results, feature_importances, X_test, y_test, best_model_name, best_model_pipeline, le=None):
    """Plot classification model results and feature importances."""
    # Plot model accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = [results[model]['Accuracy'] for model in results.keys()]
    plt.bar(results.keys(), accuracies)
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('plots/classification_model_comparison.png')
    
    # Plot feature importances for the best model
    if best_model_name in feature_importances:
        plt.figure(figsize=(10, 8))
        feature_importances[best_model_name].head(10).plot(kind='barh')
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('plots/classification_feature_importance.png')
    
    # Plot confusion matrix
    conf_matrix = results[best_model_name]['Confusion Matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/classification_confusion_matrix.png')

def run_regression_analysis():
    """Run the complete regression analysis pipeline."""
    print("Running regression analysis...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Define target variables for regression
    regression_targets = ['impact_score', 'weighted_impact_score', 'impact_per_year']
    
    for target in regression_targets:
        print(f"\nAnalyzing target: {target}")
        
        # Prepare features and target
        X, y = prepare_features_and_target(df, target_col=target)
        
        # Train regression models
        results, feature_importances, X_train, X_test, y_train, y_test = train_regression_models(X, y)
        
        # Print results
        print("\nRegression Model Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # Find the best model based on R2 score
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        print(f"\nBest model for {target}: {best_model_name} (R2: {results[best_model_name]['R2']:.4f})")
        
        # Print feature importances for the best model
        if best_model_name in feature_importances:
            print("\nTop 5 features:")
            print(feature_importances[best_model_name].head(5))
        
        # Create a pipeline for the best model
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
        
        if best_model_name == 'Linear Regression':
            model = LinearRegression()
        elif best_model_name == 'Ridge Regression':
            model = Ridge(alpha=1.0)
        elif best_model_name == 'Lasso Regression':
            model = Lasso(alpha=0.1)
        elif best_model_name == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # Gradient Boosting
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        best_model_pipeline.fit(X_train, y_train)
        
        # Plot results
        plot_regression_results(results, feature_importances, X_test, y_test, best_model_name, best_model_pipeline)

def run_classification_analysis():
    """Run the complete classification analysis pipeline."""
    print("\nRunning classification analysis...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Define target variables for classification
    classification_targets = ['impact_score', 'weighted_impact_score']
    
    for target in classification_targets:
        print(f"\nAnalyzing target: {target} (Classification)")
        
        # Prepare features and target
        X, y = prepare_features_and_target(df, target_col=target, classification=True)
        
        # Train classification models
        results, feature_importances, X_train, X_test, y_train, y_test = train_classification_models(X, y)
        
        # Print results
        print("\nClassification Model Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            
            # Print class-wise metrics
            for class_name, class_metrics in metrics['Classification Report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"  Class {class_name}:")
                    print(f"    Precision: {class_metrics['precision']:.4f}")
                    print(f"    Recall: {class_metrics['recall']:.4f}")
                    print(f"    F1-score: {class_metrics['f1-score']:.4f}")
        
        # Find the best model based on accuracy
        best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
        print(f"\nBest model for {target}: {best_model_name} (Accuracy: {results[best_model_name]['Accuracy']:.4f})")
        
        # Print feature importances for the best model
        if best_model_name in feature_importances:
            print("\nTop 5 features:")
            print(feature_importances[best_model_name].head(5))
        
        # Create a pipeline for the best model
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
        
        if best_model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif best_model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        best_model_pipeline.fit(X_train, y_train)
        
        # Plot results
        plot_classification_results(results, feature_importances, X_test, y_test, best_model_name, best_model_pipeline)

def cluster_contributors():
    """Perform clustering analysis on contributors."""
    print("\nPerforming clustering analysis...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Select features for clustering
    cluster_features = [
        'total_commits', 'total_changes', 'years_since_first_commit', 
        'active_years', 'impact_score', 'consistency_score', 'recency_score'
    ]
    
    # Prepare data for clustering
    X_cluster = df[cluster_features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('plots/clustering_elbow_curve.png')
    
    # Choose optimal k (this is a heuristic, you might want to adjust)
    # Looking for the "elbow" in the curve
    optimal_k = 4  # This is a common choice, but you should examine the elbow curve
    
    # Perform KMeans clustering with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze the clusters
    cluster_analysis = df.groupby('cluster').agg({
        'total_commits': 'mean',
        'total_changes': 'mean',
        'years_since_first_commit': 'mean',
        'active_years': 'mean',
        'impact_score': 'mean',
        'consistency_score': 'mean',
        'recency_score': 'mean',
        'author_name': 'count'
    }).rename(columns={'author_name': 'count'})
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize the clusters using PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Contributor Clusters (PCA)')
    plt.savefig('plots/contributor_clusters.png')
    
    # Return the clustered dataframe for further analysis
    return df

if __name__ == "__main__":
    # Run regression analysis
    run_regression_analysis()
    
    # Run classification analysis
    run_classification_analysis()
    
    # Perform clustering analysis
    clustered_df = cluster_contributors()
    
    print("\nAnalysis complete. Results saved to plots/ directory.")
