import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # For garbage collection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Check available columns in the dataset
def check_columns(file_path):
    # Read just a few rows to get column names
    df_sample = pd.read_csv(file_path, nrows=5)
    print("Available columns in the dataset:")
    for col in df_sample.columns:
        print(f"- {col}")
    return list(df_sample.columns)

# Load the dataset - with expanded column selection
def load_dataset(file_path, usecols=None):
    # If no are columns specified, read all the columns
    if usecols is None:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, usecols=usecols)
    
    print(f"Initial dataset shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")
    return df

# Clean and preprocess data - enhanced with feature engineering
def clean_data(df):
    # Clean price column
    df["price"] = df["price"].replace(r'[$,]', '', regex=True).astype(float)
    
    # Drop unnecessary columns
    drop_cols = ["calendar_last_scraped", "description", "bathrooms_text", "latitude", "longitude"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Feature engineering based on available columns
    # Price per person
    if 'accommodates' in df.columns:
        df['price_per_person'] = df['price'] / df['accommodates'].clip(lower=1)
    
    # Price per bedroom (if available)
    if 'bedrooms' in df.columns:
        df['price_per_bedroom'] = df['price'] / df['bedrooms'].fillna(1).clip(lower=1)
    
    # Price per bed (if available)
    if 'beds' in df.columns:
        df['price_per_bed'] = df['price'] / df['beds'].fillna(1).clip(lower=1)
    
    # Host experience (if available)
    if 'host_since' in df.columns:
        try:
            df['host_since'] = pd.to_datetime(df['host_since'])
            today = pd.to_datetime('2024-10-31')  # Use a fixed date near the script writing time
            df['host_experience_days'] = (today - df['host_since']).dt.days
            df.drop(columns=['host_since'], inplace=True)
        except:
            print("Could not process host_since column")
    
    # Response rate to numeric (if available)
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100
    
    # Host is superhost to numeric
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    
    # Extract features from amenities if available
    if 'amenities' in df.columns:
        try:
            # Key amenities that might affect price
            key_amenities = ['wifi', 'kitchen', 'washer', 'dryer', 'air conditioning', 
                             'heating', 'tv', 'pool', 'gym', 'breakfast']
            
            # Clean the amenities string and convert to lowercase
            df['amenities'] = df['amenities'].fillna('[]')
            df['amenities'] = df['amenities'].str.lower().str.replace('[{}"\']', '', regex=True)
            
            # Create binary features for each key amenity
            for amenity in key_amenities:
                df[f'has_{amenity.replace(" ", "_")}'] = df['amenities'].str.contains(amenity).astype(int)
            
            # Count total amenities
            df['amenities_count'] = df['amenities'].str.count(',') + 1
            df.loc[df['amenities'] == '[]', 'amenities_count'] = 0
            
            # Drop the original amenities column to save memory
            df.drop(columns=['amenities'], inplace=True)
        except:
            print("Could not process amenities column")
    
    # Neighborhood grouping if many unique values (to reduce dimensionality)
    if 'neighbourhood' in df.columns and df['neighbourhood'].nunique() > 20:
        # Group neighborhoods by average price
        neigh_prices = df.groupby('neighbourhood')['price'].median().reset_index()
        neigh_prices['price_quantile'] = pd.qcut(neigh_prices['price'], 5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        neigh_map = dict(zip(neigh_prices['neighbourhood'], neigh_prices['price_quantile']))
        df['neighbourhood_price_level'] = df['neighbourhood'].map(neigh_map)
    
    # Interaction terms for better model performance
    if all(col in df.columns for col in ['accommodates', 'review_scores_rating']):
        df['accommodates_x_rating'] = df['accommodates'] * df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    
    # Minimum nights categorisation if it's available
    if 'minimum_nights' in df.columns:
        df['stay_category'] = pd.cut(
            df['minimum_nights'], 
            bins=[0, 1, 3, 7, 30, float('inf')], 
            labels=['one_night', 'short_stay', 'week_stay', 'month_stay', 'long_term']
        )
    
    return df

# Remove outliers based on IQR method
def remove_outliers(df, columns=['price']):
    original_shape = df.shape
    
    for column in columns:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Only apply if it's not removing too much data
            filtered_df = df[(df[column] >= lower) & (df[column] <= upper)]
            if len(filtered_df) > 0.5 * len(df):  # Ensure we're not removing too much data
                df = filtered_df
                print(f"Removed outliers from {column}: {original_shape[0] - len(df)} rows")
    
    print(f"Shape after outlier removal: {df.shape} (was {original_shape})")
    return df

# This will compare the imputation techniques
def compare_imputation_methods(df, target_col='price', missing_cols=['bathrooms', 'beds', 'review_scores_rating']):
    """Compare different imputation techniques and their impact on model performance."""
    # Create copies of data with artificially introduced missingness to test
    test_df = df.copy()
    
    test_df = test_df.sample(2000, random_state=42)
    print(f"Using {len(test_df)} samples for imputation comparison")

    # Create log_price if it doesn't exist yet
    if 'log_price' not in test_df.columns:
        test_df['log_price'] = np.log1p(test_df[target_col])
    
    # Introduce some artificial missingness (e.g., 10% of values)
    for col in missing_cols:
        if col in test_df.columns:
            mask = np.random.rand(len(test_df)) < 0.1
            test_df.loc[mask, col] = np.nan
    
    # Split data
    X = test_df.drop(columns=[target_col, 'log_price'])
    y = test_df['log_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define imputation strategies to compare
    imputation_methods = {
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'KNN': KNNImputer(n_neighbors=3)
    }
    
    results = {}
    
    # Test each imputation method
    for name, imputer in imputation_methods.items():
        print(f"Testing {name} imputation...")
        # Create pipeline with imputer
        numeric_transformer = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessor with categorical handling as well
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
        
        # Create and evaluate model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression()) 
        ])
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results[name] = {'R²': r2, 'RMSE': rmse}
    
    # Create comparison visualisation
    results_df = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=results_df.index, y=results_df['R²'])
    plt.title('Imputation Method vs R² Score')
    plt.ylabel('R² Score (higher is better)')
    plt.ylim(0.5, max(results_df['R²']) * 1.1)  # Adjust as needed
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=results_df.index, y=results_df['RMSE'])
    plt.title('Imputation Method vs RMSE')
    plt.ylabel('RMSE (lower is better)')
    
    plt.tight_layout()
    plt.savefig('imputation_comparison.png')
    plt.close()
    
    return results_df

def plot_price_before_after(df_before, df_after, output_path="./price_before_after_outliers.png"):
    
    # This creates a side-by-side boxplot of price before and after outlier removal.
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df_before["price"])
    plt.title("Price Before Outlier Removal")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_after["price"])
    plt.title("Price After Outlier Removal")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualise_data_distributions(df, numeric_cols, categorical_cols=None, output_dir="./"):
    
   # This creates visualisations of data distributions for numeric and categorical variables.
    
    print("Creating data distribution visualisations...")
    
    # 1. Price distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["price"])
    plt.title("Price Distribution", fontsize=14)
    plt.xlabel("Price (£)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}price_distribution.png")
    plt.close()
    
    # 2. Histograms of key numeric features
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(numeric_cols, 1):
        if col in df.columns:
            plt.subplot(2, 3, i)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}feature_distributions.png")
    plt.close()
    
    # 3. Pairplot of numeric features with price
    print("Creating pairplot (this may take a moment)...")
    pairplot_cols = ["log_price"] + [col for col in numeric_cols if col != "price"]
    pairplot_cols = [col for col in pairplot_cols if col in df.columns]
    
    # Check if we have enough columns for a meaningful pairplot
    if len(pairplot_cols) >= 2:
        # Use a sample if the dataset is large
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        
        sns.pairplot(df_sample[pairplot_cols], diag_kind="kde")
        plt.suptitle("Relationships Between Features", y=1.02, fontsize=16)
        plt.savefig(f"{output_dir}pairplot_features.png")
        plt.close()
    
    # 4. Boxplots of price by categorical features
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns and df[col].nunique() < 10:  # Only for categorical with reasonable number of categories
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=df, x=col, y="log_price")
                plt.title(f"Log Price Distribution by {col}", fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_dir}log_price_by_{col}.png")
                plt.close()
    
    # 5. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Keep only the columns with valid names for the correlation matrix
    valid_cols = [col for col in numeric_df.columns if isinstance(col, str)]
    numeric_df = numeric_df[valid_cols]
    
    plt.figure(figsize=(12, 8))

    corr_matrix = numeric_df.corr(numeric_only=True)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}correlation_heatmap.png", dpi=300)
    plt.close()

    
    print("Data distribution visualisations created.")

def visualise_residuals(y_true, y_pred, model_name, output_dir="./"):
    
    # Create residual plots for model evaluation.
    
    residuals = y_true - y_pred
    
    # 1. Residual histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f"Residual Distribution - {model_name}", fontsize=14)
    plt.xlabel("Residual (y_true - y_pred)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}residual_dist_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # 2. Residual vs. Predicted scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f"Residual vs. Predicted - {model_name}", fontsize=14)
    plt.xlabel("Predicted Value", fontsize=12)
    plt.ylabel("Residual", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}residual_vs_pred_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # 3. Q-Q plot for normality check
    plt.figure(figsize=(10, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}qq_plot_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

def visualise_feature_importance(model, feature_names, top_n=20, output_dir="./"):
    
    # Create feature importance plots for tree-based models.
    
    if hasattr(model, 'feature_importances_'):
        # Create DataFrame of feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Limit to top_n features
        top_n = min(top_n, len(feature_names))
        top_indices = indices[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices])
        plt.title(f"Top {top_n} Feature Importances", fontsize=16)
        plt.xlabel("Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}feature_importances.png")
        plt.close()
        
        # Return the importances for further analysis
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    else:
        print("Model does not have feature_importances_ attribute")
        return None

def perform_clustering_with_visualisation(X, k_range=range(2, 7), output_dir="./"):
    
    # Perform K-means clustering with visualisation of elbow method and silhouette scores.
    # Returns the optimal k and cluster labels.
    
    from sklearn.cluster import KMeans
    
    # Elbow method and silhouette analysis
    print("Finding optimal number of clusters...")
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score if k >= 2
        if k >= 2:
            silhouette_avg = silhouette_score(X, labels)
            silhouette_scores.append(silhouette_avg)
            print(f"k={k}, silhouette score: {silhouette_avg:.4f}")
        else:
            silhouette_scores.append(0)
    
    # Create elbow method and silhouette score plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.title("Elbow Method for Optimal k", fontsize=14)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.xticks(k_range)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range[1:], silhouette_scores[1:], 'o-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.title("Silhouette Score for Optimal k", fontsize=14)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(k_range[1:])
    plt.tight_layout()
    plt.savefig(f"{output_dir}kmeans_optimisation.png")
    plt.close()
    
    # Determine the optimal k (highest silhouette score)
    optimal_k = k_range[1:][np.argmax(silhouette_scores[1:])]
    print(f"Optimal number of clusters (based on silhouette score): {optimal_k}")
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return optimal_k, labels, kmeans.cluster_centers_

def visualise_clusters_pca(X, labels, centers=None, output_dir="./"):
    
    # Create PCA visualisation of clusters.
    
    # PCA for visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=30)
    
    # Add centers if provided
    if centers is not None:
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, marker='X')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"Cluster visualisation (PCA, k={len(np.unique(labels))})", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}cluster_pca_visualisation.png")
    plt.close()

def create_cluster_radar_chart(df, labels, feature_names, output_dir="./"):
    """
    Create radar chart visualisation of cluster profiles.
    """
    # Get numeric data
    numeric_data = df.select_dtypes(include=[np.number])
    numeric_cols = [col for col in numeric_data.columns if col not in ['Cluster', 'log_price', 'price']]
    
    # Limit to a reasonable number of features for readability
    max_features = min(8, len(numeric_cols))
    
    # Calculate variance of each feature across clusters
    feature_variance = []
    for col in numeric_cols:
        cluster_means = df.groupby('Cluster')[col].mean()
        variance = np.var(cluster_means)
        feature_variance.append((col, variance))
    
    # Select features with highest variance across clusters
    top_features = sorted(feature_variance, key=lambda x: x[1], reverse=True)[:max_features]
    radar_features = [feat[0] for feat in top_features]
    
    # Log top discriminating features
    print(f"Top {max_features} discriminating features for clusters:")
    for feature, variance in top_features:
        print(f"- {feature}: variance = {variance:.4f}")
    
    # Calculate cluster means for each feature
    cluster_means = df.groupby('Cluster')[radar_features].mean()
    
    # Scale features for radar chart
    scaler = MinMaxScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means), 
        index=cluster_means.index, 
        columns=cluster_means.columns
    )
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Plot each cluster
    for cluster_idx in sorted(df['Cluster'].unique()):
        values = cluster_means_scaled.loc[cluster_idx].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_idx}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features)
    plt.title('Cluster Profiles - Top Discriminating Features', fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}cluster_radar_chart.png")
    plt.close()

def compare_models_performance(results_df, metrics=['R²', 'RMSE', 'MAE', 'MSE'], output_dir="./"):
    """
    Create visualisations comparing model performance.
    """
    print("Creating model comparison visualisations...")
    
    # Calculate number of subplots needed
    n_metrics = len([m for m in metrics if m in results_df.columns])
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    plot_idx = 1
    for metric in metrics:
        if metric in results_df.columns:
            plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            
            # Sort by metric value (ascending or descending based on the metric)
            ascending = False if metric in ['R²', 'R²(orig)'] else True
            sorted_df = results_df.sort_values(by=metric, ascending=ascending)
            
            # Create bar chart
            bars = plt.barh(sorted_df['Model'], sorted_df[metric])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + (0.01 * width), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', 
                        ha='left', 
                        va='center')
            
            # Set appropriate title based on the metric
            direction = "Higher is better" if metric in ['R²', 'R²(orig)'] else "Lower is better"
            plt.title(f"{metric} Comparison ({direction})", fontsize=14)
            plt.xlabel(metric, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}model_comparison.png")
    plt.close()
    
    print("Model comparison visualisations created.")

def compare_global_vs_local_models(results_df, output_dir="./"):
    """
    Create visualisation comparing global vs. cluster-specific models.
    """
    print("Comparing global vs. local models...")
    
    # Filter for global and cluster models
    global_models = results_df[~results_df['Model'].str.contains('Cluster', na=False)]
    cluster_models = results_df[results_df['Model'].str.contains('Cluster', na=False)]
    
    # Combine and sort by R²
    all_models = pd.concat([global_models, cluster_models]).sort_values(by='R²', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(all_models['Model'], all_models['R²'])
    
    # Color bars based on model type
    colors = ['#1f77b4' if 'Cluster' in model else '#ff7f0e' for model in all_models['Model']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.01, 
                f'{height:.4f}', 
                ha='center', 
                va='bottom')
    
    plt.title('R² Comparison: Global vs. Cluster-Specific Models', fontsize=16)
    plt.ylabel('R² Score (higher is better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f0e', label='Global Models'),
        Patch(facecolor='#1f77b4', label='Cluster-Specific Models')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}global_vs_local_models.png")
    plt.close()
    
    print("Global vs. local model comparison created.")

# Prepare features and target with enhanced feature selection
def prepare_features(df, target='price'):
    # Create log price target
    df['log_price'] = np.log1p(df[target])
    
    # Get target variable
    y = df['log_price']
    
    # Drop price columns from features
    price_cols = [target, 'log_price']
    if 'price_per_person' in df.columns:
        price_cols.append('price_per_person')
    if 'price_per_bedroom' in df.columns:
        price_cols.append('price_per_bedroom')
    if 'price_per_bed' in df.columns:
        price_cols.append('price_per_bed')
    
    # Create feature set, excluding ID columns
    exclude_cols = price_cols + ['id', 'name', 'host_id', 'host_name', 'first_review', 'last_review']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    # Convert categorical columns with few levels to dummy variables to reduce memory
    for col in X.select_dtypes(include=['object']).columns:
        if X[col].nunique() < 10:  # Only for columns with few unique values
            # Get dummies and drop the first to avoid multicollinearity
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    return X, y

# Build enhanced preprocessing pipeline
def build_preprocessor(X):
    # Split features by type
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Only create categorical transformer if there are categorical features
    transformers = [('num', numeric_transformer, numeric_features)]
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features

# Train and evaluate a model - with enhanced metrics and progress tracking
def evaluate_model(name, model, X_train, X_test, y_train, y_test, verbose=True):
    if verbose:
        print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"Making predictions with {name}...")
    
    # Make predictions
    preds = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    # Calculate R2 on original scale
    y_test_orig = np.expm1(y_test)
    preds_orig = np.expm1(preds)
    r2_orig = r2_score(y_test_orig, preds_orig)
    
    if verbose:
        print(f"{name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, R²(orig): {r2_orig:.4f}")
    
    # Return model info and predictions
    return {"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2, "R²(orig)": r2_orig}, preds

# Memory-efficient model comparison visualisation
def plot_model_comparisons(results_df):
    plt.figure(figsize=(12, 10))
    
    # Plot R² - higher is better
    plt.subplot(2, 2, 1)
    bars = plt.bar(results_df['Model'], results_df['R²'])
    plt.title('R² Comparison (higher is better)')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot RMSE - lower is better
    plt.subplot(2, 2, 2)
    bars = plt.bar(results_df['Model'], results_df['RMSE'])
    plt.title('RMSE Comparison (lower is better)')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot MAE - lower is better
    plt.subplot(2, 2, 3)
    bars = plt.bar(results_df['Model'], results_df['MAE'])
    plt.title('MAE Comparison (lower is better)')  
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot R² on original scale
    plt.subplot(2, 2, 4)
    bars = plt.bar(results_df['Model'], results_df['R²(orig)'])
    plt.title('R² on Original Scale')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png')
    print(f"Model comparison plot saved to 'model_metrics_comparison.png'")
    plt.close()

# Memory-efficient k-means clustering with optimal k selection
def perform_kmeans_clustering(X_scaled, k_range=range(2, 7)):
    print("Finding optimal number of clusters...")
    silhouette_scores = []
    
    # Try different values of k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"k={k}, silhouette score: {score:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k Values')
    plt.grid(True)
    plt.savefig('kmeans_silhouette_scores.png')
    plt.close()
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform clustering with optimal k
    print(f"Performing KMeans clustering with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    return kmeans_labels, optimal_k

# Memory-efficient PCA visualisation of clusters
def plot_clusters_pca(X_scaled, labels, optimal_k):
    print("Performing PCA for visualisation...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'KMeans Clustering Results (k={optimal_k})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('kmeans_clusters.png')
    plt.close()

# analyse cluster characteristics with more detailed profiling
def analyse_clusters(df, y, labels):
    # Create a dataframe with cluster labels
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels
    df_analysis['log_price'] = y
    
    # Calculate summary statistics by cluster
    cluster_stats = df_analysis.groupby('Cluster').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'log_price': ['mean', 'median', 'std']
    })
    
    # Flatten the multi-index for better display
    cluster_stats.columns = [f'{col[0]}_{col[1]}' for col in cluster_stats.columns]
    print("\nCluster Price Statistics:")
    print(cluster_stats)
    
    # Feature importance for each cluster
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['price', 'log_price', 'Cluster']]
    
    # Create a detailed profile of each cluster
    profile_data = []
    
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        
        # Calculate the Z-score of each feature in this cluster compared to the overall dataset
        cluster_profile = {}
        for col in numeric_cols:
            if col in df_analysis.columns:
                # Calculate z-score of mean
                overall_mean = df_analysis[col].mean()
                overall_std = df_analysis[col].std()
                
                if overall_std > 0:  # Avoid division by zero
                    cluster_mean = cluster_data[col].mean()
                    z_score = (cluster_mean - overall_mean) / overall_std
                    cluster_profile[col] = z_score
        
        # Add key categorical variables if available
        for cat_col in ['room_type', 'property_type', 'neighbourhood']:
            if cat_col in df_analysis.columns:
                # Get most common value in this cluster
                most_common = cluster_data[cat_col].value_counts().index[0]
                cluster_profile[f'{cat_col}_most_common'] = most_common
        
        cluster_profile['cluster'] = cluster
        cluster_profile['size'] = len(cluster_data)
        cluster_profile['price_mean'] = cluster_data['price'].mean()
        
        profile_data.append(cluster_profile)
    
    # Create cluster profile dataframe
    cluster_profiles = pd.DataFrame(profile_data)
    print("\nCluster Profiles (Z-scores of numeric features):")
    print(cluster_profiles)
    
    return cluster_profiles, df_analysis

# Enhanced radar chart that automatically handles features
def plot_cluster_radar(cluster_profiles, numeric_cols, n_features=8):
    # Select top features based on variance across clusters
    feature_variance = {}
    for col in numeric_cols:
        if col in cluster_profiles.columns and col not in ['cluster', 'size', 'price_mean']:
            feature_variance[col] = cluster_profiles[col].var()
    
    # Sort features by variance and take top n
    top_features = sorted(feature_variance.items(), key=lambda x: abs(x[1]), reverse=True)[:n_features]
    top_feature_names = [f[0] for f in top_features]
    
    print(f"Top {n_features} discriminating features for clusters:")
    for feature, variance in top_features:
        print(f"- {feature}: variance = {variance:.4f}")
    
    # Set up the radar chart
    n_clusters = len(cluster_profiles)
    angles = np.linspace(0, 2*np.pi, len(top_feature_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Plot each cluster
    for i, cluster_idx in enumerate(cluster_profiles['cluster']):
        values = cluster_profiles.loc[i, top_feature_names].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_idx}')
        ax.fill(angles, values, alpha=0.1)
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_feature_names)
    plt.title('Cluster Profiles - Top Discriminating Features')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('cluster_radar_chart.png')
    plt.close()

# Build and tune an optimised Random Forest model
def build_optimised_rf(X_train, y_train, X_test, y_test, preprocessor, verbose=True):
    print("Building optimised Random Forest model...")
    
    # Start with a simple RF to get a baseline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
    ])
    
    # Train and evaluate baseline model
    baseline_result, _ = evaluate_model("RF Baseline", rf_pipeline, X_train, X_test, y_train, y_test, verbose)
    
    # Define a small grid of important parameters to tune
    # Keep it small to be memory efficient
    param_grid = {
        'model__max_depth': [15, 20, 25],
        'model__min_samples_split': [2, 5]
    }
    
    print("Performing quick grid search to optimise Random Forest...")
    # Use a small subset for grid search if dataset is large
    if len(X_train) > 10000:
        print("Using subset of data for grid search to save memory...")
        X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=10000, random_state=42)
    else:
        X_sample, y_sample = X_train, y_train
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf_pipeline, 
        param_grid, 
        cv=3, 
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_sample, y_sample)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Create optimised model with best parameters
    best_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=150,  # Slightly more trees than baseline
            random_state=42,
            n_jobs=-1,
            **{k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        ))
    ])
    
    # Train and evaluate optimised model
    optimised_result, preds = evaluate_model("RF optimised", best_rf, X_train, X_test, y_train, y_test, verbose)
    
    return best_rf, baseline_result, optimised_result, preds

# Train cluster-specific models with optimisation
def train_cluster_models(X, y, labels, preprocessor):
    print("\nTraining cluster-specific models...")
    results = []
    
    # Train a global model first for comparison
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Build optimised global model
    _, baseline_result, optimised_result, _ = build_optimised_rf(
        X_train_global, y_train_global, X_test_global, y_test_global, preprocessor)
    
    results.append(baseline_result)
    results.append(optimised_result)
    
    # Free memory
    del X_train_global, X_test_global, y_train_global, y_test_global
    gc.collect()
    
    # Add cluster labels to X for filtering
    X_with_clusters = X.copy()
    X_with_clusters['Cluster'] = labels
    
    # Train a model for each cluster
    cluster_models = {}
    
    for cluster_id in np.unique(labels):
        # Get data for this cluster
        cluster_data = X_with_clusters[X_with_clusters['Cluster'] == cluster_id]
        X_cluster = cluster_data.drop(columns=['Cluster'])
        y_cluster = y.iloc[cluster_data.index]
        
        # Only proceed if enough samples
        if len(X_cluster) < 100:
            print(f"Cluster {cluster_id} has only {len(X_cluster)} samples - skipping")
            continue
            
        # Split data
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_cluster, y_cluster, test_size=0.2, random_state=42)
            
        # Build optimised cluster model (with less verbose output)
        model_c, _, cluster_result, _ = build_optimised_rf(
            X_train_c, y_train_c, X_test_c, y_test_c, preprocessor, verbose=False)
        
        # Store results
        cluster_result["Model"] = f"Cluster {cluster_id}"
        results.append(cluster_result)
        
        # Store model
        cluster_models[cluster_id] = model_c
        
        # Free memory
        del X_train_c, X_test_c, y_train_c, y_test_c
        gc.collect()
    
    # Create dataframe of results
    results_df = pd.DataFrame(results)
    
    print("\nCluster-specific vs. Global Model Results:")
    print(results_df[['Model', 'R²', 'RMSE']])
    
    # Plot comparison of R² scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['Model'], results_df['R²'])
    plt.title('R² Comparison: Global vs. Cluster-Specific Models')
    plt.ylabel('R² Score (higher is better)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cluster_models_comparison.png')
    plt.close()
    
    return results_df, cluster_models

# Main script execution
if __name__ == "__main__":
    # Path to dataset
    dataset_path = "London_Listings.csv"
    
    # First, check what columns are available
    print("\n=== Checking dataset columns ===")
    available_columns = check_columns(dataset_path)
    
    # Define columns to use based on what's available
    desired_columns = [
        'id', 'name', 'host_id', 'host_name', 'host_since', 'host_response_rate', 
        'host_is_superhost', 'neighbourhood', 'property_type', 'room_type', 'accommodates', 
        'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 
        'review_scores_rating'
    ]
    
    # Only include columns that actually exist in the dataset
    usecols = [col for col in desired_columns if col in available_columns]
    
    print(f"\nUsing these columns for analysis: {usecols}")
    
    # Load and clean data
    print("\n=== Loading and cleaning data ===")
    df = load_dataset(dataset_path, usecols=usecols)
    df = clean_data(df)

    df_before_outliers = df.copy()
    
    # Remove outliers from both price and review_scores_rating if available
    outlier_columns = ['price']
    if 'review_scores_rating' in df.columns:
        outlier_columns.append('review_scores_rating')
    df = remove_outliers(df, outlier_columns)

    plot_price_before_after(df_before_outliers, df)

    print("\n=== Comparing imputation methods ===")
    imputation_results = compare_imputation_methods(df)
    print("Imputation comparison results:")
    print(imputation_results)
    
    df['log_price'] = np.log1p(df['price'])  # This is needed for boxplots in visualisation

    visualise_data_distributions(
    df,
    numeric_cols=['price', 'accommodates', 'beds', 'review_scores_rating', 'bathrooms', 'bedrooms'],
    categorical_cols=['room_type', 'property_type', 'host_is_superhost']
)

    # Prepare features and target
    print("\n=== Preparing features ===")
    X, y = prepare_features(df)
    
    # Create explicit indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Build preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    
    # Split data
    print("\n=== Splitting data ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    print("\n=== Training models ===")
    results = []
    
    # Linear Regression
    lin_reg = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
    model_info, y_pred_lr = evaluate_model("Linear Regression", lin_reg, X_train, X_test, y_train, y_test)
    results.append(model_info)

    visualise_residuals(y_test, y_pred_lr, "Linear_Regression")

    # Free memory
    del lin_reg, y_pred_lr
    gc.collect()
    
    # Ridge Regression
    ridge = Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))])
    model_info, _ = evaluate_model("Ridge Regression", ridge, X_train, X_test, y_train, y_test)
    results.append(model_info)

    visualise_residuals(y_test, _, "Ridge_Regression")

    del ridge, _
    gc.collect()
    
    # Lasso Regression
    lasso = Pipeline([('preprocessor', preprocessor), ('model', Lasso(alpha=0.1))])
    model_info, _ = evaluate_model("Lasso Regression", lasso, X_train, X_test, y_train, y_test)
    results.append(model_info)

    visualise_residuals(y_test, _, "Lasso_Regression")

    del lasso, _
    gc.collect()
    
    # optimised Random Forest
    print("\n=== Building optimised Random Forest model ===")
    best_rf, baseline_result, optimised_result, y_pred_rf = build_optimised_rf(
        X_train, y_train, X_test, y_test, preprocessor)
    
    results.append(baseline_result)
    results.append(optimised_result)

    visualise_residuals(y_test, y_pred_rf, "Random_Forest")

    if hasattr(best_rf.named_steps['model'], 'feature_importances_'):
        # Get feature names
        cat_transformer = best_rf.named_steps['preprocessor'].transformers_[1][1]
        cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names)
    
        # visualise feature importances
        visualise_feature_importance(best_rf.named_steps['model'], all_feature_names)


    # Add Histogram Gradient Boosting Regressor
    print("\n=== Training Histogram Gradient Boosting Regressor ===")
    hist = Pipeline([
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=15,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    hist_result, hist_preds = evaluate_model("HistGradientBoosting", hist, X_train, X_test, y_train, y_test)
    results.append(hist_result)

    # Add visualisation of HistGradientBoosting residuals
    visualise_residuals(y_test, hist_preds, "HistGradientBoosting")

    del hist, hist_preds
    gc.collect()
    
    # visualise model comparison
    print("\n=== Comparing models ===")
    results_df = pd.DataFrame(results)
    plot_model_comparisons(results_df)

    # Add additional model comparison visualisation
    compare_models_performance(results_df)
    
    # Free memory
    del results, results_df
    gc.collect()
    
    # Prepare for clustering
    print("\n=== Preparing for clustering ===")
    # Use only numeric features for clustering
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Fill any NaN values
    for col in X_numeric.columns:
        if X_numeric[col].isna().any():
            X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
    
    # Scale numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Free memory
    del X_numeric
    gc.collect()
    
    # Perform clustering with visualisation
    print("\n=== Performing clustering analysis ===")
    optimal_k, kmeans_labels, centers = perform_clustering_with_visualisation(X_scaled)
        
    # Visualise clusters using PCA
    visualise_clusters_pca(X_scaled, kmeans_labels, centers)
        
    # Add cluster labels to original data
    df['Cluster'] = kmeans_labels
        
    # Create radar chart for cluster profiles
    create_cluster_radar_chart(df, kmeans_labels, X.select_dtypes(include=[np.number]).columns.tolist())
    
    # Analyse cluster characteristics
    print("\n=== Analysing cluster characteristics ===")
    cluster_profiles, df_with_clusters = analyse_clusters(df, y, kmeans_labels)
    
    # Select numeric columns for radar chart (excluding ID and price columns)
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if col not in ['id', 'price', 'log_price']]
    
    # Create radar chart for clusters
    plot_cluster_radar(cluster_profiles, numeric_columns, n_features=8)
    
    # Train cluster-specific models
    print("\n=== Training cluster-specific models ===")
    cluster_results, cluster_models = train_cluster_models(X, y, kmeans_labels, preprocessor)

    # Add global vs. local models comparison
    compare_global_vs_local_models(cluster_results)
    
    # Print summary of best models
    print("\n=== Performance Summary ===")

    # Find the best global model - include HistGradientBoosting in the search
    global_models = cluster_results[cluster_results['Model'].str.contains('RF|Ridge|Linear|Hist')]
    best_global_idx = global_models['R²'].idxmax()
    best_global = global_models.loc[best_global_idx]

    # Find the best cluster model
    cluster_models_df = cluster_results[cluster_results['Model'].str.contains('Cluster')]
    if len(cluster_models_df) > 0:
        best_cluster_idx = cluster_models_df['R²'].idxmax()
        best_cluster = cluster_models_df.loc[best_cluster_idx]
        
        improvement = (best_cluster['R²'] - best_global['R²']) / best_global['R²'] * 100
        
        print("\nPerformance comparison:")
        print(f"Best Global Model: {best_global['Model']} with R² = {best_global['R²']:.4f}")
        print(f"Best Cluster Model: {best_cluster['Model']} with R² = {best_cluster['R²']:.4f}")
        print(f"Improvement: {improvement:.2f}%")
    else:
        print("No valid cluster models found.")
    
    print("\n=== Analysis complete! ===")
    print("All visualisations have been saved to disk.")
    
    # Create a summary report
    summary_text = f"""
# Airbnb London Price Prediction - Summary Report

## Dataset Overview
- Initial shape: {df.shape[0]} rows, {df.shape[1]} columns
- Features used: {len(X.columns)} features

## Best Model Performance
- Best Global Model: {best_global['Model']} with R² = {best_global['R²']:.4f}
"""
    
    if len(cluster_models_df) > 0:
        summary_text += f"""- Best Cluster Model: {best_cluster['Model']} with R² = {best_cluster['R²']:.4f}
- Improvement from Clustering: {improvement:.2f}%
"""
    
    summary_text += f"""
## Clustering Results
- Optimal number of clusters: {optimal_k}
"""
    
    # Save summary report
    with open('analysis_summary.md', 'w') as f:
        f.write(summary_text)
    
    print("Summary report saved to 'analysis_summary.md'")