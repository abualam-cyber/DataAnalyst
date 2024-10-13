"""
DATA ANALYTICS AND VISUALIZATION APPLICATION

This application performs various types of data analysis and visualization on input data.
It supports multiple analysis types, visualization methods, and output formats.

Author: [Your Name]
Date: October 12, 2024
Version: 2.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from lifelines import KaplanMeierFitter
from sklearn.ensemble import IsolationForest
import logging
import os
import sys
from typing import Dict, Any, List
import configparser

# Load configuration
def load_config(config_file='config.ini'):
    """
    Load configuration from the specified file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Loaded configuration object.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = load_config()

# Set up logging
log_file = config.get('DEFAULT', 'log_file', fallback='data_analysis_app.log')
log_level = config.get('DEFAULT', 'log_level', fallback='INFO')
logging.basicConfig(filename=log_file, level=getattr(logging, log_level),
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_data(file_name: str) -> pd.DataFrame:
    """
    Load data from various file formats.

    Args:
        file_name (str): Name of the file to load.

    Returns:
        pandas.DataFrame: Loaded data.

    Raises:
        ValueError: If the file format is unsupported or file doesn't exist.
    """
    try:
        if not os.path.exists(file_name):
            raise ValueError(f"File does not exist: {file_name}")

        file_extension = file_name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_name)
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_name)
        elif file_extension == 'txt':
            df = pd.read_csv(file_name, sep='\t')
        elif file_extension == 'json':
            df = pd.read_json(file_name)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        logging.info(f"Data loaded successfully from {file_name}")
        return df
    except Exception as ex:
        logging.error(f"Error loading data: {str(ex)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data.

    Args:
        df (pandas.DataFrame): Input dataframe.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    try:
        # Handle missing values
        df = df.dropna()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle outliers (remove rows with values > 3 standard deviations)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df[(np.abs(df[numeric_columns] - df[numeric_columns].mean()) <= (3 * df[numeric_columns].std())).all(axis=1)]

        # Perform data type conversions
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        # Normalize numeric data
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        logging.info("Data cleaned successfully")
        return df
    except Exception as ex:
        logging.error(f"Error cleaning data: {str(ex)}")
        raise

def perform_analysis(df: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
    """
    Perform the specified analysis on the data.

    Args:
        df (pandas.DataFrame): Input dataframe.
        analysis_type (str): Type of analysis to perform.

    Returns:
        dict: Results of the analysis and additional data for visualization.

    Raises:
        ValueError: If the analysis type is unsupported.
    """
    try:
        if analysis_type == "regression":
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            return {
                "type": "regression",
                "predictions": predictions,
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "X": X,
                "y": y
            }

        elif analysis_type == "clustering":
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(df)
            return {
                "type": "clustering",
                "clusters": clusters,
                "centroids": kmeans.cluster_centers_
            }

        elif analysis_type == "dimensionality_reduction":
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df)
            return {
                "type": "dimensionality_reduction",
                "reduced_data": reduced_data,
                "explained_variance_ratio": pca.explained_variance_ratio_
            }

        elif analysis_type == "time_series":
            model = ARIMA(df.iloc[:, 0], order=(1, 1, 1))
            results = model.fit()
            forecast = results.forecast(steps=5)
            return {
                "type": "time_series",
                "forecast": forecast,
                "aic": results.aic,
                "original_data": df.iloc[:, 0]
            }

        elif analysis_type == "nlp":
            text = ' '.join(df.iloc[:, 0].astype(str))
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            return {
                "type": "nlp",
                "tokens": filtered_tokens,
                "word_freq": nltk.FreqDist(filtered_tokens)
            }

        elif analysis_type == "machine_learning":
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            model = RandomForestClassifier()
            model.fit(X, y)
            predictions = model.predict(X)
            train_sizes, train_scores, test_scores = learning_curve(model, X, y)
            return {
                "type": "machine_learning",
                "predictions": predictions,
                "feature_importance": model.feature_importances_,
                "train_sizes": train_sizes,
                "train_scores": train_scores,
                "test_scores": test_scores,
                "confusion_matrix": confusion_matrix(y, predictions),
                "X": X,
                "y": y
            }

        elif analysis_type == "deep_learning":
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            predictions = model.predict(X)
            return {
                "type": "deep_learning",
                "predictions": predictions,
                "history": history.history,
                "X": X,
                "y": y
            }

        elif analysis_type == "bayesian":
            data = df.iloc[:, 0]
            prior = stats.norm(loc=0, scale=1)
            likelihood = stats.norm(loc=data.mean(), scale=data.std())
            x = np.linspace(-10, 10, 1000)
            posterior = stats.norm(loc=0, scale=1)  # Placeholder for actual posterior calculation
            return {
                "type": "bayesian",
                "prior": prior.pdf(x),
                "likelihood": likelihood.pdf(x),
                "posterior": posterior.pdf(x),
                "x": x
            }

        elif analysis_type == "survival":
            T = df.iloc[:, 0]
            E = df.iloc[:, 1]
            kmf = KaplanMeierFitter()
            kmf.fit(T, E)
            return {
                "type": "survival",
                "survival_function": kmf.survival_function_,
                "median": kmf.median_survival_time_,
                "T": T,
                "E": E
            }

        elif analysis_type == "anomaly_detection":
            clf = IsolationForest(contamination=0.1)
            anomalies = clf.fit_predict(df)
            return {
                "type": "anomaly_detection",
                "anomalies": anomalies,
                "decision_function": clf.decision_function(df)
            }

        else:
            raise ValueError(f"Unsupported Analysis Type: {analysis_type}")
    except Exception as ex:
        logging.error(f"Error performing analysis: {str(ex)}")
        raise

def visualize_data(df: pd.DataFrame, results: Dict[str, Any], analysis_type: str, visualization_type: str, output_format: str) -> None:
    """
    Visualize the analyzed data

    Args:
        df (pandas.DataFrame): Original dataframe.
        results (dict): Results from the analysis.
        analysis_type (str): Type of analysis performed.
        visualization_type (str): Type of visualization to create.
        output_format (str): Output format for the visualization.
    """
    try:
        dpi = config.getint('VISUALIZATION', 'dpi', fallback=300)
        plt.figure(figsize=(12, 8), dpi=dpi)

        if analysis_type == "regression":
            if visualization_type == "scatter":
                plt.scatter(results['X'].iloc[:, 0], results['y'], color='blue', label='Actual')
                plt.plot(results['X'].iloc[:, 0], results['predictions'], color='red', label='Predicted')
                plt.title("Regression Analysis")
                plt.xlabel("Independent Variable")
                plt.ylabel("Dependent Variable")
                plt.legend()
            elif visualization_type == "residual":
                residuals = results['y'] - results['predictions']
                plt.scatter(results['predictions'], residuals)
                plt.title("Residual Plot")
                plt.xlabel("Predicted Values")
                plt.ylabel("Residuals")
                plt.axhline(y=0, color='r', linestyle='--')

        elif analysis_type == "clustering":
            if visualization_type == "scatter":
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=results['clusters'], cmap='viridis')
                plt.scatter(results['centroids'][:, 0], results['centroids'][:, 1], c='red', marker='x', s=200, linewidths=3)
                plt.title("Clustering Analysis")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
            elif visualization_type == "dendrogram":
                linkage_matrix = linkage(df, 'ward')
                dendrogram(linkage_matrix)
                plt.title("Hierarchical Clustering Dendrogram")
                plt.xlabel("Sample Index")
                plt.ylabel("Distance")

        elif analysis_type == "dimensionality_reduction":
            plt.scatter(results['reduced_data'][:, 0], results['reduced_data'][:, 1])
            plt.title("PCA Results")
            plt.xlabel("First Principal Component")
            plt.ylabel("Second Principal Component")

        elif analysis_type == "time_series":
            plt.plot(df.index, results['original_data'], label='Original')
            plt.plot(pd.date_range(start=df.index[-1], periods=6)[1:], results['forecast'], label='Forecast')
            plt.title("Time Series Forecast")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()

        elif analysis_type == "nlp":
            results['word_freq'].plot(20, cumulative=False)
            plt.title("Top 20 Words")
            plt.xlabel("Words")
            plt.ylabel("Frequency")

        elif analysis_type == "machine_learning":
            if visualization_type == "feature_importance":
                feature_importance = pd.Series(results['feature_importance'], index=df.columns[:-1])
                feature_importance.nlargest(10).plot(kind='barh')
                plt.title("Top 10 Feature Importances")
                plt.xlabel("Importance")
            elif visualization_type == "learning_curve":
                train_scores_mean = np.mean(results['train_scores'], axis=1)
                train_scores_std = np.std(results['train_scores'], axis=1)
                test_scores_mean = np.mean(results['test_scores'], axis=1)
                test_scores_std = np.std(results['test_scores'], axis=1)
                plt.fill_between(results['train_sizes'], train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
                plt.fill_between(results['train_sizes'], test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(results['train_sizes'], train_scores_mean, 'o-', color="r", label="Training score")
                plt.plot(results['train_sizes'], test_scores_mean, 'o-', color="g", label="Cross-validation score")
                plt.title("Learning Curves")
                plt.xlabel("Training examples")
                plt.ylabel("Score")
                plt.legend(loc="best")
            elif visualization_type == "confusion_matrix":
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")

        elif analysis_type == "deep_learning":
            plt.plot(results['history']['loss'], label='Training Loss')
            plt.plot(results['history']['val_loss'], label='Validation Loss')
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

                elif analysis_type == "bayesian":
            plt.plot(results['x'], results['prior'], label='Prior')
            plt.plot(results['x'], results['likelihood'], label='Likelihood')
            plt.plot(results['x'], results['posterior'], label='Posterior')
            plt.title("Bayesian Analysis")
            plt.xlabel("Value")
            plt.ylabel("Probability Density")
            plt.legend()

        elif analysis_type == "survival":
            if visualization_type == "kaplan_meier":
                results['survival_function'].plot()
                plt.title("Kaplan-Meier Survival Curve")
                plt.xlabel("Time")
                plt.ylabel("Survival Probability")

        elif analysis_type == "anomaly_detection":
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=results['anomalies'], cmap='viridis')
            plt.title("Anomaly Detection")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.colorbar(label='Anomaly Score')

        # Add general visualization types that can be applied to multiple analysis types
        if visualization_type == "heatmap":
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.title(f"Correlation Heatmap - {analysis_type}")
        elif visualization_type == "pair_plot":
            sns.pairplot(df)
            plt.suptitle(f"Pair Plot - {analysis_type}", y=1.02)

        plt.tight_layout()
        output_filename = f"visualization_{analysis_type}_{visualization_type}.{output_format}"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Visualization saved as {output_filename}")
        print(f"Visualization saved as {output_filename}")

    except Exception as ex:
        logging.error(f"Error visualizing data: {str(ex)}")
        print(f"Error visualizing data: {str(ex)}")

def export_data(df: pd.DataFrame, results: Dict[str, Any], analysis_type: str, output_format: str) -> None:
    """
    Export the analyzed data in the specified format

    Args:
        df (pandas.DataFrame): Original dataframe.
        results (dict): Results from the analysis.
        analysis_type (str): Type of analysis performed.
        output_format (str): Output format for the data.
    """
    try:
        if analysis_type == "regression":
            output_df = pd.DataFrame({'Actual': results['y'], 'Predicted': results['predictions']})
        elif analysis_type == "clustering":
            output_df = df.copy()
            output_df['Cluster'] = results['clusters']
        elif analysis_type == "dimensionality_reduction":
            output_df = pd.DataFrame(results['reduced_data'], columns=['PC1', 'PC2'])
        elif analysis_type == "time_series":
            output_df = pd.DataFrame(results['forecast'], columns=['Forecast'])
        elif analysis_type == "nlp":
            output_df = pd.DataFrame(results['tokens'], columns=['Tokens'])
        elif analysis_type in ["machine_learning", "deep_learning"]:
            output_df = pd.DataFrame(results['predictions'], columns=['Predictions'])
        elif analysis_type == "bayesian":
            output_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
        elif analysis_type == "survival":
            output_df = results['survival_function'].reset_index()
        elif analysis_type == "anomaly_detection":
            output_df = df.copy()
            output_df['Anomaly'] = results['anomalies']
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        if output_format == 'csv':
            output_df.to_csv(f'analyzed_data_{analysis_type}.csv', index=False)
        elif output_format == 'xlsx':
            output_df.to_excel(f'analyzed_data_{analysis_type}.xlsx', index=False, engine='openpyxl')
        elif output_format == 'html':
            output_df.to_html(f'analyzed_data_{analysis_type}.html', index=False)
        elif output_format == 'json':
            output_df.to_json(f'analyzed_data_{analysis_type}.json', orient='records')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        logging.info(f"Analyzed data exported as analyzed_data_{analysis_type}.{output_format}")
        print(f"Analyzed data exported as analyzed_data_{analysis_type}.{output_format}")

    except Exception as ex:
        logging.error(f"Error exporting data: {str(ex)}")
        print(f"Error exporting data: {str(ex)}")

def main():
    """
    Main function to run the data analysis and visualization application.
    """
    try:
        # Get user inputs
        data_file = input("Enter the name of the data file: ")

        print("\nAvailable analysis types:")
        print("1. Regression")
        print("2. Clustering")
        print("3. Dimensionality Reduction")
        print("4. Time Series Analysis")
        print("5. Natural Language Processing")
        print("6. Machine Learning")
        print("7. Deep Learning")
        print("8. Bayesian Analysis")
        print("9. Survival Analysis")
        print("10. Anomaly Detection")
        analysis_type = input("Enter the number(s) of the analysis type(s) you want to perform (comma-separated): ")
        analysis_types = [int(x.strip()) for x in analysis_type.split(',')]

        print("\nAvailable visualization types:")
        print("1. Line Graph")
        print("2. Bar Chart")
        print("3. Scatter Plot")
        print("4. Histogram")
        print("5. Heatmap")
        print("6. Pair Plot")
        print("7. Residual Plot")
        print("8. Learning Curve")
        print("9. Confusion Matrix")
        print("10. Kaplan-Meier Plot")
        print("11. Dendrogram")
        visualization_type = input("Enter the number(s) of the visualization type(s) you want (comma-separated): ")
        visualization_types = [int(x.strip()) for x in visualization_type.split(',')]

        print("\nAvailable output formats:")
        print("1. CSV")
        print("2. Excel")
        print("3. HTML")
        print("4. JSON")
        output_format = int(input("Enter the number of the output format you want: "))

        print("\nAvailable visualization output formats:")
        print("1. PNG")
        print("2. SVG")
        print("3. PDF")
        print("4. JPEG")
        vis_output_format = int(input("Enter the number of the visualization output format you want: "))

        # Load and clean data
        df = load_data(data_file)
        df = clean_data(df)

        # Perform analysis
        analysis_map = {
            1: "regression", 2: "clustering", 3: "dimensionality_reduction",
            4: "time_series", 5: "nlp", 6: "machine_learning", 7: "deep_learning",
            8: "bayesian", 9: "survival", 10: "anomaly_detection"
        }
        visualization_map = {
            1: "line", 2: "bar", 3: "scatter", 4: "histogram", 5: "heatmap",
            6: "pair_plot", 7: "residual", 8: "learning_curve", 9: "confusion_matrix",
            10: "kaplan_meier", 11: "dendrogram"
        }
        output_format_map = {1: "csv", 2: "xlsx", 3: "html", 4: "json"}
        vis_output_format_map = {1: "png", 2: "svg", 3: "pdf", 4: "jpg"}

        for analysis in analysis_types:
            try:
                results = perform_analysis(df, analysis_map[analysis])
                for vis_type in visualization_types:
                    visualize_data(df, results, analysis_map[analysis],
                                   visualization_map[vis_type],
                                   vis_output_format_map[vis_output_format])
                export_data(df, results, analysis_map[analysis], output_format_map[output_format])
            except Exception as ex:
                logging.error(f"Error in analysis {analysis_map[analysis]}: {str(ex)}")
                print(f"Error in analysis {analysis_map[analysis]}: {str(ex)}")

        print("Analysis complete. Check the output files and log for details.")

    except Exception as ex:
        logging.error(f"An error occurred: {str(ex)}")
        print(f"An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()