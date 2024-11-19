import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class EnterpriseAnalytics:
    def __init__(self):
        st.set_page_config(page_title="Enterprise Analytics Suite", layout="wide")
        self.setup_styling()
        self.initialize_session_state()

    def setup_styling(self):
        st.markdown("""
            <style>
               .main {padding: 0 1rem;}
               .stMetric {
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                }
               .stButton>button {
                    background-color: #1f77b4;
                    color: white;
                    border-radius: 4px;
                    padding: 0.5rem 1rem;
                }
               .plot-container {
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    padding: 1rem;
                    background-color: white;
                }
               .data-stats {
                    padding: 1rem;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    margin: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'cleaned_data' not in st.session_state:
            st.session_state.cleaned_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'data_types' not in st.session_state:
            st.session_state.data_types = None

    def detect_data_types(self, df):
        """Automatically detect and categorize columns"""
        type_dict = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_dict['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_dict['datetime'].append(col)
            elif df[col].nunique() < df.shape[0] * 0.05:  # If unique values < 5% of total rows
                type_dict['categorical'].append(col)
            else:
                type_dict['text'].append(col)

        return type_dict

    def load_data(self):
        st.title("Data Master")
        uploaded_file = st.file_uploader(
            "Upload (CSV/Excel/JSON)", type=['csv', 'xlsx', 'json']
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)

                st.session_state.data = df
                st.session_state.data_types = self.detect_data_types(df)

                # Display filter options
                selected_columns = st.multiselect(
                    "Select columns to display:",
                    df.columns.tolist(),
                    default=df.columns.tolist()
                )

                filtered_df = df[selected_columns]
                st.dataframe(filtered_df)

                return df

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        return None

    def display_data_profile(self, df):
        st.subheader("Data Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        # Data types summary
        st.write("Column Types:")
        for type_name, cols in st.session_state.data_types.items():
            if cols:
                st.write(f"- {type_name.title()}: {len(cols)} columns")

    def clean_data(self, df):
        st.header("Data Cleaning & Preprocessing")

        # Initial step: Check if the first row is the header
        if st.checkbox("Is the first row the header?"):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)

        cleaned_df = df.copy()

        st.write("### Current Data Overview")
        st.dataframe(df.head())

        # Display current missing values and statistics
        missing_counts = df.isnull().sum()
        outlier_counts = {}
        for col in st.session_state.data_types['numeric']:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_counts[col] = (z_scores > 3).sum()

        st.write("### Initial Data Issues")
        st.write("#### Missing Values")
        st.dataframe(missing_counts[missing_counts > 0])
        st.write("#### Outliers (Z-Score > 3)")
        st.dataframe(pd.Series(outlier_counts).rename("Outlier Count"))

        # Cleaning operations
        cleaning_options = st.multiselect(
            "Select Cleaning Operations:",
            [
                "Remove Duplicates",
                "Handle Missing Values",
                "Remove Outliers",
                "Remove Columns",
                "Format Data Types",
                "Feature Scaling",
                "Handle Categorical Variables"
            ]
        )

        operation_changes = {}

        if "Remove Duplicates" in cleaning_options:
            duplicate_count = cleaned_df.duplicated().sum()
            cleaned_df = cleaned_df.drop_duplicates()
            operation_changes["Duplicates Removed"] = duplicate_count

        if "Handle Missing Values" in cleaning_options:
            cols_to_handle = st.multiselect("Select columns to handle missing values:", df.columns)
            strategy = st.selectbox(
                "Choose missing value strategy:",
                ["Drop", "Fill Mean/Mode", "Forward Fill", "Backward Fill"]
            )

            for col in cols_to_handle:
                before_missing = cleaned_df[col].isnull().sum()
                if strategy == "Drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                elif strategy == "Fill Mean/Mode":
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                elif strategy == "Forward Fill":
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                elif strategy == "Backward Fill":
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')

                after_missing = cleaned_df[col].isnull().sum()
                operation_changes[f"Missing Values Removed ({col})"] = before_missing - after_missing

        if "Remove Outliers" in cleaning_options:
            cols_to_handle = st.multiselect("Select columns to handle outliers:",
                                            st.session_state.data_types['numeric'])
            method = st.selectbox(
                "Choose outlier detection method:",
                ["Z-Score", "IQR"]
            )

            for col in cols_to_handle:
                before_outliers = outlier_counts[col]
                if method == "Z-Score":
                    z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
                    cleaned_df = cleaned_df[(z_scores < 3) | cleaned_df[col].isnull()]
                elif method == "IQR":
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    cleaned_df = cleaned_df[
                        (cleaned_df[col] >= Q1 - 1.5 * IQR) &
                        (cleaned_df[col] <= Q3 + 1.5 * IQR)
                        ]

                after_outliers = np.abs(stats.zscore(cleaned_df[col].dropna())).sum()
                operation_changes[f"Outliers Removed ({col})"] = before_outliers - after_outliers

        if "Remove Columns" in cleaning_options:
            cols_to_remove = st.multiselect("Select columns to remove:", df.columns)
            cleaned_df = cleaned_df.drop(columns=cols_to_remove)
            operation_changes["Columns Removed"] = cols_to_remove

        if "Format Data Types" in cleaning_options:
            cols_to_handle = st.multiselect("Select columns to format data types:", df.columns)
            for col in cols_to_handle:
                if pd.api.types.is_object_dtype(cleaned_df[col]):
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                        operation_changes[f"Formatted to Datetime ({col})"] = True
                    except:
                        try:
                            cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                            operation_changes[f"Formatted to Numeric ({col})"] = True
                        except:
                            operation_changes[f"Formatting Failed ({col})"] = True

        if "Feature Scaling" in cleaning_options:
            cols_to_scale = st.multiselect("Select columns to scale:", st.session_state.data_types['numeric'])
            scaler = StandardScaler()
            cleaned_df[cols_to_scale] = scaler.fit_transform(cleaned_df[cols_to_scale])
            operation_changes["Scaled Features"] = cols_to_scale

        if "Handle Categorical Variables" in cleaning_options:
            cols_to_encode = st.multiselect("Select columns to encode:", st.session_state.data_types['categorical'])
            encoding = st.selectbox(
                "Choose encoding method:",
                ["One-Hot Encoding", "Label Encoding"]
            )
            for col in cols_to_encode:
                if encoding == "One-Hot Encoding":
                    dummies = pd.get_dummies(cleaned_df[col], prefix=col)
                    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
                    cleaned_df.drop(col, axis=1, inplace=True)
                elif encoding == "Label Encoding":
                    cleaned_df[col] = pd.Categorical(cleaned_df[col]).codes
                operation_changes[f"Encoded ({col})"] = encoding

        st.session_state.cleaned_data = cleaned_df

        st.write("### Cleaning Impact Summary")
        st.dataframe(pd.DataFrame.from_dict(operation_changes, orient='index', columns=["Changes"]))

        return cleaned_df

    def get_cleaning_suggestions(self, df):
        suggestions = []

        # Check for duplicates
        if df.duplicated().sum() > 0:
            suggestions.append(f"Found {df.duplicated().sum()} duplicate rows")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            suggestions.append(f"Found columns with missing values: {', '.join(missing[missing > 0].index)}")

        # Check for potential outliers
        for col in st.session_state.data_types['numeric']:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            if (z_scores > 3).any():
                suggestions.append(f"Potential outliers in {col}")

        return suggestions

    def apply_cleaning_operations(self, df, operations):
        if "Remove Duplicates" in operations:
            df = df.drop_duplicates()

        if "Handle Missing Values" in operations:
            strategy = st.selectbox(
                "Choose missing value strategy:",
                ["Drop", "Fill Mean/Mode", "Forward Fill", "Backward Fill"]
            )

            if strategy == "Drop":
                df = df.dropna()
            elif strategy == "Fill Mean/Mode":
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "Forward Fill":
                df = df.fillna(method='ffill')
            elif strategy == "Backward Fill":
                df = df.fillna(method='bfill')

        if "Remove Outliers" in operations:
            method = st.selectbox(
                "Choose outlier detection method:",
                ["Z-Score", "IQR"]
            )

            for col in st.session_state.data_types['numeric']:
                if method == "Z-Score":
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    df = df[(z_scores < 3)]
                elif method == "IQR":  # Corrected the elif condition here
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[
                        (df[col] >= Q1 - 1.5 * IQR) &
                        (df[col] <= Q3 + 1.5 * IQR)
                        ]

        if "Format Data Types" in operations:
            for col, type_name in st.session_state.data_types.items():
                if col == 'datetime':
                    for dt_col in type_name:
                        df[dt_col] = pd.to_datetime(df[dt_col])
                elif col == 'numeric':
                    for num_col in type_name:
                        df[num_col] = pd.to_numeric(df[num_col])

        if "Feature Scaling" in operations:
            scaler = StandardScaler()
            for col in st.session_state.data_types['numeric']:
                df[col] = scaler.fit_transform(df[[col]])

        if "Handle Categorical Variables" in operations:
            encoding = st.selectbox(
                "Choose encoding method:",
                ["One-Hot Encoding", "Label Encoding"]
            )

            for col in st.session_state.data_types['categorical']:
                if encoding == "One-Hot Encoding":
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
                elif encoding == "Label Encoding":  # Corrected the elif condition here
                    df[col] = pd.Categorical(df[col]).codes

        return df

    def calculate_cleaning_impact(self, original_df, cleaned_df):
        return {
            "Rows Removed": f"{original_df.shape[0] - cleaned_df.shape[0]:,}",
            "Missing Values Fixed": f"{original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum():,}",
            "Columns Modified": f"{np.sum(original_df.dtypes!= cleaned_df.dtypes):,}"
        }

    def analyze_data(self, df):
        st.header("Advanced Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Descriptive Analytics",
                "Temporal Analysis",
                "Distribution Analysis",
                "Correlation Analysis",
                "Pattern Mining",
                "Clustering Analysis"
            ]
        )

        if analysis_type == "Descriptive Analytics":
            self.descriptive_analytics(df)
        elif analysis_type == "Temporal Analysis":
            self.temporal_analysis(df)
        elif analysis_type == "Distribution Analysis":
            self.distribution_analysis(df)
        elif analysis_type == "Correlation Analysis":
            self.correlation_analysis(df)
        elif analysis_type == "Pattern Mining":
            self.pattern_mining(df)
        elif analysis_type == "Clustering Analysis":
            self.clustering_analysis(df)

    def descriptive_analytics(self, df):
        st.subheader("Descriptive Statistics")

        # Numerical summaries
        numeric_summary = df[st.session_state.data_types['numeric']].describe()
        st.dataframe(numeric_summary)

        # Categorical summaries
        if st.session_state.data_types['categorical']:
            st.subheader("Category Distributions")
            for col in st.session_state.data_types['categorical']:
                fig = px.pie(df, names=col, title=f"{col} Distribution")
                st.plotly_chart(fig)
    def detect_and_parse_dates(self, df, column):
        """
        Detects and parses dates in the specified column using pandas' built-in capabilities.
        """
        try:
            # Try common date formats
            formats_to_try = [
                '%Y-%m-%d',  # 2023-12-31
                '%d/%m/%Y',  # 31/12/2023
                '%m/%d/%Y',  # 12/31/2023
                '%Y/%m/%d',  # 2023/12/31
                '%d-%m-%Y',  # 31-12-2023
                '%m-%d-%Y',  # 12-31-2023
                '%Y%m%d',  # 20231231
                '%d.%m.%Y',  # 31.12.2023
                '%Y.%m.%d'  # 2023.12.31
            ]

            for date_format in formats_to_try:
                try:
                    df[column] = pd.to_datetime(df[column], format=date_format)
                    st.success(f"Successfully parsed dates using format: {date_format}")
                    return df
                except ValueError:
                    continue

            # If no specific format works, try the general parser
            df[column] = pd.to_datetime(df[column], infer_datetime_format=True)
            st.success("Successfully parsed dates using automatic format detection")
            return df

        except Exception as e:
            st.error(f"Failed to parse dates in column '{column}': {e}")
            return df
    def temporal_analysis(self, df):
        st.header("Temporal Analysis")

        # Allow user to select date and value columns
        date_column = st.selectbox("Select the date column:", df.columns)
        value_column = st.selectbox("Select the numeric value column:", df.select_dtypes(include=['number']).columns)

        # Attempt to parse the date column if it is not already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            st.write(f"Attempting to parse dates in column: {date_column}")
            df = self.detect_and_parse_dates(df, date_column)

        # Ensure the column is now datetime
        if pd.api.types.is_datetime64_any_dtype(df[date_column]):
            # Date filtering
            min_date, max_date = df[date_column].min(), df[date_column].max()
            st.write(f"Data ranges from {min_date.date()} to {max_date.date()}")

            date_filter = st.date_input(
                "Filter data by date range:",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )

            # Apply date filter
            if date_filter:
                start_date, end_date = date_filter
                df = df[(df[date_column] >= pd.Timestamp(start_date)) & (df[date_column] <= pd.Timestamp(end_date))]

            # Aggregate data for visualization
            st.write("### Temporal Trends")
            time_series = df.groupby(df[date_column])[value_column].sum()

            # Plot time-series data
            st.line_chart(time_series)

            st.write("Filtered Data Preview:")
            st.dataframe(df)

        else:
            st.error("The selected column could not be parsed as a date. Please check your data.")

        def distribution_analysis(self, df):
            st.subheader("Distribution Analysis")

        for col in st.session_state.data_types['numeric']:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[col], name='Histogram'))
            fig.add_trace(go.Box(x=df[col], name='Box Plot'))
            fig.update_layout(title=f"{col} Distribution")
            st.plotly_chart(fig)

    def correlation_analysis(self, df):
        st.subheader("Correlation Analysis")

        numeric_df = df[st.session_state.data_types['numeric']]
        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Correlation Matrix"
        )
        st.plotly_chart(fig)

        # Feature importance
        if len(st.session_state.data_types['numeric']) > 1:
            pca = PCA()
            pca.fit(StandardScaler().fit_transform(numeric_df))

            explained_variance = pd.DataFrame(
                pca.explained_variance_ratio_,
                index=[f'PC{i + 1}' for i in range(len(pca.explained_variance_))],
                columns=['Explained Variance Ratio']
            )

            st.write("Principal Components Analysis:")
            st.dataframe(explained_variance)

    def pattern_mining(self, df):
        st.subheader("Pattern Mining")

        if len(st.session_state.data_types['numeric']) >= 2:
            cols = st.multiselect(
                "Select columns for pattern analysis",
                st.session_state.data_types['numeric'],
                default=st.session_state.data_types['numeric'][:4]
            )

            if cols:
                fig = px.scatter_matrix(df[cols])
                st.plotly_chart(fig)

    def clustering_analysis(self, df):
        st.subheader("Clustering Analysis")

        features = st.multiselect(
            "Select features for clustering",
            st.session_state.data_types['numeric']
        )

        if features:
            n_clusters = st.slider("Number of clusters", 2, 10, 3)

            X = StandardScaler().fit_transform(df[features])
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(X)

            if len(features) >= 2:
                fig = px.scatter(
                    df, x=features[0], y=features[1],
                    color='Cluster',
                    title='Cluster Distribution')
                st.plotly_chart(fig)

            # Cluster profiles
            cluster_profiles = df.groupby('Cluster')[features].mean()
            st.write("Cluster Profiles:")
            st.dataframe(cluster_profiles)

    def generate_report(self, df):
        st.header("Report Generation")

        if st.button("Generate Comprehensive Report"):
            try:
                output = io.BytesIO()

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary statistics
                    df.describe().to_excel(writer, sheet_name='Summary Stats')

                    # Pivot analysis
                    if st.session_state.data_types['datetime']:
                        date_col = st.session_state.data_types['datetime'][0]
                        for metric in st.session_state.data_types['numeric']:
                            pivot = df.pivot_table(
                                index=pd.Grouper(key=date_col, freq='M'),
                                values=metric,
                                aggfunc=['mean', 'sum', 'count']
                            )
                            pivot.to_excel(writer, sheet_name=f'{metric[:28]}_Analysis')

                # Create download link
                output.seek(0)
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'''
                       <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
                           download="analytics_report.xlsx"
                           style="text-decoration:none;">
                           <button style="
                               background-color: #1f77b4;
                               color: white;
                               padding: 12px 18px;
                               border: none;
                               border-radius: 4px;
                               cursor: pointer;">
                               Download Analytics Report
                           </button>
                       </a>
                   '''
                st.markdown(href, unsafe_allow_html=True)
                st.success("Report generated successfully!")

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Section:",
                                ["Data Upload", "Data Cleaning", "Analysis", "Report"]
                                )

        if page == "Data Upload":
            data = self.load_data()
            if data is not None:
                st.session_state.data = data

        elif page == "Data Cleaning":
            if st.session_state.data is not None:
                cleaned_df = self.clean_data(st.session_state.data)
                if cleaned_df is not None:
                    st.session_state.cleaned_data = cleaned_df
            else:
                st.warning("Please upload data first!")

        elif page == "Analysis":
            if st.session_state.cleaned_data is not None:
                self.analyze_data(st.session_state.cleaned_data)
            elif st.session_state.data is not None:
                self.analyze_data(st.session_state.data)
            else:
                st.warning("Please upload and clean data first!")

        elif page == "Report":
            if st.session_state.cleaned_data is not None:
                self.generate_report(st.session_state.cleaned_data)
            elif st.session_state.data is not None:
                self.generate_report(st.session_state.data)
            else:
                st.warning("Please complete analysis first!")


if __name__ == "__main__":
    app = EnterpriseAnalytics()
    app.run()
