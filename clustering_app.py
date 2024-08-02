import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load the models
def load_kmeans_model():
    try:
        with open('kmeans_model1.pkl', 'rb') as file:
            kmeans_model = pickle.load(file)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        st.error(f"Error loading the KMeans model: {e}")
        return None
    return kmeans_model

def load_final_model():
    try:
        with open('final_model.sav', 'rb') as file:
            final_model = pickle.load(file)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        st.error(f"Error loading the final model: {e}")
        return None
    return final_model

st.markdown('<style>body{background-color:Blue;}</style>', unsafe_allow_html=True)
st.title('Product Clusters')

# Display author name
st.write("Author: Tushar Dupare")

kmeans_model = load_kmeans_model()
loaded_model = load_final_model()
df = pd.read_csv('Clustered_Customer_Data.csv')

# Display the DataFrame structure
st.write("DataFrame structure:")
st.write(df.head())


if kmeans_model is not None and loaded_model is not None:
    def main():
        with st.form("my_form"):
            st.title("Sales Data Input")

            st.header("Warehouse Sales")
            warehouse_sales = st.number_input("Enter warehouse sales", min_value=0, step=1)

            st.header("Retail Sales")
            retail_sales = st.number_input("Enter retail sales", min_value=0, step=1)

            st.header("Retail Transfers")
            retail_transfers = st.number_input("Enter retail transfers", min_value=0, step=1)

            submitted = st.form_submit_button("Submit")

            if submitted:
                data = np.array([[warehouse_sales, retail_sales, retail_transfers]])
                st.write("Input data for prediction:")
                st.write(data)

                # Normalize the data if required (assuming the model was trained on normalized data)
                # scaler = preprocessing.StandardScaler().fit(df[['Warehouse Sales', 'Retail Sales', 'Retail Transfers']])
                # data = scaler.transform(data)
                # st.write("Normalized data for prediction:")
                # st.write(data)

                clust = kmeans_model.predict(data)[0]
                st.write(f'Data Belongs to Cluster {clust}')

                if 'Cluster' in df.columns:
                    cluster_df1 = df[df['Cluster'] == clust]
                    plt.rcParams["figure.figsize"] = (20, 3)
                    
                    # Using FacetGrid to create the plots
                    for c in cluster_df1.drop(['Cluster'], axis=1).columns:
                        grid = sns.FacetGrid(cluster_df1, col='Cluster')
                        grid.map(plt.hist, c, bins=30, edgecolor='k')
                        for ax in grid.axes.flat:
                            for label in ax.get_xticklabels():
                                label.set_rotation(45)
                        st.pyplot(grid.fig)
                else:
                    st.error("'Cluster' column not found in the DataFrame. Please check your data.")

    if __name__ == "__main__":
        main()
else:
    st.error("Failed to load the KMeans model or final model. Please ensure the model files are available and not corrupted.")







    