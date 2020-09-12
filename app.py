import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

matplotlib.use('Agg')

def main():
    "Semi auto ML APP with Streamlit"
    st.title("Semi Auto ML APP")
    st.text("Using Streamlit")

    activities = ["EDA","Plot","Model Building"]

    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload Dataset", type=["csv","text",])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            
            if st.checkbox("Select Columns to show"):
                selected_columns = st.multiselect("Select Columns to show",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Summary"):
                st.write(df.describe())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
            
            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
            
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Select 1 Column",all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%2f%%")
                st.write(pie_plot)
                st.pyplot()


    elif choice == 'Plot':
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload Dataset", type=["csv","text",])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        
        st.subheader("Customizable Plots")
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns to Plot",all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))
            

            if type_of_plot == "area":
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            
            elif type_of_plot == "bar":
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            
            elif type_of_plot == "line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)
            
            elif type_of_plot == 'hist':
                custom_plot = df[selected_column_names].plot(kind=type_of_plot,bins=2)
                st.write(custom_plot)
                st.pyplot()
            
            elif type_of_plot == 'box':
                custom_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(custom_plot)
                st.pyplot()

            elif type_of_plot == 'kde':
                custom_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(custom_plot)
                st.pyplot()
		    
            else:
                cust_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()



    elif choice == 'Model Building':
        st.subheader("Building ML Model")

        data = st.file_uploader("Upload Dataset", type=["csv","text",])
        if data is not None:
            df = pd.read_csv(data)
            
            if st.checkbox("Show Features"):
                all_features = df.iloc[:,0:-1]
                st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
                st.dataframe(all_features.head(10))
            
            if st.checkbox("Show Target"):
                all_target = df.iloc[:,-1]
                st.text('Target/Class Name:: {}'.format(all_target.name))
                st.dataframe(all_target.head(10))

            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7

            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            for name,model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"Standard_Deviation":cv_results.std()}
                all_models.append(accuracy_results)

            if st.checkbox("Metric as Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name", "Model Accuracy","Standard Deviation"]))
            
            if st.checkbox("Metric as JSON"):
                st.json(all_models)

if __name__ == "__main__":
    main()
