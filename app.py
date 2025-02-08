import streamlit as st


# train a model for Iris classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a random forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# make predictions and show the iris class
def iris_classification(features):
    pred = model.predict([features])
    return iris.target_names[pred][0]





def main():


    # create a side bar
    st.sidebar.title("Streamlit App")

    # take radio input
    page = st.sidebar.radio("Select a page", ["Introduction", "Input & Output Methods", "Iris Classification"])

    if page == "Introduction":
        st.title("Introduction")
        st.write("Hello, World!")
        st.write("This is a streamlit app created by me.")

    elif page == "Input & Output Methods":
        # take radio input
        choice = st.radio("Select a user type", ["Student", "Teacher", "Parent"])
        # take text input
        name = st.text_input("Enter your name")
        # take number input 
        age = st.number_input("Enter your age", min_value=1, max_value=100)
        # take date input 
        dob = st.date_input("Enter your date of birth")

        # take input weight using a slider 
        weight = st.slider("Enter your weight", 1, 150)

        # show the weight 
        st.write(f"Your weight is {weight} kg")

        # take height of the user 
        height = st.slider("Enter your height", 1, 250)

        # show the height
        st.write(f"Your height is {height} cm")

        # show the bmi of the user
        bmi = weight / ((height/100)**2)
        st.write(f"Your BMI is {bmi}")


    elif page == "Iris Classification":
        st.title("Iris Classification")
        # take input features from the user
        sl = st.number_input("Enter sepal length", min_value=0.0, max_value=10.0)
        sw = st.number_input("Enter sepal width", min_value=0.0, max_value=10.0)
        pl = st.number_input("Enter petal length", min_value=0.0, max_value=10.0)
        pw = st.number_input("Enter petal width", min_value=0.0, max_value=10.0)


       

        if st.button("Predict class"):
            features = [sl, sw, pl, pw]
            prediction = iris_classification(features)
            st.write(f"The iris class is: {prediction}")




if __name__ == "__main__":
    main()

