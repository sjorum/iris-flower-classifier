{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a27d623b-afe8-4736-9b30-0507c9f4f574",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd  \n",
    "import numpy as np  \n",
    "  \n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "af93f551-d14d-4215-b370-08c7cf7a8962",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (2.1.4)\n",
      "Requirement already satisfied: numpy<2,>=1.23.2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas) (1.26.4)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: numpy in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (1.2.2)\n",
      "Requirement already satisfied: numpy>=1.17.3 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from scikit-learn) (1.26.4)\n",
      "Requirement already satisfied: scipy>=1.3.2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from scikit-learn) (1.10.1)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from scikit-learn) (1.2.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: streamlit in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (1.30.0)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (4.2.2)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: importlib-metadata<8,>=1.4 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (7.0.1)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (23.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (2.1.4)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (10.2.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (3.20.3)\n",
      "Requirement already satisfied: pyarrow>=6.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: python-dateutil<3,>=2.7.3 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (2.8.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (4.9.0)\n",
      "Requirement already satisfied: tzlocal<6,>=1.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (2.1)\n",
      "Requirement already satisfied: validators<1,>=0.2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (0.18.2)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (6.3.3)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.3)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: zipp>=0.5 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from importlib-metadata<8,>=1.4->streamlit) (3.17.0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.2.2)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: decorator>=3.4.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from validators<1,>=0.2->streamlit) (5.1.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\user\\programingfiles\\anaconda\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip3 install pandas  \n",
    "!pip3 install numpy  \n",
    "!pip3 install scikit-learn  \n",
    "!pip3 install streamlit  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b8005d40-1e05-496f-ae93-2cdf714fecec",
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_df = pd.read_csv('iris.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "62440743-8f69-4a50-9f86-1e88cb1a44f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method NDFrame.head of       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \\\n",
       "0      1            5.1           3.5            1.4           0.2   \n",
       "1      2            4.9           3.0            1.4           0.2   \n",
       "2      3            4.7           3.2            1.3           0.2   \n",
       "3      4            4.6           3.1            1.5           0.2   \n",
       "4      5            5.0           3.6            1.4           0.2   \n",
       "..   ...            ...           ...            ...           ...   \n",
       "145  146            6.7           3.0            5.2           2.3   \n",
       "146  147            6.3           2.5            5.0           1.9   \n",
       "147  148            6.5           3.0            5.2           2.0   \n",
       "148  149            6.2           3.4            5.4           2.3   \n",
       "149  150            5.9           3.0            5.1           1.8   \n",
       "\n",
       "            Species  \n",
       "0       Iris-setosa  \n",
       "1       Iris-setosa  \n",
       "2       Iris-setosa  \n",
       "3       Iris-setosa  \n",
       "4       Iris-setosa  \n",
       "..              ...  \n",
       "145  Iris-virginica  \n",
       "146  Iris-virginica  \n",
       "147  Iris-virginica  \n",
       "148  Iris-virginica  \n",
       "149  Iris-virginica  \n",
       "\n",
       "[150 rows x 6 columns]>"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris_df.head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "66fb960f-89b9-47f4-b1e6-363963d2102d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#  Dropping the Id column    \n",
    "iris_df.drop('Id', axis=1, inplace=True)  \n",
    "# mapping the target column to numeric values for model training  \n",
    "iris_df['Species'] = iris_df['Species'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})  \n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "92d96df6-9c6a-407c-82bd-318731f1e2fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SepalLengthCm    float64\n",
       "SepalWidthCm     float64\n",
       "PetalLengthCm    float64\n",
       "PetalWidthCm     float64\n",
       "Species            int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris_df.shape\n",
    "iris_df.dtypes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ad892472-773b-4e8f-8bae-53ca0fe1d06d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SepalLengthCm    0\n",
       "SepalWidthCm     0\n",
       "PetalLengthCm    0\n",
       "PetalWidthCm     0\n",
       "Species          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "fd9778c0-2ac4-495f-a5a7-b74e476defa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now, we will split the data into the columns   \n",
    "X = iris_df.iloc[:, :-1]    \n",
    "y = iris_df.iloc[:, -1]    \n",
    "# After splitting the data into training and testing data with 30 % of data as test data respectively    \n",
    "from sklearn.model_selection import train_test_split  \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "8f86f505-3627-4167-abda-67bef276ba01",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction Accuracy:  0.9777777777777777\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier  \n",
    "classifier1 = RandomForestClassifier()  \n",
    "classifier1.fit(X_train, y_train)  \n",
    "  \n",
    "y_pred = classifier1.predict(X_test)  \n",
    "# Accuracy  \n",
    "from sklearn.metrics import accuracy_score  \n",
    "score = accuracy_score(y_test, y_pred)  \n",
    "print(\"Prediction Accuracy: \", score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "316027f7-598b-4d9c-8dc4-bf929b9e74f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle   \n",
    "pickle_out1 = open(\"classifier1.pkl\", \"wb\")    #opens a file for writing in binary mode (wb)\n",
    "pickle.dump(classifier1, pickle_out1)          #serializes the model (classifier1) into the file.\n",
    "pickle_out1.close()                            #closes the file after saving.\n",
    "                                  #Here’s a step-by-step breakdown:\n",
    "#open(\"classifier1.pkl\", \"wb\"):\n",
    "#This opens a file named classifier1.pkl in write (w) and binary (b) mode. If the file does not exist, it will be created. If it exists, it will be overwritten.\n",
    "#The result of this function is a file object, which is assigned to the variable pickle_out1.\n",
    "\n",
    "#pickle.dump(classifier1, pickle_out1):\n",
    "#This serializes the classifier1 object (your model) and writes it to the file object (pickle_out1), which points to classifier1.pkl.\n",
    "#pickle_out1.close():\n",
    "\n",
    "#This closes the file once the model has been written to it, ensuring that all resources are released."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "002f4a22-f5e4-4edd-ada1-34b98e9e4e06",
   "metadata": {},
   "outputs": [],
   "source": [
    "#A new file called \"classifier1.pkl\", will be created in the same directory. We can now use Streamlit to deploy our model -\n",
    "import streamlit as st  \n",
    "from PIL import Image  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "8607f54a-ee00-44b8-97de-5b1017588355",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-13 17:14:42.369 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\USER\\programingfiles\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Load the trained model  \n",
    "pickle_in1 = open('classifier1.pkl', 'rb')  \n",
    "classifier1 = pickle.load(pickle_in1)  \n",
    "  \n",
    "def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):  \n",
    "    prediction = classifier1.predict([[sepal_length1, sepal_width1, petal_length1, petal_width1]])  \n",
    "    return prediction  \n",
    "  \n",
    "def main():  \n",
    "    st.title(\"Iris Flower Prediction\")  \n",
    "  \n",
    "    html_temp = \"\"\"  \n",
    "    <div style=\"background-color: #FFFF00; padding: 16px\">  \n",
    "    <h1 style=\"color: #000000; text-align: center;\">Streamlit Iris Flower Classifier ML App</h1>  \n",
    "    </div>  \n",
    "    \"\"\"  \n",
    "  \n",
    "    st.markdown(html_temp, unsafe_allow_html=True)  \n",
    "  \n",
    "    sepal_length1 = st.text_input(\"Sepal Length\", \"Type Here\")  \n",
    "    sepal_width1 = st.text_input(\"Sepal Width\", \"Type Here\")  \n",
    "    petal_length1 = st.text_input(\"Petal Length\", \"Type Here\")  \n",
    "    petal_width1 = st.text_input(\"Petal Width\", \"Type Here\")  \n",
    "    result = \"\"  \n",
    "  \n",
    "    if st.button(\"Predict\"):  \n",
    "        result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)  \n",
    "    st.write('The output of the above is', result)  \n",
    "  \n",
    "if __name__ == '__main__':  \n",
    "    main()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e0454566-f287-442b-ba04-8a8e24d6e456",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
