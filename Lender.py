import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import hvplot.pandas
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC,Accuracy,BinaryAccuracy
import os
import json
import openai
import math
import time
from datetime import datetime
from collections import defaultdict






class LenderSystem:
    def __init__(self, filepath, load_from = True):
        self.filepath = filepath
        
        if load_from:
            self.data = pd.read_csv('./modified_data.csv')
        else:
            self.data = self.load_and_process_data()
        self.original_data = pd.read_csv('./original_data.csv')
        #print(self.data.dtypes)
        #print(self.data.columns)
        self.X_train = None 
        self.y_train = None 
        self.X_test = None 
        self.y_test = None
        self.X_train_original = None
        self.X_test_original = None
        self.y_train_original = None
        self.y_test_original = None
        self.scenarios = None
        self.fine_tuned_model_id = None
        self.api_key = self.get_api_key()  # get the API key from the file
        self.scenarios_length = 0
        self.conversations = None
        self.prommpts = None
        self.file_id = None
    
    def get_api_key(self):
        with open('apikey.txt', 'r') as file:
            return file.read().strip()


    def load_and_process_data(self):
        # Load your data from the specified filepath.
        data = pd.read_csv(self.filepath)
        data.drop('emp_title', axis=1, inplace=True)
        data.drop('emp_length', axis=1, inplace=True)
        data.drop('title', axis=1, inplace=True)
        #data['mort_acc'] = data.apply(lambda x: self.fill_mort_acc(data, x['total_acc'], x['mort_acc']), axis=1)
        data.dropna(inplace=True)
        data.drop('grade', axis=1, inplace=True)
        data['zip_code'] = data.address.apply(lambda x: x[-5:])
        data.drop('address', axis=1, inplace=True)
        data.drop('issue_d', axis=1, inplace=True)
        data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
        data['earliest_cr_line'] = data.earliest_cr_line.dt.year
        data['mort_acc'] = data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
        data['pub_rec'] = data.pub_rec.apply(pub_rec)
        data['mort_acc'] = data.mort_acc.apply(mort_acc)
        data['pub_rec_bankruptcies'] = data.pub_rec_bankruptcies.apply(pub_rec_bankruptcies)
        data['loan_status'] = data.loan_status.map({'Fully Paid':1, 'Charged Off':0})
        data.to_csv('./original_data.csv',index = False)
        dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status', 
           'application_type', 'home_ownership', 'zip_code']
        data = pd.get_dummies(data, columns=dummies, drop_first=True)
        term_values = {' 36 months': 36, ' 60 months': 60}
        data['term'] = data.term.map(term_values)
        data.to_csv('./modified_data.csv',index=False)
        #print(data[0])
        return data
    
    def fill_mort_acc(self, data, total_acc, mort_acc):
        total_acc_avg = data.groupby(by='total_acc').mean(numeric_only=True).mort_acc
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc].round()
        else:
            return mort_acc
    
    def split_data(self):
        train, test = train_test_split(self.data, test_size=0.33, random_state=42)
        train_original, test_original = train_test_split(self.original_data, test_size=0.33, random_state=42)
        train = train[train['annual_inc'] <= 250000]
        train = train[train['dti'] <= 50]
        train = train[train['open_acc'] <= 40]
        train = train[train['total_acc'] <= 80]
        train = train[train['revol_util'] <= 120]
        train = train[train['revol_bal'] <= 250000]

        train_original = train_original[train_original['annual_inc'] <= 250000]
        train_original = train_original[train_original['dti'] <= 50]
        train_original = train_original[train_original['open_acc'] <= 40]
        train_original = train_original[train_original['total_acc'] <= 80]
        train_original = train_original[train_original['revol_util'] <= 120]
        train_original = train_original[train_original['revol_bal'] <= 250000]

        X_train, y_train = train.drop('loan_status', axis=1), train.loan_status
        X_train_original, y_train_original = train_original.drop('loan_status', axis=1), train_original.loan_status
        X_test, y_test = test.drop('loan_status', axis=1), test.loan_status
        X_test_original, y_test_original = test_original.drop('loan_status', axis=1), test_original.loan_status
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.array(X_train).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        self.X_train = X_train 
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test
        self.X_train_original = X_train_original
        self.X_test_original = X_test_original
        self.y_train_original = y_train_original
        self.y_test_original = y_test_original

    def find_correlation(self, target_variable):
        # 'loan_status' has two unique values "Fully Paid" and "Charged Off"
        self.data['loan_status'] = self.data['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
        # Find correlation of numerical features with target variable
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])
        correlation = numerical_data.corr()[target_variable]
        print(correlation)

        # Plotting the correlations
        plt.figure(figsize=(10,8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
        plt.show()
    
    def print_score(self,true, pred, train=True):
        if train:
            clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
            
        elif train==False:
            clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
            print("Test Result:\n================================================")        
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
    
    def evaluate_nn(self, true, pred, train=True):
        if train:
            clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
            
        elif train==False:
            clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
            print("Test Result:\n================================================")        
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    def plot_learning_evolution(self,r):

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(r.history['loss'], label='Loss')
        plt.plot(r.history['val_loss'], label='val_Loss')
        plt.title('Loss evolution during trainig')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(r.history['AUC'], label='AUC')
        plt.plot(r.history['val_AUC'], label='val_AUC')
        plt.title('AUC score evolution during trainig')
        plt.legend();
    
    def nn_model(self, num_labels, hidden_units, dropout_rates, learning_rate):
        num_columns = self.X_train.shape[1]
        inp = tf.keras.layers.Input(shape=(num_columns, ))
        x = BatchNormalization()(inp)
        x = Dropout(dropout_rates[0])(x)
        for i in range(len(hidden_units)):
            x = Dense(hidden_units[i], activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rates[i + 1])(x)
        x = Dense(num_labels, activation='sigmoid')(x)
    
        model = Model(inputs=inp, outputs=x)
        model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=[AUC(name='AUC'),BinaryAccuracy()])
        
        r = model.fit(
        self.X_train, self.y_train,
        validation_data=(self.X_test, self.y_test),
        epochs=20,
        batch_size=32
        )
        self.plot_learning_evolution(r)
        y_train_pred = model.predict(self.X_train)
        self.evaluate_nn(self.y_train, y_train_pred.round(), train=True)
        model.save('nn.keras')
        return model
    
        
    def ml_model(self,type):
        if type == 'XGB':
            clf = XGBClassifier(use_label_encoder=False)
        elif type == 'RF':
            clf = RandomForestClassifier(n_estimators=100)
        clf.fit(self.X_train, self.y_train)
        y_train_pred = rf_clf.predict(self.X_train)
        print_score(self.y_train, y_train_pred, train=True)
        if type == 'XGB':
            clf.save_model("/Users/amrmohamed/Downloads/openai/XGB.json")
        else:
            joblib.dump(rf, "/Users/amrmohamed/Downloads/openai/random_forest.joblib")



        return clf

    def evaluate_model(self,model,type):

        y_test_pred = model.predict(self.X_test)
        if type == 'nn':
            model = keras.models.load_model('nn.keras')
            evaluate_nn(self.y_test, y_test_pred.round(), train=False)
        if type == 'ml':
            print_score(self.y_test, y_test_pred, train=False)
            disp = RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)
        
        disp = ConfusionMatrixDisplay.from_estimator(
        model, self.X_test, self.y_test, 
        cmap='Blues', values_format='d', 
        display_labels=['Default', 'Fully-Paid']
        )  
    
    def predict_label(self,model,X):
        # Convert the input string to a list
        X_list = eval(X)

        # Convert the list to a pandas Series
        X_series = pd.Series(X_list)

        # Apply the same data processing steps as in load_and_process_data
        #X_series.drop(['emp_title', 'emp_length', 'title'], inplace=True)
        X_series['mort_acc'] = fill_mort_acc(X_series['total_acc'], X_series['mort_acc'])
        #X_series.drop('grade', inplace=True)
        dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status', 
                   'application_type', 'home_ownership']
        X_series = pd.get_dummies(X_series, columns=dummies, drop_first=True)
        X_series['zip_code'] = X_series.address[-5:]
        X_series = pd.get_dummies(X_series, columns=['zip_code'], drop_first=True)
        X_series.drop('address', inplace=True)
        #X_series.drop('issue_d', inplace=True)
        X_series['earliest_cr_line'] = X_series.earliest_cr_line.year
        label = model.predict(X_series)
        if label == 0:
            print("The loan is likely to be Fully Paid.")
        else:
            print("The loan is likely to be Charged Off.")

        return label
    
    def generate_scenarios(self,first_time):
        scenarios = []
        if first_time:
            column_names = ", ".join(self.X_train_original.columns)
            for i in range(len(self.X_train_original)):
                user_data = self.X_train_original.iloc[i,:]
                label = self.y_train[i]
                # Create a narrative around this user's data
                scenario = "A user is applying for a loan with the following data: "
                for (column, value) in zip(self.X_train_original.columns, user_data):
                    scenario += f"The {column} is {value}. "
                # Add a task for GPT-3.5 Turbo
                scenario += f"The model predicts that the loan should be {'accepted' if label == 1 else 'rejected'}. "

                # Add a task for suggestions
                scenario += "If the loan is likely to be rejected, suggest ways the user could improve their chances. "

                # Add a task for explaining financial terms
                scenario += "Explain any financial terms that are relevant to this user's situation."

                scenarios.append(scenario)

                        # Save scenarios to a file
            with open('scenarios.json1', 'w') as f:
                json.dump(scenarios, f)
        
        else:
            with open('scenarios.json1', 'r') as f:
                scenarios = json.load(f)
             
        self.scenarios = scenarios
        self.scenarios_length = len(self.scenarios)

    def create_conversations(self, precentage_of_data=1.0):

        # Convert scenarios to the format expected by the OpenAI API
        prompts = [{"prompt": scenario, "completion": ""} for scenario in self.scenarios]

        prompts = prompts[:math.floor(precentage_of_data*self.scenarios_length)]
        self.prommpts = prompts

        with open('prompts.json1', 'w') as new_file:
            # Write the prompts to the new file
            for prompt in prompts:
                new_file.write(json.dumps(prompt) + '\n')
        print("created prompts")
        
        conversations = []
        for scenario in self.scenarios[:math.floor(precentage_of_data*self.scenarios_length)]:
            system_message = {
                "role": "system", 
                "content": "You are a helpful assistant."
            }
            user_message = {
                "role": "user",
                "content": scenario
            }
            model_message = {
                "role": "assistant",
                "content": "I'm reviewing your request. Please wait for my analysis."
            }
            conversation = {"messages":[system_message, user_message, model_message]}
            conversations.append(conversation)
        
        self.check_conversations(conversations)

        self.conversations = conversations

        with open('conversations.json1', 'w') as new_file:
            # Write the prompts to the new file
            for conversation in conversations:
                new_file.write(json.dumps(conversation) + '\n')

        print("Created Conversations")
    
    def check_conversations(self,conversations):
        # Format error checks
        format_errors = defaultdict(int)

        for ex in conversations:

            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue
                
            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue
                
            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1
                
                if any(k not in ("role", "content", "name") for k in message):
                    format_errors["message_unrecognized_key"] += 1
                
                if message.get("role", None) not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1
                    
                content = message.get("content", None)
                if not content or not isinstance(content, str):
                    format_errors["missing_content"] += 1
            
            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No errors found")
        
    def fine_tune_model(self,uploaded,fine_tuned):

        openai.api_key = self.api_key
 
        if not uploaded:
            #upload File 
            start_upload = time.time()
            
            with open('/Users/amrmohamed/Downloads/openai/conversations.json1','rb') as file:
                response = openai.File.create(
                    file = file,
                    purpose = 'fine-tune'
                )
            self.file_id = response['id']
            # Save the ID to an external file
            with open('file_id.txt', 'w') as file:
                file.write(self.file_id)
            end_upload = time.time()
            print(end_upload-start_upload)
            print("finished uploading")
        else:
            with open('file_id.txt', 'r') as file:
                self.file_id = file.read().strip()
            print("File is already Uploaded!")

        # # Fine-tune the model
        # response = openai.FineTune.create(
        #     api_key= openai.api_key,
        #     model="gpt-3.5-turbo",
        #     dataset=prompts,
        #     # ... other parameters ...
        # )

        if not fine_tuned:
            start_fine = time.time()
            fine_tune_response = openai.FineTuningJob.create(
                training_file=self.file_id,
                model='gpt-3.5-turbo',
                #messages=self.conversations
            )
            model_id = openai.FineTune.retrieve(fine_tune_response.id).fine_tuned_model
            
            #model_id = response['id']
            end_fine = time.time()
            print(end_fine - start_fine)
            # Save the ID of the fine-tuned model
            self.fine_tuned_model_id = model_id
            print("Finished Fine-Tuning Model")

            # Save the ID to an external file
            with open('model_id.txt', 'w') as file:
                file.write(self.fine_tuned_model_id)
        
        else:
            # with open('model_id.txt', 'r') as file:
            #     self.fine_tuned_model_id = file.read().strip()
            print("Model is already Fine-Tuned!")
 

   
    def post(self, request):

        openai.api_key = self.api_key
        # Get the user's message from the request
        message = json.loads(request.body)["message"]

        # Read the ID of the fine-tuned model from the file
        with open('model_id.txt', 'r') as file:
            fine_tuned_model_id = file.read().strip()
            self.fine_tuned_model_id = fine_tuned_model_id

        openai.fine_tuned

        # Generate a response using the fine-tuned model
        response = openai.Completion.create(
            #engine="davinci",
            #model="gpt-3.5-turbo-" + self.fine_tuned_model_id,
            model=self.fine_tuned_model_id,
            prompt=message
            # ... other parameters ...
        )

        # Return the response as a JSON object
        return JsonResponse({"response": response.choices[0].text.strip()})


def pub_rec(number):
    if number == 0.0:
        return 0
    else:
        return 1
    
def mort_acc(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number
    
def pub_rec_bankruptcies(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number

def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc