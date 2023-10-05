from Lender import LenderSystem
from xgboost import Booster
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# def warn(*args, **kwargs):
#     pass
# warnings.warn = warn
#import warnings
warnings.filterwarnings('ignore')

def main():
    #print("Hello")
    load_from = False
    Lender = LenderSystem('/Users/amrmohamed/Downloads/openai/lending_club_loan_two.csv', load_from)
    #print("Hello")
    while True:
        print("1. Split data")
        print("2. Train Neural Networks")
        print("3. Train XGB Classifier")
        print("4. Train Random Forest Classifier")
        print("5. Evaluate Neural Network")
        print("6. Evaluate XGB Classifier")
        print("7. Evaluate Random Forest Classifier")
        print("8. Predict the probability of loan acceptance with Neural Networks")
        print("9. Predict the probability of loan acceptance with XGB Classifier")
        print("10. Predict the probability of loan acceptance with Random Forest Classifier")
        print("11. Fine Tune ChatGPT Turbo 3.5")
        print("12. Exit")
        print("13. Test")

        choice = input("\nEnter your choice: ")


        if choice == '1':
            Lender.split_data()
            print("\nData has been split into training and testing sets.")
        elif choice == '2':
            num_labels = 1
            hidden_units = [150, 150, 150]
            dropout_rates = [0.1, 0, 0.1, 0]
            learning_rate = 1e-3
            model = Lender.nn_model(num_labels, hidden_units, dropout_rates, learning_rate)
        elif choice == '3':
            model = Lender.ml_model('XGB')
        elif choice == '4':
            model = Lender.ml_model('RF')
        elif choice == '5':
            nn = keras.models.load_model('nn.keras')
            model = Lender.evaluate_model(nn,'nn')
        elif choice == '6':
            xgb = Booster()
            xgb.load_model("XGB.json")
            model = Lender.evaluate_model(xgb,'XGB')
        elif choice == '7':
            rf = joblib.load("./random_forest.joblib")
            model = Lender.evaluate_model(rf,'RF')
        elif choice == '8':
            model = keras.models.load_model('nn.keras')
            X = input("\n please Enter the Data of the Client in a sorted array")
            prediction = Lender.predict_label(model,X)
        elif choice == '9':
            X = input("\n please Enter the Data of the Client in a sorted array as following")
            xgb = Booster()
            xgb.load_model("XGB.json")
            prediction = Lender.predict_label(xgb,X)
        elif choice == '10':
            X = input("\n please Enter the Data of the Client in a sorted array as following")
            rf = joblib.load("./random_forest.joblib")
            prediction = Lender.predict_label(rf,X)
        elif choice == '11':
            first_time = False
            uploaded = True
            fine_tuned = True
            Lender.generate_scenarios(first_time)
            Lender.create_conversations(0.1)
            Lender.fine_tune_model(uploaded,fine_tuned)
        elif choice == '12':
            break
        elif choice == '13':
            print(type(Lender.conversations))
            print(type(Lender.conversations[0]))
            print(type(Lender.file_id))
            print(Lender.file_id)
        else:
            print("\nInvalid choice. Please enter a number between 1 and 13.")

if __name__ == "__main__":
    main()
