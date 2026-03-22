from src.train import train
from src.predict import predict

if __name__ == "__main__":
    print("1. Train Model")
    print("2. Predict Price")

    choice = input("Enter choice: ")

    if choice == "1":
        train()
    elif choice == "2":
        predict()
