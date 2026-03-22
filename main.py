<<<<<<< HEAD
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
=======
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
>>>>>>> 318354071a996c8ceb2f9881f6f6bdfc14a25dd7
