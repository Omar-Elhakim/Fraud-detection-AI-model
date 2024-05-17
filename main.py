from tkinter import *
from tkinter import messagebox
import pickle
import sklearn
import pandas as pd
import numpy as np


# load the trained model
def load_model(filename):
    with open(filename, "rb") as f:
        loaded_svm_model = pickle.load(f)
    return loaded_svm_model


d = {
    "misc_net": 9,
    "grocery_pos": 4,
    "entertainment": 0,
    "gas_transport": 2,
    "misc_pos": 10,
    "grocery_net": 3,
    "shopping_net": 12,
    "shopping_pos": 13,
    "food_dining": 1,
    "personal_care": 11,
    "health_fitness": 5,
    "travel": 14,
    "kids_pets": 8,
    "home": 6,
    "kids_": 7,
}
average = 70.25615704174125
filename = "../rf_model.pth"
loaded_rf_model = load_model(filename)
# load the data
train_df = pd.read_csv("../fraudTrain.csv")  # no need for all the data


# getting prediction
def predict(id, Amount, Time, Name, Merchant, CardNumber, TransitionNumber, Category):
    # category	Amount	is_fraud	Amount_diff_avg	Hour
    r = np.array(
        [d[Category], float(Amount), float(Amount) - average, pd.to_datetime(Time).hour]
    )
    r = r.reshape(1, -1)
    # getting prediction
    return loaded_rf_model.predict(r)


# saving inputs in variables&showing output
def Save_output():

    id = entry1.get()
    Amount = entry2.get()
    Time = entry3.get()
    Name = entry4.get()
    Merchant = entry5.get()
    CardNumber = entry6.get()
    TransitionNumber = entry7.get()
    Category = entry8.get()

    # showing output
    messagebox.showinfo(
        "Test Result",
        (
            "It's Fraud"
            if predict(
                id, Amount, Time, Name, Merchant, CardNumber, TransitionNumber, Category
            )
            else "It's clean"
        ),
    )


root = Tk()


entryposX = 200

# icon&title
# root.iconbitmap("C:\\Users\\legan\\Downloads\\icon (1).ico")
root.title("Fraud Detection")
root.geometry("900x600")
root.configure(bg="grey")
title = Label(
    root, text="Fraud Detection Application", font=("Arial", 20, "bold"), bg="grey"
)
title.pack()

label1 = Label(root, text="Transaction Id:", font=("Arial", 15, "bold"), bg="grey")
label1.pack()
label1.place(y=50)

entry1 = Entry(root, borderwidth=3)
entry1.pack()
entry1.place(x=entryposX, y=55)

label2 = Label(root, text="Amount:", font=("Arial", 15, "bold"), bg="grey")
label2.pack()
label2.place(y=90)

entry2 = Entry(root, borderwidth=3)
entry2.pack()
entry2.place(x=entryposX, y=95)

label3 = Label(root, text="Time:", font=("Arial", 15, "bold"), bg="grey")
label3.pack()
label3.place(y=130)

entry3 = Entry(root, borderwidth=3)
entry3.pack()
entry3.place(x=entryposX, y=135)

label4 = Label(root, text="Cardholder Name:", font=("Arial", 15, "bold"), bg="grey")
label4.pack()
label4.place(y=170)

entry4 = Entry(root, borderwidth=3)
entry4.pack()
entry4.place(x=entryposX, y=175)

label5 = Label(root, text="Merchant:", font=("Arial", 15, "bold"), bg="grey")
label5.pack()
label5.place(y=210)

entry5 = Entry(root, borderwidth=3)
entry5.pack()
entry5.place(x=entryposX, y=215)

label6 = Label(root, text="Card Number:", font=("Arial", 15, "bold"), bg="grey")
label6.pack()
label6.place(y=250)

entry6 = Entry(root, borderwidth=3, show="*")
entry6.pack()
entry6.place(x=entryposX, y=255)

label7 = Label(root, text="Transition Number:", font=("Arial", 15, "bold"), bg="grey")
label7.pack()
label7.place(y=290)

entry7 = Entry(root, borderwidth=3)
entry7.pack()
entry7.place(x=entryposX, y=295)

label8 = Label(root, text="Category:", font=("Arial", 15, "bold"), bg="grey")
label8.pack()
label8.place(y=330)

entry8 = Entry(root, borderwidth=3)
entry8.pack()
entry8.place(x=entryposX, y=335)


button = Button(
    root,
    text="Submit Data",
    command=Save_output,
    font=("Arial", 15, "bold"),
    bg="Lightblue",
    borderwidth=5,
)
button.pack()
button.place(x=390, y=400)

root.mainloop()
