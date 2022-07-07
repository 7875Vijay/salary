
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def welcome():
    print("Welcome in salary Predication System")
    print("Please Press ENTER key to Proceed")
    input()
    
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content=os.listdir(cur_dir)
    for file_name in content:
        if file_name.split('.')[-1]=='csv':
            csv_files.append(file_name)   

    return csv_files        


def check_and_select_csv(csv_files):
       i=0
       for file_name in csv_files:
           print(i,'....',file_name)
           i+=1
       return     csv_files[int(input("Select Your CSV file:----> "))]
    
     

def main():
    welcome()
    try:
        csv_files=checkcsv()
        csv_file=check_and_select_csv(csv_files)
        print(csv_file)
        print("CSV file is selected!")
        print("Reading CSV file.........")
        dataset = pd.read_csv(csv_file)
        print("Creating dataset.........")
        x = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        print(dataset)
        print(x)
        print(y)
        s = float(input("Enter the datasize between 0 to 1:-----> "))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = s)
        print("Dataset is created")
        print(x_train)
        print(y_train)
        print("Creating the ml model")
        model = LinearRegression()
        model.fit(x_train, y_train)
        print("ml model is created")
        print(input("please ENTER key for Accuracy:-----> "))
        
        print(x_test)
        a = model.predict(x_test)
        print(a)
        print(y_test)
        accuracy = r2_score(a,y_test)
        print("aur model accuracy is %2.2f%%"%(accuracy*100))
        print("model is ready for use\n press Enter key for use:------> ")
        input()
        print("Enter an Experience in year seperated by ','")
        user = list(map(float, input().split(",")))
        ex = []
        print(user)
        for i in user:
            ex.append([i])

        array = np.array(ex)
        print(array)
        result = model.predict(array)
        print(result)
        print("Your salary is: ")
        result = pd.DataFrame({"Experience":user, "Salaries": result})
        print(result)
        print("Thanks for using aur ML Model")
        # print(user)

        plt.scatter(x_train, y_train)  #provide trining data for scatter
        plt.plot(array, result)        #prvide the test data and result
        plt.show()
    except FileNotFoundError:
        print("csv file not found")


main()
