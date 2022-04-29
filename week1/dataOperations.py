import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
The DataOperations class takes in a dataframe, a string, or a numpy array and converts it into a
pandas dataframe. It also provides methods to get the statistics of the data, plot the data.
If it doesn't take any value, it creates a dataset itself.
'''
class DataOperations:
    def __init__(self, data=None):
        '''
        The function takes in a dataframe, a string, or a numpy array and converts it into a pandas
        dataframe.
        
        :param data: the data to be used for the model. If None, then the data will be randomly
        generated
        '''
        self.data = data
        if type(self.data) == np.ndarray:
            self.data = pd.DataFrame(self.data)
        elif type(self.data) == pd.DataFrame:
            self.data = data
        elif type(self.data) == str:
            if 'csv' in self.data:
                self.data = pd.read_csv(self.data)
            if 'json' in self.data:
                self.data = pd.read_json(self.data)
        else:
            df = pd.DataFrame(self.create_data(2000, 0, 100), columns=["x", "y", "label"])
            self.data = df

    def create_data(self, number_of_data:int, low_value:int, high_value:int) -> list:
        '''
        Create data between the interval of low_value and high_value.
        
        :param number_of_data: The number of data points to be created
        :type number_of_data: int
        :param low_value: the lowest value of the data
        :type low_value: int
        :param high_value: The maximum value of the data
        :type high_value: int
        :return: a list of lists. Each list contains three elements: x1, x2, and y.
        '''
        data = []
        for i in range(number_of_data):
            x1 = np.random.randint(low=low_value, high=high_value)
            x2 = np.random.randint(low=low_value, high=high_value)
            if x1 < high_value//2 and x2 > high_value//2:
                data.append([x1,x2,1])
            elif x1 < high_value//2 and x2 < high_value//2:
                data.append([x1,x2,0])
            elif x1 > high_value//2 and x2 > high_value//2:
                data.append([x1,x2,0])
            else:
                data.append([x1,x2,1])
        return data

    def get_stats(self):
        """Returns the statistics of the data."""
        return self.data.describe()
    
    def plot_data(self):
        """Plots the numeric data."""
        num_columns = self.data.describe().columns.tolist()
        plt.scatter(self.data[num_columns[0]], self.data[num_columns[1]], c=self.data[num_columns[2]])
        plt.show()
        

def main():
    # 1. koşul, NumPy array'i verilmesi durumu
    data = np.random.randint(0, 100, size=(100, 3))
    opNp = DataOperations(data)

    print(opNp.get_stats())
    print(opNp.plot_data())

    
    # 2. koşul, dosya path'i verilmesi durumu
    pathCsv = "data/data.csv"
    pathJson = "data/iris.json"
    opPathCsv = DataOperations(pathCsv)
    opPathJson = DataOperations(pathJson)

    print(opPathCsv.get_stats())
    print(opPathCsv.plot_data())
    print(opPathJson.get_stats())
    print(opPathJson.plot_data())

    
    # 3. koşul, pandas dataframe'i verilmesi durumu
    dataPd = pd.DataFrame(data)
    opPd = DataOperations(dataPd)

    print(opPd.get_stats())
    print(opPd.plot_data())

    
    # 4. koşul, herhangi bir değer verilmemesi durumu
    opNone = DataOperations()

    print(opNone.get_stats())
    print(opNone.plot_data())
    

if __name__ == "__main__":
    main()
