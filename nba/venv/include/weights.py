# Imports for dataframe manipulation and matrix calculations
import pandas as pd
import numpy

# Reads csv files of our relevant data (stats and salary info) in dataframe form
# then converts that into an interable matrix that will be used to create the weights vector
stats_df = pd.read_csv("statsi.csv")
stats_df.head()
stats_matrix = stats_df.iloc[:,5:].values

salary_df = pd.read_csv("contractsall.csv")
salary_df.head()
salary_vector = salary_df.iloc[:,7:].values

# Reads csv files of our relevant data (stats and salary info) in dataframe form
# then converts that into an interable matrix that will be used to create our testing data
test1_df = pd.read_csv("testii.csv")
test1_df.head()
test_stats = test1_df.iloc[:,5:].values

test2_df = pd.read_csv("testcall.csv")
test2_df.head()
test_contracts = test2_df.iloc[:,7:].values


class LinearModel:
    """
    A class used to represent a Linear statistical
    model of multiple variables. This model takes
    a vector of input variables and predicts that
    the measured variable will be their weighted sum.
    """

    def __init__(self, weights):
        """
        Create a new LinearModel.
        """
        self._weights = weights

    def __str__(self):
        """
        Return weights as a human readable string.
        """
        return str(self._weights)

    def get_weights(self):
        """
        Return the weights associated with the model.
        """
        return self._weights

    def generate_predictions(self, inputs):
        """
        Use this model to predict a matrix of
        measured variables given a matrix of input data.
        """

        return numpy.matmul(inputs, self._weights)

def fit_least_squares(input_data, output_data):
    """
    Create a Linear Model which predicts the output vector
    given the input matrix with minimal Mean-Squared Error.
    """
    # This function's code follows the formula for finding the weights
    # that create the least mean-squared error, which is:
    #  w = (((y_t)x)(inv((x_t)x))_t)

    xtx = numpy.matmul(numpy.transpose(input_data),input_data)
    xtx_inv = numpy.linalg.inv(xtx)
    ytx = numpy.matmul(numpy.transpose(output_data),input_data)

    return LinearModel(numpy.transpose(numpy.matmul(ytx,xtx_inv)))




def inflation (years, salaries):
    """
    Modifies a vector of salaries to account for inflation of players' salaries
    Determines (based on regression) how much a salary would be today based on the year it was in
    Inputs a salary vector and a corresponding years vector
    """
    for i in range(len(salaries)):
        salaries[i][0] = float(salaries[i][0])/(0.368+0.025*(years[i][0]-2000))

def mse (vec1, vec2):
    """
    Calculates the mean squared error between two vectors of the same length (presumably a predicted value and an actual value)
    """
    sum = 0.0   #Initializes sum to 0
    count = len(vec1)      #Number of total elements in each vector
    for i in range(count):
        sum += (vec2[i]-vec1[i])**2      #Adds the square of the difference between the values at each position in the two vectors
    return sum/count



weights = fit_least_squares(stats_matrix,salary_vector)    #Creates weights vector
weight = weights.get_weights()
print(weight)



#reb, ast, pts, PER, TO%, USG, BPM, VORP, ORtg

#2018-19
greeni = [[4.0,1.6,10.3,13.0,10.4,14.1,2.9,2.7,119.0]]   #Danny Green's stats in the categories we are using

butler = [[5.1,4.0,19.0,20.6,8.3,22.4,3.5,2.3,122.0]]
bazemore = [[4.0,2.3,12.0,12.3,14.0,22.1,-1.9,0.0,99.0]]
klay = [[4.0,2.4,22.2,17.2,7.2,26.0,-0.5,0.9,110.0]]
evans = [[2.8,2.4,10.1,11.9,14.0,26.5,-3.2,-0.4,94.0]]
redick = [[2.4,2.7,17.6,14.4,8.2,21.6,-1.0,0.6,117.0]]
kcp = [[2.7,1.1,10.1,12.2,7.7,16.8,-1.0,0.5,113.0]]
burks = [[3.7,2.0,9.0,12.7,10.8,19.2,-2.1,0.0,106.0]]
ross = [[3.5,1.6,14.7,14.7,7.8,23.9,-0.8,0.6,105.0]]
shumpert = [[3.0,1.9,7.7,9.2,9.4,14.6,-1.1,0.3,103.0]]
temple = [[2.9,1.4,7.9,8.2,11.2,13.3,-1.3,0.3,101.0]]
lamb = [[5.7,2.1,15.2,16.6,7.1,22.4,-0.4,0.8,112.0]]
hez = [[3.8,1.2,8.1,10.1,13.2,20.5,-4.7,-0.7,93.0]]
holiday = [[4.0,1.9,9.9,9.6,11.2,15.3,-0.8,0.7,99.0]]
lance = [[3.0,2.0,7.2,12.5,15.7,20.8,-2.1,0.0,102.0]]
robinson = [[1.3,0.4,4.1,8.0,9.5,15.3,-4.7,-0.4,101.0]]
hood = [[2.2,1.7,11.3,12.5,6.5,18.2,-2.4,-0.2,115.0]]
mitchell = [[4.1,4.1,23.4,16.7,11.6,31.9,0.1,1.3,103.0]]
seth = [[1.5,0.9,7.2,10.9,10.9,16.0,-1.5,0.1,113.0]]
carter = [[2.5,1.1,7.3,12.2,8.9,16.6,-1.3,0.2,113.0]]
middleton = [[6.1,4.4,17.8,16.2,12.4,24.5,0.7,1.5,109.0]]
chandler = [[4.3,1.7,5.9,8.6,14.4,11.3,-0.5,0.4,108.0]]
bojan = [[4.1,1.9,17.9,16.1,10.0,22.3,0.1,1.2,115.0]]
trier = [[3.1,1.9,10.9,12.2,15.8,21.3,-4.1,-0.8,104.0]]
daniels = [[1.3,0.5,6.0,10.9,10.3,18.1,-3.6,-0.2,106.0]]
satoransky = [[3.5,4.9,8.8,14.3,16.5,14.1,-0.3,0.8,121.0]]
bullock = [[2.7,2.1,11.1,10.2,9.5,15.8,-1.0,0.5,110.0]]
crawford = [[1.2,3.3,6.3,9.0,18.4,19.6,-7.6,-1.4,95.0]]
wade = [[3.8,4.0,14.3,15.5,13.7,27.3,-0.8,0.5,102.0]]
harden = [[6.5,7.5,36.4,30.4,14.4,40.7,11.7,9.3,118.0]]
levert = [[3.8,3.9,13.7,14.9,11.4,24.0,-0.2,0.5,105]]
harris = [[3.1,2.9,14.9,16.5,9.8,18.4,1.5,1.6,121.0]]
hardaway = [[2.8,2.3,14.5,15.2,9.5,22.5,-0.3,0.9,110.0]]

#injured
cousins = [[12.9,5.4,25.2,22.6,18.9,31.9,5.5,3.3,104.0]]
paul = [[5.4,7.9,18.6,24.4,12.5,24.5,7.1,4.3,126.0]]
parker = [[4.9,1.9,12.6,17.1,11.2,24.4,-1.8,0.0,106.0]]
curry = [[2.6,2.7,12.8,15.5,11.0,19.5,1.4,1.7,113.0]]
lavine = [[3.9,3.0,16.7,14.6,9.7,29.5,-2.7,-0.1,99.0]]
bradley = [[2.5,2.0,14.3,9.6,13.3,23.5,-4.2,-0.8,92.0]]
thomas = [[2.1,4.8,15.2,12.6,16.6,28.8,-5.2,-0.7,99.0]]
noel = [[5.6,0.7,4.4,16.2,18.1,13.8,1.0,0.4,109.0]]
gay = [[6.3,2.8,18.7,17.9,12.8,25.8,0.7,0.7,105.0]]
evans = [[3.4,3.1,10.3,15.5,12.9,26.7,-0.9,0.2,99.0]]
hill = [[3.4,4.2,16.9,19.3,10.9,23.5,3.6,2.2,119.0]]
lowry = [[4.8,7.0,22.4,22.9,13.8,24.9,6.6,4.9,123.0]]
waiters = [[3.3,4.3,15.8,14.5,12.5,26.3,-0.9,0.4,101.0]]
ginobili = [[2.5,3.1,9.6,17.8,17.0,23.6,3.6,1.6,107.0]]
kiddg = [[7.6,1.4,10.9,16.9,9.8,18.0,-1.4,0.0,106.0]]
jack = [[4.3,7.4,12.8,14.5,18.8,21.7,-1.4,0.2,101.0]]
noah = [[8.8,3.8,4.3,14.1,25.4,14.2,1.9,0.6,98.0]]
gordon = [[2.2,2.7,15.2,13.5,10.7,22.1,-1.4,0.4,108.0]]
jefferson = [[6.4,1.5,12.0,18.2,5.8,24.2,-1.1,0.2,101.0]]


val = numpy.matmul(greeni,weight)    #Danny Green's predicted salary/worth (not accounting for age)


print(val)

predict = numpy.matmul(test_stats,weight)

mse = mse(predict,test_contracts)    #Calculates error (how accurate the model is) based on mean squred error
#print(mse)
