# Machine learning aglorithm

Machine learning gradient descent aglorithm.

2 class included into this model:
  a. base.py:\
     i. base class: This is the base class for set up your x input and y input. The setup X function can take more than one variable.\
     ii.graph class: This is the graphing class for graph the prediction output. It does not graph the residual.\
     iii.residual class: This is the class using least squares error method to calculate the residual. It can also graph out the residual.\
     \
  b. regression class:\
     i.Linear regression class: This class takes the setup X and Y to do the regressional analysis.\
     ii.Logistic regression class: This is the class takes the setup X and Y to do the logistic regression analysis.\
  

## Getting Started

testObj = GradientDescent(your x, your y, your file path)\
testObj.setX(your x needs to include to the model)\
testObj.setY(your y needs to include to the model)\
testObj.computeGradientDescent(your theta, start alpha, nums of running)\
testObj.graphRegression(your x, your output theta)\
testObj.calcResidual()
testObj.plotResidual()

## Sample Output

![capture](https://user-images.githubusercontent.com/19805677/52031774-c0109880-24e3-11e9-83aa-3c2189016038.JPG)
![capture](https://user-images.githubusercontent.com/19805677/52612604-18cf2200-2e50-11e9-894a-b0e8f6b70d98.JPG)
![capture](https://user-images.githubusercontent.com/19805677/53146474-e9b06300-3569-11e9-95c5-cd04b3264f79.JPG)

### Prerequisites

Pandas, Numpy, Matplotlib

## Versioning

v1.0

## Authors

Mai He - (https://github.com/mablic)

## License

N/A

## Acknowledgments

Data from
https://www.coursera.org/learn/machine-learning/home/welcome
