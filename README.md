# ml
This is machine learning relative functions.

1. gradientDescent 
This is a class object to create gradient descent method from signal to multiple factors.

a.Function:
  read_data: This function read the data from csv file to pandas dataframe.    

b.Class:

  gradient descent class:
  i.   initialize this object with the pandas dataframe.
  ii.  getX function: Set up the x factor/ factors based on user input (column name from the dataframe) as list object.
  iii. getY function: Set up the y for the prediction.
  iv.  computeCost: compute the cost function by taking the user input "theta". 

![o752n](https://user-images.githubusercontent.com/19805677/52031615-0f09fe00-24e3-11e9-93b8-58c435abf005.png)
  
  v.   computeGradientDescent: This is to compute the theta with gradient descent algorithm.
  
![capture](https://user-images.githubusercontent.com/19805677/52031704-6c9e4a80-24e3-11e9-9fbc-cabe9d312268.JPG)
  
  vi.  graph: Graph the prediction y compare with actual y.  
  
![capture](https://user-images.githubusercontent.com/19805677/52031774-c0109880-24e3-11e9-83aa-3c2189016038.JPG)
