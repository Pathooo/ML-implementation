import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_iris

class Sigmoid:

    def __call__(self,x):
        return 1/(1+np.exp(-x))
    
    def gradient(self,x):
        return self.__call__(x)*(1-self.__call__(x))

class Log_loss:

    def __init__(self):
        pass

    def gradient(self,y,prediction_y):
        return (prediction_y-y)
        

class Perceptron:
    def __init__(self,n_iterations=20000,activation_func=Sigmoid,loss_func=Log_loss,learning_rate=0.01):

        self.n_iterations=n_iterations
        self.activation=activation_func()
        self.loss=loss_func()
        self.learning_rate=learning_rate

    def fit(self,x,y):

        _,n_features=np.shape(x)
        _,n_outputs=np.shape(y)

        self.weights=np.random.uniform(-1/np.sqrt(n_features),1/np.sqrt(n_features),(n_features,n_outputs))
        self.biases=np.zeros((1,n_outputs))

        for i in tqdm(range(self.n_iterations)):

            linear_function=x.dot(self.weights)+self.biases
            predicted_y=self.activation(linear_function)

            error_gradient=self.activation.gradient(linear_function)*self.loss.gradient(y,predicted_y)
            weight_gradient=x.T.dot(error_gradient)
            bias_gradient=np.sum(error_gradient,keepdims=True,axis=0)

            self.weights-=weight_gradient*self.learning_rate
            self.biases-=bias_gradient*self.learning_rate

    def predict(self,x):
        ans= self.activation(x.dot(self.weights)+self.biases)
        return ans


X, y = load_iris(return_X_y=True)
y=np.asarray(y).reshape(-1,1)
clf = Perceptron()
clf.fit(X, y)
y_new=clf.predict(X[40:70,:])
print(np.round_(np.hstack((y[40:70,:],y_new)),2))
#print(y[40:70])