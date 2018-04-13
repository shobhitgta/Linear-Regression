import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import math

## function to get error for current prediction using Theta
def getError(X, Y, Theta):
	error_mat = Y.transpose() - np.dot(Theta.transpose(), X)
	Error = (np.dot(error_mat, error_mat.transpose())/2)
	return Error	

## function for gradient descent
def gradient_descent(X, Y, Theta):
	all_parameters = []
	all_parameters.append(([Theta[0,0], Theta[1,0]], getError(X,Y,Theta)[0,0]))
	i = 0;
	prev = getError(X,Y,Theta)[0,0]
	while(True):
		## initialize learning rate
		alpha = 0.009
		m = 100.0
		Z = np.dot(Theta.transpose(), X) - Y.transpose()
		Update = np.dot(Z, X.transpose())
		Theta = np.subtract(Theta , (alpha)*Update.transpose())
		print('Iteration - ',i,' : Loss - ', getError(X,Y,Theta)[0,0])
		new = getError(X,Y,Theta)[0,0]
		i = i + 1;
		print(Theta)
		all_parameters.append(([Theta[0,0], Theta[1,0]], getError(X,Y,Theta)[0,0]))
		## check for terminating condition
		if(abs(prev - new) < 0.000000000001):
			break
		if(i > 200):
			break
		prev = new

	return all_parameters

def drawContour(Xt, Yt, Zt, parameters):
	fig = plt.figure()
	cset = plt.contour(Xt,Yt,Zt)
	plt.ion()
	for i in range(0,len(parameters)):
		param = parameters[i]
		m = param[0][0]
		b = param[0][1]
		err = param[1]
		print(m,' ',b,' ',err)
		plt.plot(m,b, 'ro')
		plt.pause(0.2)

def drawMesh(Xt, Yt, Zt, parameters):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(Xt, Yt, Zt, edgecolor='none', alpha=0.25)
	plt.ion()
	for i in range(0, len(parameters)):
		param = parameters[i]
		m = param[0][0]
		b = param[0][1]
		err = param[1]
		print(m,' ',b,' ',err)
		ax.scatter(m, b, err ,color = "r", s = 25)
		plt.pause(0.2)

def main():

	## Reading data
	X = np.genfromtxt('linearX.csv', delimiter='\n').reshape(100,1)
	X = X.transpose()
	Y = np.genfromtxt('linearY.csv', delimiter='\n').reshape(100,1)

	##Data Preprocessing
	mean = np.mean(X);
	var = np.var(X);
	X = (X-mean)/math.sqrt(var);
	
	## Initializing thetha to zero
	Theta = np.zeros([2, 1], dtype = float)
	Theta[0,0] = 0.0
	Theta[1,0] = 0.0

	## Adding intercept term to X
	X = np.vstack([X, X[0,:]])
	X[0,:] = 1.0

	print('Initial Error - ', getError(X, Y, Theta))

	## Run batch gradient descent
	parameters = gradient_descent(X, Y, Theta)
	Theta[0,0] = parameters[len(parameters)-1][0][0]
	Theta[1,0] = parameters[len(parameters)-1][0][1]
	
	## Draw mesh or contour
	xt = yt = np.arange(-3.0, 3.0, 0.05)
	Xt, Yt = np.meshgrid(xt, yt)
	zs = np.array([getError(X,Y,np.array([[xt], [yt]]))[0,0] for xt,yt in zip(np.ravel(Xt), np.ravel(Yt))])
	Zt = zs.reshape(Xt.shape)
	drawMesh(Xt, Yt, Zt,parameters)
	#drawContour(Xt, Yt, Zt, parameters)

	## Ploting data and hypothesis
	plt.plot(X[1,:].transpose(), Y, 'ro')
	prediction = np.dot(Theta.transpose(), X);
	plt.plot(X[1,:].transpose(), prediction.transpose(), label='Hypothesis function learned')
	plt.title('Linear Regression')
	plt.xlabel('x')
	plt.ylabel('y')
	legend = plt.legend(loc='lower right', fontsize='small')
	plt.show()	

if __name__ == "__main__":
    main()