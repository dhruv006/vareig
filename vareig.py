from pylab import *
from scipy.integrate import quad
from numpy import linalg as LA

###-----------------------------------------------###
###               Define parameters               ###
###-----------------------------------------------###

L = 2
Nbasis = 15
k = 100

###-----------------------------------------------###
###     Matrix elements of the kinetic energy     ###
###                   operator                    ###
###-----------------------------------------------###

def KE(m, j):
    if m == j:
       ke = (1 / 2) * ((m * pi / L) ** 2)
    else:
       ke = 0
    return ke

###-----------------------------------------------###
###                Basis functions                ###
###-----------------------------------------------###

def basis(m, x):
    if m % 2 == 0:
       basis = sin(m * pi * x / L)
    else:
       basis = cos(m * pi * x / L)
    return basis

###-----------------------------------------------###
###               Potential energy                ###
###-----------------------------------------------###

def steppot(x):
    pe = (1 / 2) * k * x ** 2
    return pe

###-----------------------------------------------###
###      Integrand in calculation of matrix       ###
###          energy of potential energy           ###
###-----------------------------------------------###

def integrand(x, m, j):
    return basis(m, x) * steppot(x) * basis(j, x)

###-----------------------------------------------###
###    Matrix elements of the potential energy    ###
###      Call scipy function quad to perform      ###
###             numerical integration             ###
###-----------------------------------------------###

def V(m, j):
    result = quad(integrand, -L/2, L/2, args=(m, j))
    return result[0]

###-----------------------------------------------###
###       Initialise the Hamiltonian matrix       ###
###                  with zeros                   ###
###-----------------------------------------------###

H = np.zeros((Nbasis, Nbasis))

###-----------------------------------------------###
###       Assign Hamiltonian matrix elements      ###
###-----------------------------------------------###

for m in range(0, Nbasis):
    for j in range(0, Nbasis):
        H[m][j] = KE(m + 1, j + 1) + V(m + 1, j + 1)

###-----------------------------------------------###
###      Solve eigenvalue problem using numpy     ###
###         and sort in order. Compute the        ###
###       eigenvalues and right eigenvectors      ###
###                  of the array                 ###
###-----------------------------------------------###

eigenValues, eigenVectors = LA.eig(H)

###-----------------------------------------------###
###       Sort eigenvalues and eigenvectors       ###
###-----------------------------------------------###

mdx = eigenValues.argsort()
eigenValues = eigenValues[mdx]

###-----------------------------------------------###
###               Print eigenvalues               ###
###-----------------------------------------------###

for state in range(0, Nbasis):
    print('n =', state + 1, 'E =', eigenValues[state])