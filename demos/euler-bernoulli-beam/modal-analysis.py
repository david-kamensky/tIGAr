"""
Solve and plot several modes of the cantilevered Euler--Bernoulli beam, 
using a pure displacement formulation, which would not be possible with 
standard $C^0$ finite elements.  

Note: This demo uses interactive plotting, which may cause errors on systems
without GUIs.  
"""

from tIGAr import *
from tIGAr.BSplines import *
import math
import matplotlib.pyplot as plt


####### Preprocessing #######

# Polynomial degree of the basis functions: must be >1 for this demo, because
# functions need to be at least $C^1$ for the formulation.
p = 3

# Number of elements to divide the beam into:
Nel = 100

# Length of the beam:
L = 1.0

# Create a univariate B-spline.
splineMesh = ExplicitBSplineControlMesh([p,],[uniformKnots(p,0.0,L,Nel),])
splineGenerator = EqualOrderSpline(1,splineMesh)

# Apply Dirichlet BCs to the first two nodes, for a clamped BC.
field = 0
parametricDirection = 0
side = 0
scalarSpline = splineGenerator.getScalarSpline(field)
sideDofs = scalarSpline.getSideDofs(parametricDirection,side,nLayers=2)
splineGenerator.addZeroDofs(field,sideDofs)


####### Analysis #######

QUAD_DEG = 2*p
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

# Displacement test and trial functions:  
u = TrialFunction(spline.V)
v = TestFunction(spline.V)

# Laplace operator:
def lap(f):
    return spline.div(spline.grad(f))

# Material constants for the Euler--Bernoulli beam problem:
E = Constant(1.0)
I = Constant(1.0)
mu = Constant(1.0)

# Elasticity form:
a = inner(E*I*lap(u),lap(v))*spline.dx

# Mass form:
b = mu*inner(u,v)*spline.dx

# Assemble the matrices for a generalized eigenvalue problem.  The reason that
# the diagonal entries for A corresponding to Dirichlet BCs are set to a
# large value is to shift the corresponding eigenmodes to the high end of
# the frequency spectrum.  
A = spline.assembleMatrix(a,diag=1.0/DOLFIN_EPS)
B = spline.assembleMatrix(b)

# Solve the eigenvalue problem, ordering values from smallest to largest in
# magnitude.  
solver = SLEPcEigenSolver(A,B)
solver.parameters["spectrum"]="smallest magnitude"
solver.solve()

# Look at the first N_MODES modes of the problem.
N_MODES = 5
for n in range(0,N_MODES):
    # Due to the structure of the problem, we know that the eigenvalues are
    # real, so we are passing the dummy placeholder _ for the complex parts
    # of the eigenvalue and mode.
    omega2, _, uVectorIGA, _ = solver.get_eigenpair(n)
    print("omega_"+str(n)+" = "+str(math.sqrt(omega2)))

    # The solution from the eigensolver is a vector of IGA DoFs, and must be
    # extracted back to an FE representation for plotting.
    u = Function(spline.V)
    u.vector()[:] = spline.M*uVectorIGA
    plot(u)

plt.autoscale()
plt.show()
