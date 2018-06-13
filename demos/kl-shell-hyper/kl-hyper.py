"""
This script demonstrates the implementation of the incompressible case of the
hyperelastic Kirchhoff--Love shell formulation from

 http://www.public.iastate.edu/~jmchsu/files/Kiendl_et_al-2015-CMAME.pdf

We implement the neo-Hookean elastic potential for simplicity, but changing
this involves altering only the short function psi_el().

This example quasi-statically inflates a membrane with clamped edges, 
modeled geometrically by an explicit B-spline.

Note: Due to the algebraic complexity of the formulation, the form 
compilation in this example takes *much* longer than it does for typical 
forms. 
"""

from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *


####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Specify the number of elements in each direction.
NELu = 10
NELv = 10

# Specify degree in each direction.
degs = [2,2]

# Generate open knot vectors for each direction.
kvecs = [uniformKnots(degs[0],-1.0,1.0,NELu),
         uniformKnots(degs[1],-1.0,1.0,NELv)]

# Generate an explicit B-spline control mesh.  The argument extraDim allows
# for increasing the dimension of physical space beyond that of parametric
# space.  We want to model deformations in 3D, so one extra dimension is
# required of physical space.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs,extraDim=1)

# Create a spline generator with three unknown fields for the displacement
# components.  
splineGenerator = EqualOrderSpline(3,controlMesh)

# Apply clamped boundary conditions to the displacement.  (Pinned BCs are in
# comments, but need more load steps and/or a smaller load to converge.)
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
for side in range(0,2):
    for direction in range(0,2):
        sideDofs = scalarSpline.getSideDofs(direction,side,
                                            ########################
                                            nLayers=2) # clamped BC
                                            #nLayers=1) # pinned BC
                                            ########################
        for i in range(0,3):
            splineGenerator.addZeroDofs(i,sideDofs)

# Write extraction data to the filesystem.
DIR = "./extraction"
splineGenerator.writeExtraction(DIR)


####### Analysis #######

if(mpirank==0):
    print("Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 4

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)
#spline = ExtractedSpline(DIR,QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")
    print("** NOTE: Form compilation in this example is unusually-slow. **")
    print("Please be patient...")
    
# Unknown midsurface displacement
y_hom = Function(spline.V) # in homogeneous representation
y = spline.rationalize(y_hom) # in physical coordinates

# Reference configuration:
X = spline.F

# Current configuration:
x = X + y

# Normalize a vector v.
def unit(v):
    return v/sqrt(inner(v,v))

# Geometrical quantities for the shell midsurface in a configuration x.
def midsurfaceGeometry(x):

    # Covariant basis vectors
    dxdxi = spline.parametricGrad(x)
    a0 = as_vector([dxdxi[0,0],dxdxi[1,0],dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1],dxdxi[1,1],dxdxi[2,1]])
    a2 = unit(cross(a0,a1))

    # Midsurface metric tensor
    a = as_matrix(((inner(a0,a0),inner(a0,a1)),\
                   (inner(a1,a0),inner(a1,a1))))
    # Curvature
    deriv_a2 = spline.parametricGrad(a2)
    b = -as_matrix(((inner(a0,deriv_a2[:,0]),inner(a0,deriv_a2[:,1])),\
                    (inner(a1,deriv_a2[:,0]),inner(a1,deriv_a2[:,1]))))
    
    return (a0,a1,a2,deriv_a2,a,b)

# Obtain shell geometry for reference and current configuration midsurfaces
A0,A1,A2,deriv_A2,A,B = midsurfaceGeometry(X)
a0,a1,a2,deriv_a2,a,b = midsurfaceGeometry(x)

# Generate a curvilinear basis from midsurface data at the through-thickness
# coordinate xi2, based geometrical data from the midsurface.
def curvilinearBasis(a0,a1,deriv_a2,xi2):
    g0 = a0 + xi2*deriv_a2[:,0]
    g1 = a1 + xi2*deriv_a2[:,1]
    return g0,g1

# Generate a metric tensor at through-thickness coordinate xi2, based on the
# midsurface metric and curvature.  
def metric(a,b,xi2):
    return a - 2.0*xi2*b

# Obtain a local Cartesian basis, given a curvilinear basis a0, a1.
def localCartesianBasis(a0,a1):

    # Perform Gram--Schmidt orthonormalization to get e0 and e1.
    e0 = unit(a0)
    e1 = unit(a1 - e0*inner(a1,e0))

    return e0, e1

# Obtain the local Cartesian representation of the tensor T, represented in the
# basis a0,a1, at a point with metric a.  Note that, for points away from the
# midsurface, the metric (as defined by the function metric()) does not follow
# directly from the curvilinear basis (due to dropping of terms quadratic in
# the through-thickness coordinate), and is therefore not redundant.
#
# Note that while the cited paper by Kiendl et al. works primarily with
# curvilinear coordinates, we assume that the strain energy density is given
# in Cartesian coordinates, and implement something equivalent to what is
# described in the discussion preceding (52)--(53).  
def tensorToCartesian(T,a,a0,a1):
    
    # raise indices on curvilinear basis
    ac = inv(a)
    a0c = ac[0,0]*a0 + ac[0,1]*a1
    a1c = ac[1,0]*a0 + ac[1,1]*a1

    e0,e1 = localCartesianBasis(a0,a1)

    ea = as_matrix(((inner(e0,a0c),inner(e0,a1c)),\
                    (inner(e1,a0c),inner(e1,a1c))))
    ae = ea.T

    return ea*T*ae

# Return a 3D elastic strain energy density, given E in Cartesian coordinates.
def psi_el(E):
    # Neo-Hookean potential, as an example:
    mu = Constant(1e4)
    C = 2.0*E + Identity(3)
    I1 = tr(C)
    return 0.5*mu*(I1 - 3.0)
    
# Compute the pressure Lagrange multiplier for the incompressibility
# constraint, given E in Cartesian coordinates.
def p(E):
    E = variable(E)
    dpsi_el_dC = 0.5*diff(psi_el(E),E)
    C22 = 2.0*E[2,2] + 1.0
    return 2.0*dpsi_el_dC[2,2]*C22

# Compute the total strain energy density, at coordinate xi^2 in the
# through-thickness direction.
def psi(xi2):
    G = metric(A,B,xi2)
    g = metric(a,b,xi2)
    E_flat = 0.5*(g - G)
    G0,G1 = curvilinearBasis(A0,A1,deriv_A2,xi2)
    E_2D = tensorToCartesian(E_flat,G,G0,G1)
    C_2D = 2.0*E_2D + Identity(2)
    C22 = 1.0/det(C_2D)
    E22 = 0.5*(C22-1.0)
    E = as_tensor([[E_2D[0,0], E_2D[0,1], 0.0],
                   [E_2D[1,0], E_2D[1,1], 0.0],
                   [0.0,       0.0,       E22]])
    C = 2.0*E + Identity(3)
    J = sqrt(det(C))
    
    return psi_el(E) - p(E)*(J-1.0)    

# Shell thickness:
h_th = Constant(0.03)

# Obtain a quadrature rule for numerical through-thickness integration:
N_QUAD_PTS = 4
xi2, w = getQuadRuleInterval(N_QUAD_PTS,h_th)

# Define the numerically-integrated energy:
energySurfaceDensity = 0.0
for i in range(0,N_QUAD_PTS):
    energySurfaceDensity += psi(xi2[i])*w[i]

# The total elastic potential energy:
Wint = energySurfaceDensity*spline.dx

# Take the Gateaux derivative of Wint in test function direction z_hom.
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)
dWint = derivative(Wint,y_hom,z_hom)

# External follower load magnitude:
PRESSURE = Constant(1e2)

# Divide loading into steps to improve nonlinear convergence.
N_STEPS = 100
DELTA_T = 1.0/float(N_STEPS)
stepper = LoadStepper(DELTA_T)

# Parameterize loading by a pseudo-time associated with the load stepper.
dWext = -(PRESSURE*stepper.t)*sqrt(det(a)/det(A))*inner(a2,z)*spline.dx

# Full nonlinear residual:
res = dWint + dWext

# Consistent tangent:
dRes = derivative(res,y_hom)

# Allow many nonlinear iterations.
spline.maxIters = 100

# Files for output:  Because an explicit B-spline is used, we can simply use
# the homogeneous (= physical) representation of the displacement in a
# ParaView Warp by Vector filter.

d0File = File("results/disp-x.pvd")
d1File = File("results/disp-y.pvd")
d2File = File("results/disp-z.pvd")

# Iterate over load steps.
for i in range(0,N_STEPS):
    if(mpirank==0):
        print("------- Step: "+str(i+1)+" , t = "+str(stepper.tval)+" -------")

    # Execute nonlinear solve.
    spline.solveNonlinearVariationalProblem(res,dRes,y_hom)

    # Advance to next load step.
    stepper.advance()

    # Output solution.
    (d0,d1,d2) = y_hom.split()
    d0.rename("d0","d0")
    d1.rename("d1","d1")
    d2.rename("d2","d2")
    d0File << d0
    d1File << d1
    d2File << d2

    
####### Postprocessing #######

# Because we are using an explicit B-spline with unit weights on all control
# points and equal physical and parametric spaces, it is sufficient to simply
# load all three displacement files, apply the Append Attributes filter to
# combine them all, use the Calculator filter to produce the vector field
#
#  d0*iHat + d1*jHat + d2*kHat
#
# and then apply the Warp by Vector filter.
