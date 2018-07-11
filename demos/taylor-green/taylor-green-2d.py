"""
Solving the Taylor--Green vortex problem in 2D, using div-conforming B-splines
to obtain pointwise divergence-free velocity solutions.  A comprehensive
reference on div-conforming B-splines applied to incompressible flow problems 
is the PhD thesis of J.A. Evans:

  https://repositories.lib.utexas.edu/bitstream/handle/2152/ETD-UT-2011-12-4506/EVANS-DISSERTATION.pdf

This demo uses NURBS to describe a distorted mesh.
"""

from tIGAr import *
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
from numpy import array
from tIGAr.compatibleSplines import *
from tIGAr.timeIntegration import *
import math


####### Geometry creation #######

if(mpirank==0):
    print("Creating geometry with igakit...")

# Parameter determining level of refinement:
REF_LEVEL = 5

# Open knot vectors for a one-Bezier-element bi-unit square:
uKnots = [-1.0,-1.0,-1.0,1.0,1.0,1.0]
vKnots = [-1.0,-1.0,-1.0,1.0,1.0,1.0]

# Array of control points, for a bi-unit square with the interior
# parameterization distorted, magnified by a factor of pi:
cpArray = math.pi*array([[[-1.0,-1.0],[0.0,-1.0],[1.0,-1.0]],
                         [[-1.0,0.0],[0.7,0.3],[1.0,0.0]],
                         [[-1.0,1.0],[0.0,1.0],[1.0,1.0]]])

# Create initial mesh
ikNURBS = NURBS_ik([uKnots,vKnots],cpArray)

# Refinement
numNewKnots = 1
for i in range(0,REF_LEVEL):
    numNewKnots *= 2
h = 2.0/float(numNewKnots)
numNewKnots -= 1
knotList = []
for i in range(0,numNewKnots):
    knotList += [float(i+1)*h-1.0,]
newKnots = array(knotList)
ikNURBS.refine(0,newKnots)
ikNURBS.refine(1,newKnots)

# Output in PetIGA format
if(mpirank==0):
    PetIGA().write("out.dat",ikNURBS)
MPI.barrier(worldcomm)


####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Create a control mesh from the igakit NURBS object
controlMesh = NURBSControlMesh("out.dat",useRect=True)

# The degrees of the unknown fields here are independent of the control mesh.
# Note: For RT B-splines, "degree 1" includes some quadratic scalar fields (for
# the component of a basis function parallel to its flow direction).  
degs = [1,1]

# Create the RT-type compatible spline space.
splineGenerator = BSplineCompat(controlMesh,"RT",degs)

# Apply strong BCs in parametric normal directions.  While it may at first seem
# wrong, because the constrained DoFs are not strictly normal in the physical
# domain, their coefficients are necessarily zero in any flow field for which
# the physical normal flow is zero (since the parametric tangential direction
# is always physically tangential as well).
for field in range(0,2):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        sideDofs = scalarSpline.getSideDofs(field,side)
        splineGenerator.addZeroDofs(field,sideDofs)

# Write extraction data to filesystem.
DIR = "./extraction"
splineGenerator.writeExtraction("./extraction")


####### Analysis #######

if(mpirank==0):
    print("Setting up the extracted spline...")

# Set up extracted spline.
QUAD_DEG = 4
spline = ExtractedBSplineRT(splineGenerator,QUAD_DEG)
#spline = ExtractedBSplineRT("./extraction",QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")

# Tie the time integration parameters to the mesh refinement level.
TIME_INTERVAL = 1.0
N_STEPS = int(2**REF_LEVEL)
DELTA_T = TIME_INTERVAL/float(N_STEPS)

# The initial condition for the Taylor--Green vortex:
x = spline.spatialCoordinates()
soln0 = sin(x[0])*cos(x[1])
soln1 = -cos(x[0])*sin(x[1])
# (Because igakit produces a 3D physical domain, physical velocities are 3D.)
soln = as_vector([soln0,soln1,Constant(0.0)])

# Gradient of the Taylor--Green initial condition
gradSoln = spline.grad(soln)

# Mass density and dynamic viscosity of the fluid:
DENS = Constant(1.0)
VISC = Constant(0.1)

# The time dependency of the Taylor--Green unsteady solution:
solnt = Expression("exp(-2.0*VISC*t/DENS)",VISC=float(VISC),t=0.0,
                   DENS=float(DENS),degree=1)

# The unknown velocity solution in the parametric domain:
u_hat = Function(spline.V)

# Note that spline.V contains velocity only, and no pressure.  The solution
# algorithm will constrain the velocity to a solenoidal subspace.  The
# pressure can always be postprocessed from the velocity solution.  

# The parametric velocity solution at the previous time step:
u_old_hat = spline.divFreeProject(soln)
# The parametric velocity partial time derivative at the previous time step,
# selected to be consistent with the exact solution:
udot_old_hat = spline.divFreeProject(-2.0*VISC/DENS*soln)

# Create a generalized-alpha time integrator for the velocity.
RHO_INF = Constant(0.5)
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,u_hat,
                                     (u_old_hat, udot_old_hat))

# Get the alpha-level velocity and partial time derivative, pushed forward
# to the physical domain (via the Piola transform).
u = spline.pushforward(timeInt.x_alpha())
udot = spline.pushforward(timeInt.xdot_alpha())

# Helper function defining the symmetric gradient of u:
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# Test function for the velocity, in the parametric and physical domains:
v_hat = TestFunction(spline.V)
v = spline.pushforward(v_hat)

# Material time derivative of the velocity:
Du_Dt = udot + spline.grad(u)*u

# Viscous part of the Cauchy stress:
sigmaVisc = 2.0*VISC*eps(u)

# The problem is posed on a solenoidal subspace, as enforced by iterative
# penalty solution algorithm.  No pressure terms are necessary in the
# variational form.
res = DENS*inner(Du_Dt,v)*spline.dx + inner(sigmaVisc,eps(v))*spline.dx

# Files for time series of the velocity and mesh distortion:
u0File = File("results/u-x.pvd")
u1File = File("results/u-y.pvd")
# (Although this is a 2D problem, igakit produces a 3D control mesh.)
F0File = File("results/F-x.pvd")
F1File = File("results/F-y.pvd")
F2File = File("results/F-z.pvd")
F3File = File("results/F-w.pvd")

# Time stepping loop:
for i in range(0,N_STEPS):

    # Update the time variable in the time-dependent part of the exact
    # solution.
    solnt.t = timeInt.t
    
    if(mpirank == 0):
        print("------- Time step "+str(i+1)
              +" , t = "+str(timeInt.t)+" -------")

    # Use the iterated penalty solution algorithm to obtain a divergence-free
    # velocity field.
    spline.iteratedDivFreeSolve(res,u_hat,v_hat,penalty=Constant(1e6))

    # For visualization, we want the physical components of velocity, but
    # these are not Function objects and cannot be written directly to
    # ParaView files.  Instead, project them onto a space of linears.
    u0 = spline.projectScalarOntoLinears(u[0])
    u1 = spline.projectScalarOntoLinears(u[1])
    u0.rename("u0","u0")
    u1.rename("u1","u1")
    u0File << u0
    u1File << u1

    # Output the control mesh.
    spline.cpFuncs[0].rename("F0","F0")
    spline.cpFuncs[1].rename("F1","F1")
    spline.cpFuncs[2].rename("F2","F2")
    spline.cpFuncs[3].rename("F3","F3")
    F0File << spline.cpFuncs[0]
    F1File << spline.cpFuncs[1]
    F2File << spline.cpFuncs[2]
    F3File << spline.cpFuncs[3]

    # Advance to next time step.
    timeInt.advance()


####### Postprocessing #######

# Check the $L^2$ error at the final time.  Because the time step is kept
# proportional to mesh size, and the time integrator is of second order, this
# will converge at second order, even if higher-order splines are used in
# space.
errRes = u - solnt*soln
L2Error = sqrt(assemble(inner(errRes,errRes)*spline.dx))
if(mpirank == 0):
    print("L2 Error = "+str(L2Error))
    
# Notes on ParaView plotting:
#
# Load all time series, then combine them with the Append Attributes filter.
# Create a Calculator filter to define the displacement from the parametric
# to physical domain:
#
#  (F0/F3 - coordsX)*iHat + (F1/F3 - coordsY)*jHat
#
# Use this vector field to warp the mesh with the Warp by Vector filter.  The
# velocity components may then be plotted as scalars, or turned into a vector
# and/or otherwise processed using the Calculator filter.
