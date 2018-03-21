"""
Div-conforming B-spline discretization of 3D Taylor--Green flow, following 
Section 9.11.2 of

  https://repositories.lib.utexas.edu/bitstream/handle/2152/ETD-UT-2011-12-4506/EVANS-DISSERTATION.pdf

discretizing the vector potential, instead of the velocity, i.e., we discretize
the field

  \mathbf{A} = \nabla\times\mathbf{u}

in an N-type space, instead of $\mathbf{u}$ in an RT-type space.  UFL makes 
this especially straightforward.  The viscous terms are therefore
fourth-order, which is why this formulation has not been widely-used
with traditional finite elements.  (A related formulation is the 
streamfunction--vorticity formulation sometimes applied in 2D, where only one
component of the vector potential (i.e., the scalar streamfunction) is 
nonzero, and viscous terms are handled by solving for an extra vorticity 
field.)

Geometry is described for this problem using an explicit B-spline.

Note: Due to the algebraic complexity added by the curl operations to
convert vector potentials to velocities in this formulation, the form 
compilation in this example takes *much* longer than it does for typical 
problems. 

Note: Due to the use of explicit B-splines, more efficient assembly is possible
by neglecting pushforwards and geometrical mappings.  However, the purpose
of this demo is primarily didactic, so the unnecessary algebra is included.
"""

from tIGAr import *
from tIGAr.compatibleSplines import *
from tIGAr.timeIntegration import *
import math


####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Number of elements in each direction:
NEL = 16

# Polynomial degree (in the sense of $k'$ from the cited reference) in each
# parametric direction.  (Note that, in this case, degree=1 still implies some
# $C^1$ quadratic B-splines for the div-conforming unknown fields.)
degs = [1,1,1]

# Knot vectors for defining the control mesh.
kvecs = [uniformKnots(degs[0],0.0,math.pi,NEL,False),\
         uniformKnots(degs[1],0.0,math.pi,NEL,False),\
         uniformKnots(degs[2],0.0,math.pi,NEL,False)]

# Define a trivial mapping from parametric to physical space, via explicit
# B-spline.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs)

# Define the spaces for N-type compatible splines on this geometry.
splineGenerator = BSplineCompat(controlMesh,"N",degs)

# Apply strong BCs to velocity in parametric normal directions by constraining
# the vector potential in tangential directions. 
for field in range(0,3):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for direction in range(0,3):
        for side in range(0,2):
            if(field != direction):
                sideDofs = scalarSpline.getSideDofs(direction,side)
                splineGenerator.addZeroDofs(field,sideDofs)


# Write the extraction data to the filesystem.
DIR = "./extraction"
splineGenerator.writeExtraction("./extraction")


####### Analysis #######

if(mpirank==0):
    print("Setting up extracted spline...")

QUAD_DEG = 2
spline = ExtractedBSplineN(splineGenerator,QUAD_DEG)
#spline = ExtractedBSplineN("./extraction",QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")
    print("** NOTE: Form compilation in this example is unusually-slow. **")
    print("Please be patient...")

# Parameters of the time discretization:
TIME_INTERVAL = 16.0
N_STEPS = 8*NEL
DELTA_T = TIME_INTERVAL/float(N_STEPS)

# Mass density:
DENS = Constant(1.0)

# Define the dynamic viscosity based on the desired Reynolds number.
Re = Constant(100.0)
VISC = DENS/Re

# The initial condition for the flow:
x = spline.spatialCoordinates()
soln0 = sin(x[0])*cos(x[1])*cos(x[2])
soln1 = -cos(x[0])*sin(x[1])*cos(x[2])
soln = as_vector([soln0,soln1,Constant(0.0)])

# For 3D computations, use an iterative solver.
spline.linearSolver = PETScKrylovSolver("gmres","jacobi")

# In the discretization using an RT-type velocity field, the solver tolerances
# can lead to nonzero velocity divergence.  In the vector potential
# discretization, the velocity solution is solenoidal by construction.  
spline.linearSolver.parameters["relative_tolerance"] = 1e-2
spline.relativeTolerance = 1e-3

# The unknown parametric vector potential:
A_hat = Function(spline.V)

# Parametric velocity at the old time level:
if(mpirank==0):
    print("Projecting velocity IC...")
A_old_hat = spline.projectCurl(soln)

# Parametric $\partial_t A$ at the old time level:
Adot_old_hat = Function(spline.V)

# Create a generalized-alpha time integrator.
RHO_INF = 1.0
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,A_hat,\
                                     (A_old_hat, Adot_old_hat))

# The alpha-level parametric velocity and its partial derivative w.r.t. time:
A_hat_alpha = timeInt.x_alpha()
Adot_hat_alpha = timeInt.xdot_alpha()

# A helper function to take the symmetric gradient:
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# The physical velocity and its temporal partial derivative:
u = spline.curl(spline.pushforward(A_hat_alpha))
udot = spline.curl(spline.pushforward(Adot_hat_alpha))

# The parametric and physical test functions:
B_hat = TestFunction(spline.V)
v = spline.curl(spline.pushforward(B_hat))

# Notice that, using UFL, the residual for the Navier--Stokes problem
# is programmed in the "normal" way, in terms of velocity, without reference
# to the vector potential.

# The material time derivative of the velocity:
Du_Dt = udot + spline.grad(u)*u

# The viscous part of the Cauchy stress:
sigmaVisc = 2.0*VISC*eps(u)

# The problem is posed on solenoidal subspace, as enforced by the fact that
# u and v are curls of vector fields, so there is no need to include a
# pressure term.
res = DENS*inner(Du_Dt,v)*spline.dx \
      + inner(sigmaVisc,eps(v))*spline.dx

# Jacobian for Newton's method:
J = derivative(res,A_hat)

# Time stepping loop:
for i in range(0,N_STEPS):

    if(mpirank == 0):
        print("------- Time step "+str(i+1)+" , t = "+str(timeInt.t)+" -------")

    # Solve for the vector potential.
    spline.solveNonlinearVariationalProblem(res,J,A_hat)

    # Assemble the dissipation rate, and append it to a file that can be
    # straightforwardly plotted as a function of time using gnuplot.
    dissipationRate = assemble((2.0*VISC/DENS/pi**3)
                               *inner(eps(u),eps(u))*spline.dx)
    if(mpirank==0):
        mode = "a"
        if(i==0):
            mode = "w"
        outFile = open("dissipationRate.dat",mode)
        outFile.write(str(timeInt.t)+" "+str(dissipationRate)+"\n")
        outFile.close()

    # Move to the next time step.
    timeInt.advance()
