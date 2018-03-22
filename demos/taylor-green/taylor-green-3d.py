"""
Div-conforming B-spline discretization of 3D Taylor--Green flow, following 
Section 9.11.2 of

  https://repositories.lib.utexas.edu/bitstream/handle/2152/ETD-UT-2011-12-4506/EVANS-DISSERTATION.pdf

Geometry is described for this problem using an explicit B-spline.

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
NEL = 24

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

# Define the spaces for RT-type compatible splines on this geometry.
splineGenerator = BSplineCompat(controlMesh,"RT",degs)

# Apply strong BCs in parametric normal directions. 
for field in range(0,3):
    scalarSpline = splineGenerator.getFieldSpline(field)
    for side in range(0,2):
        sideDofs = scalarSpline.getSideDofs(field,side)
        splineGenerator.addZeroDofs(field,sideDofs)


# Write the extraction data to the filesystem.
DIR = "./extraction"
splineGenerator.writeExtraction("./extraction")


####### Analysis #######

if(mpirank==0):
    print("Setting up extracted spline...")

QUAD_DEG = 3
spline = ExtractedBSplineRT(splineGenerator,QUAD_DEG)
#spline = ExtractedBSplineRT("./extraction",QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")

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
spline.linearSolver.parameters["relative_tolerance"] = 1e-2
spline.relativeTolerance = 1e-3

# The unknown parametric velocity:
u_hat = Function(spline.V)

# Parametric velocity at the old time level:
if(mpirank==0):
    print("Projecting velocity IC...")
u_old_hat = spline.divFreeProject(soln)

# Parametric $\partial_t u$ at the old time level:
udot_old_hat = Function(spline.V)

# Create a generalized-alpha time integrator.
RHO_INF = 1.0
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,u_hat,\
                                     (u_old_hat, udot_old_hat))

# The alpha-level parametric velocity and its partial derivative w.r.t. time:
u_hat_alpha = timeInt.x_alpha()
udot_hat_alpha = timeInt.xdot_alpha()

# A helper function to take the symmetric gradient:
def eps(u):
    return 0.5*(spline.grad(u) + spline.grad(u).T)

# The physical velocity and its temporal partial derivative:
u = spline.pushforward(u_hat_alpha)
udot = spline.pushforward(udot_hat_alpha)

# The parametric and physical test functions:
v_hat = TestFunction(spline.V)
v = spline.pushforward(v_hat)

# The material time derivative of the velocity:
Du_Dt = udot + spline.grad(u)*u

# The viscous part of the Cauchy stress:
sigmaVisc = 2.0*VISC*eps(u)

# The problem is posed on solenoidal subspace, as enforced by the iterative
# penalty solver; no pressure terms are necessary in the weak form.
res = DENS*inner(Du_Dt,v)*spline.dx \
      + inner(sigmaVisc,eps(v))*spline.dx

# Auxilliary Function to re-use during the iterated penalty solves:
w = Function(spline.V)

# Time stepping loop:
for i in range(0,N_STEPS):

    if(mpirank == 0):
        print("------- Time step "+str(i+1)+" , t = "+str(timeInt.t)+" -------")

    # Solve for velocity in a solenoidal subspace of the RT-type
    # B-spline space.
    spline.iteratedDivFreeSolve(res,u_hat,v_hat,penalty=Constant(1e4),w=w)

    # Assemble the dissipation rate, and append it to a file that can be
    # straightforwardly plotted as a function of time using gnuplot.
    dissipationRate = assemble((2.0*VISC/DENS/pi**3)
                               *inner(eps(u),eps(u))*spline.dx)

    # Because the algebraic problem is solved only approximately, there is some
    # nonzero divergence to the velocity field.  If the tolerances are set
    # small enough, this can be driven down to machine precision.  
    divError = assemble(spline.div(u)**2*spline.dx)
    
    if(mpirank==0):
        print("Divergence error: "+str(divError))
        mode = "a"
        if(i==0):
            mode = "w"
        outFile = open("dissipationRate.dat",mode)
        outFile.write(str(timeInt.t)+" "+str(dissipationRate)+"\n")
        outFile.close()

    # Move to the next time step.
    timeInt.advance()
