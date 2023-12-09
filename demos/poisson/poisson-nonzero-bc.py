"""
The linear algebra operations in tIGAr only directly support homogeneous (zero)
Dirichlet boundary conditions (BCs), but this is sufficient to implement
inhomogeneous (nonzero) BCs relatively easily. This demo modifies the basic
Poisson equation example to enforce inhomogeneous Dirichlet BCs.
"""

from tIGAr import *
from tIGAr.BSplines import *
import math

# Number of levels of refinement with which to run the Poisson problem:
N_LEVELS = 3

# Array to store error at different refinement levels:
L2_errors = zeros(N_LEVELS)

for level in range(0,N_LEVELS):

    ####### Preprocessing #######

    # Parameters determining the polynomial degree and number of elements in
    # each parametric direction.  By changing these and recording the error,
    # it is easy to see that the discrete solutions converge at optimal rates 
    # under refinement.
    p = 2
    q = 2
    NELu = 10*(2**level)
    NELv = 10*(2**level)

    # Parameters determining the position and size of the domain.
    x0 = 0.0
    y0 = 0.0
    Lx = 1.0
    Ly = 1.0

    if(mpirank==0):
        print("Generating extraction...")

    # Create a control mesh for which $\Omega = \widehat{\Omega}$.
    splineMesh = ExplicitBSplineControlMesh([p,q],\
                                            [uniformKnots(p,x0,x0+Lx,NELu),\
                                             uniformKnots(q,y0,y0+Ly,NELv)])

    # Create a spline generator for a spline with a single scalar field on the
    # given control mesh, where the scalar field is the same as the one used
    # to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
    splineGenerator = EqualOrderSpline(1,splineMesh)

    # Set Dirichlet boundary conditions on the 0-th (and only) field, on both
    # ends of the domain, in both directions.
    field = 0
    scalarSpline = splineGenerator.getScalarSpline(field)
    for parametricDirection in [0,1]:
        for side in [0,1]:
            sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
            splineGenerator.addZeroDofs(field,sideDofs)

    ####### Analysis #######

    if(mpirank==0):
        print("Setting up extracted spline...")

    # Choose the quadrature degree to be used throughout the analysis.
    QUAD_DEG = 2*max(p,q)

    # Create the extracted spline directly from the generator.
     spline = ExtractedSpline(splineGenerator,QUAD_DEG)

    if(mpirank==0):
        print("Solving...")

    # Create a force, f, to manufacture the exact solution; notice that
    # the exact solution is nonzero on the boundary.
    x = spline.spatialCoordinates()
    soln = cos(pi*(x[0]-x0)/Lx)*cos(pi*(x[1]-y0)/Ly)
    f = -spline.div(spline.grad(soln))

    # The simplest way to apply nonzero boundary conditions is by using
    # the nonlinear solver. It applies a homogeneous boundary condition
    # to each Newton increment, therefore preserving any inhomogenous
    # boundary data applied to the initial guess.

    # First, we project the exact solution to the Function that will serve
    # as our solution. We could project any function that satisfies the
    # BC, but we use a projection of the exact solution here because it is
    # available here. By default, the projection is done in an $L^2$ sense,
    # which requires solving a linear system. The projection method has an
    # optional Boolean keyword argument to use a lumped-mass projection, which
    # is much faster and accurate enough (2nd-order) for most applications.
    lumpMass = False
    u = spline.project(soln, rationalize=False, lumpMass=lumpMass)

    # Note that we want to use the spline's projection method rather than
    # direclty setting nodal values in the finite element respresentation of
    # u, because we want an initial guess that is in the spline space.
    
    # Next, we set up and solve the Poisson equation as if it were a
    # nonlinear problem. This will converge in a single iteration, due to
    # the linearity of the problem. The zero BC set earlier will be enforced
    # by default on each increment of the nonlinear solve.
    v = TestFunction(spline.V)
    residual = (inner(spline.grad(u),spline.grad(v)) - inner(f,v))*spline.dx
    jacobian = derivative(residual, u)
    spline.solveNonlinearVariationalProblem(residual, jacobian, u)

    # A mathematically-equivalent trick for linear problems of the form
    # $a(u,v) = L(v)~\forall v$ is to solve the linear problem
    #
    #   a(w,v) = L(v) - a(g,v)    \forall v
    #
    # for $w$ with homogeneous BCs, where $g$ satisfies the boundary condition
    # and the desired solution with inhomogenous BCs is given by $u = w + g$.
    # It is easy to see that this is just one step of Newton iteration staring
    # from the guess $g$.


    ####### Postprocessing #######

    # The solution, u, is in the homogeneous representation, but, again, for
    # B-splines with weight=1, this is the same as the physical representation.
    File("results/u.pvd") << u

    # Compute and print the $L^2$ error in the discrete solution.
    L2_error = math.sqrt(assemble(((u-soln)**2)*spline.dx))
    L2_errors[level] = L2_error
    if(level > 0):
        rate = math.log(L2_errors[level-1]/L2_errors[level])/math.log(2.0)
    else:
        rate = "--"
    if(mpirank==0):
        print("L2 Error for level "+str(level)+" = "+str(L2_error)
              +"  (rate = "+str(rate)+")")

