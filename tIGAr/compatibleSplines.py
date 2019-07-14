"""
The ``compatibleSplines`` module
--------------------------------
contains functionality relating to div- and curl-conforming B-splines, as 
originally developed by Buffa and collaborators:

https://epubs.siam.org/doi/10.1137/100786708

These spline spaces may be used with B-spline or NURBS control meshes to
define the geometrical mapping from parametric to physical space.
"""

from tIGAr.common import *
from tIGAr.BSplines import *
import copy
from numpy import concatenate
import sys

DEFAULT_RT_PENALTY = Constant(1e1)

def generateFieldsCompat(controlMesh,RTorN,degrees,periodicities=None):
    """
    This function generates a list of ``BSpline`` scalar bases corresponding
    to the scalar components of either an RT-type or N-type compatible spline
    discretization.  (See the dissertation of J.A. Evans for notation.)  The
    argument ``controlMesh`` must refer to an instance of 
    ``AbstractControlMesh`` whose underlying scalar spline is a ``BSpline``.
    (This will provide the knot vectors to be used in constructing the 
    compatible spline spaces.)  The argument ``RTorN`` is a string containing
    either "RT" or "N", indicating the type of spline spaces to produce.
    The argument ``degrees`` is a list of integers, indicating the polynomial 
    degree in each parametric direction,
    in the sense of "k'" from J.A. Evans's notation.  The optional argument
    ``periodicities`` is a list of Booleans, and indicates whether or not 
    the discretization is periodic in each direction.  
    """
    
    nvar = len(degrees)
    useRect = controlMesh.getScalarSpline().useRectangularElements()
    fields = []
    # i indexes parametric components of the velocity (i.e., scalar fields)
    for i in range(0,nvar):
        knotVectors = []
        scalarDegrees = []
        # j indexes parametric directions for building the tensor product
        # space for field i
        for j in range(0,nvar):
            degree = degrees[j]
            # different between RT and N: k-refine along (RT) or
            # perpendicular to (N) the vector direction
            if(((RTorN=="RT") and (j==i)) or ((RTorN=="N") and (not j==i))):
                degree += 1
            # ASSUMES that the underlying scalar spline of the control mesh
            # is a BSpline, and re-uses unique knots
            knots = copy.copy(controlMesh.getScalarSpline().\
                              splines[j].uniqueKnots)
            # use open knot vector by default or if non-periodic
            if(periodicities==None or (not periodicities[j])):
                for k in range(0,degree):
                    knots = concatenate((array([knots[0],]),\
                                         knots, array([knots[-1],])))
            knotVectors += [knots,]
            scalarDegrees += [degree,]
        fields += [BSpline(scalarDegrees,knotVectors,useRect),]
    return fields

# can be RT or N type B-spline
class BSplineCompat(AbstractMultiFieldSpline):

    """
    Class for generating extraction data for a compatible spline of type
    RT or N when no other fields are required.
    """
    
    # args: controlMesh, RTorN, degrees, periodicities=None,
    def customSetup(self,args):
        """
        The first argument is an ``AbstractControlMesh``, the second is
        a string containing "RT" or "N", the third is a list of polynomial
        degrees in parametric directions, and the (optional) fourth is
        a list of Booleans indicating periodicity along the parametric 
        directions.
        """
        self.controlMesh = args[0]
        self.RTorN = args[1]
        self.degrees = args[2]
        if(len(args)>3):
            self.periodicities = args[3]
        else:
            self.periodicities = None
        self.fields = generateFieldsCompat(self.controlMesh,self.RTorN,
                                           self.degrees,
                                           periodicities=self.periodicities)
    def getControlMesh(self):
        return self.controlMesh

    def getFieldSpline(self,field):
        return self.fields[field]

    def getNFields(self):
        return len(self.fields)

def iteratedDivFreeSolve(residualForm,u,v,spline,divOp=None,
                         penalty=DEFAULT_RT_PENALTY,
                         w=None,J=None,reuseLHS=True,applyBCs=True):
    """
    Use the iterated penalty method to find a solution to the 
    problem given by ``residualForm``, while constraining the test and
    trial functions (``u`` and ``v``) to solenoidal subspaces of 
    ``spline.V``, where ``spline`` is an ``ExtractedSpline``. 
    The argument ``divOp`` optionally allows a custom 
    function to be passed as the divergence operator.  By default, it is
    the physical divergence ``spline.div``, but this may not make sense
    if fields other than the velocity components are part of the spline space.
    Making the ``penalty`` larger can speed up convergence,
    at the cost of worse linear algebra conditioning.  The optional
    parameter ``w`` allows for a nonzero initial guess for the 
    pressure (in the form of a velocity function in the RT-type space
    whose divergence serves as a pressure).  The idea is that, when time
    stepping, the final ``w`` from the previous step  will be an accurate 
    initial guess for the current step.  If nothing is passed ``w`` 
    will be initialized to zero.  The optional argument ``J`` is a custom
    Jacobian for ``residualForm``, defaulting to ``None``, in which case
    ``derivative(residualForm,u)`` is used.  For nonlinear problems, one
    can optionally set ``reuseLHS`` to ``False``, to re-assemble the tangent 
    matrix in each penalty iteration.  The optional Boolean parameter 
    ``applyBCs`` indicates whether or not to apply Dirichlet boundary 
    conditions.

    For details and analysis of the iterated penalty method, see

    https://epubs.siam.org/doi/10.1137/16M1103117

    NOTE: This algorithm was developed for linear problems, but we also
    apply it in an ad hoc manner to nonlinear problems, by essentially
    performing one multiplier update per Newton step.  This appears to
    be effective in practice.  
    """

    if(divOp==None):
        # It is more efficient on explicit splines to directly use the
        # parameteric div operator.  It is still correct to use parametric
        # div on deformed splines, because the Piola transform is
        # div-conserving, but the penalty may not control the physical
        # divergence evenly on nonuniform meshes.
        divOp = lambda u_hat : \
                spline.div(cartesianPushforwardRT(u_hat,spline.F))
    
    # augmented problem
    if(w==None):
        w = Function(spline.V)

    augmentation = penalty*divOp(u)*divOp(v)*spline.dx \
                   + divOp(w)*divOp(v)*spline.dx
    residualFormAug = residualForm + augmentation
    if(J==None):
        JAug = derivative(residualFormAug,u)
    else:
        JAug = J + derivative(augmentation,u)

    # TODO: Think more about implementing separate tolerances for
    # momentum and continuity residuals.
    converged = False
    for i in range(0,spline.maxIters):
        #MTAM,MTb = spline.assembleLinearSystem(JAug,residualFormAug)
        MTb = spline.assembleVector(residualFormAug,applyBCs=applyBCs)
        if(i==0 or (not reuseLHS)):
            MTAM = spline.assembleMatrix(JAug,applyBCs=applyBCs)

        currentNorm = norm(MTb)
        if(i==0):
            initialNorm = currentNorm
        relativeNorm = currentNorm/initialNorm
        if(mpirank == 0):
            print("Solver iteration: "+str(i)+" , Relative norm: "
                  + str(relativeNorm))
            sys.stdout.flush()
        if(currentNorm/initialNorm < spline.relativeTolerance):
            converged = True
            break
        du = Function(spline.V)
        #du.assign(Constant(0.0)*du)
        spline.solveLinearSystem(MTAM,MTb,du)
        #as_backend_type(u.vector()).vec().assemble()
        #as_backend_type(du.vector()).vec().assemble()
        u.assign(u-du)
        w.assign(w+penalty*u)
    if(not converged):
        print("ERROR: Iterated penalty solver failed to converge.")
        exit()

def divFreeProject(toProject,spline,
                   getVelocity=lambda x:x,
                   getOtherFields=None,
                   penalty=DEFAULT_RT_PENALTY,w=None,applyBCs=True):
    """
    Project some expression ``toProject`` onto a solenoidal subspace of
    ``spline.V``, using the iterated penalty method with an
    optionally-specified ``penalty``.  The optional parameter ``getVelocity`` 
    (defaulting to identity) maps ``Function`` objects in ``spline.V`` to
    vector fields that should be divergence-free in the parametric domain.  
    The parameter ``getOtherFields``, if non-``None``, maps ``Function`` 
    objects in ``spline.V`` to a quantity that will be set to zero in an
    L^2 sense.  (This is intended to select fields other than the vector field
    returned by ``getVelocity``; if the default identity map for 
    ``getVelocity`` is used, then ``None`` is a sensible
    default to use simultaneously for ``getOtherFields``.)
    The optional parameter ``w`` is a ``Function`` that can 
    contain an initial guess for (and/or provide the final value of) 
    the pressure associated with the projection.  The optional Boolean argument
    ``applyBCs`` indicates whether or not to apply Dirichlet boundary
    conditions.
    """
    u_hat = Function(spline.V)
    v_hat = TestFunction(spline.V)
    u = cartesianPushforwardRT(getVelocity(u_hat),spline.F)
    v = cartesianPushforwardRT(getVelocity(v_hat),spline.F)
    res = inner(u-toProject,v)*spline.dx
    if(getOtherFields != None):
        p = getOtherFields(u_hat)
        q = getOtherFields(v_hat)
        res += inner(p,q)*spline.dx
    iteratedDivFreeSolve(res,u_hat,v_hat,spline,
                         divOp=lambda up : div(getVelocity(up)),
                         penalty=penalty,w=w,applyBCs=applyBCs)
    return u_hat

# TODO: Deprecate this class, update demos
class ExtractedBSplineRT(ExtractedSpline):
    """
    Subclass of ``ExtractedSpline`` offering some specializations to the
    case of an RT spline being used for solving problems on 
    solenoidal subspaces.
    """
    
    def pushforward(self,uhat,F=None):
        """
        An RT-type pushforward of ``uhat``, assuming a mapping of ``spline.F``
        or, optionally, some other ``F`` passed as an argument.
        """
        if(F==None):
            F = self.F
        return cartesianPushforwardRT(uhat,F)

    def iteratedDivFreeSolve(self,residualForm,u,v,
                             penalty=DEFAULT_RT_PENALTY,
                             w=None,applyBCs=True):
        """
        Wrapper for free function ``iteratedDivFreeSolve``, largely included
        for backward compatibility.
        """
        iteratedDivFreeSolve(residualForm,u,v,self,penalty=penalty,w=w,
                             applyBCs=applyBCs)

    def divFreeProject(self,toProject,penalty=DEFAULT_RT_PENALTY,w=None,
                       applyBCs=True):
        """
        Wrapper for free function ``divFreeProject``, largely included
        for backward compatibility.
        """
        return divFreeProject(toProject,self,penalty=penalty,w=w,
                              applyBCs=applyBCs)

# TODO: Deprecate this class, update demos
class ExtractedBSplineN(ExtractedSpline):
    """
    Subclass of ``ExtractedSpline`` offering some specializations to the
    case of an N-type spline being used as a vector potential.
    """

    def pushforward(self,Ahat,F=None):
        """
        An N-type pushforward of vector potential ``Ahat``, using mapping
        ``spline.F`` or, optionally, some other mapping ``F``.
        """
        if(F==None):
            F = self.F
        return cartesianPushforwardN(Ahat,F)

    # TODO: Move to free function
    def projectCurl(self,toProject,applyBCs=False):
        """
        Project ``toProject`` onto the curl of a vector potential in 
        ``spline.V``.  

        NOTE:  This is technically un-determined up to a gradient, but
        some iterative solvers can still usually pick a solution.
        """
        Ahat = TrialFunction(self.V)
        Bhat = TestFunction(self.V)
        u = self.curl(self.pushforward(Ahat))
        v = self.curl(self.pushforward(Bhat))
        res = inner(u-toProject,v)*self.dx
        retval = Function(self.V)
        self.solveLinearVariationalProblem(res,retval,applyBCs)
        return retval
