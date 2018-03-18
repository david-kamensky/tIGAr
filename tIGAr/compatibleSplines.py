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
        self.fields = generateFieldsCompat(self.controlMesh,self.RTorN,\
                                           self.degrees,\
                                           periodicities=self.periodicities)
    def getControlMesh(self):
        return self.controlMesh

    def getFieldSpline(self,field):
        return self.fields[field]

    def getNFields(self):
        return len(self.fields)

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

    # TODO: Think of a nice way to control whether the LHS form is
    # re-assembled each iteration.
    def iteratedDivFreeSolve(self,residualForm,u,v,penalty=DEFAULT_RT_PENALTY):
        """
        Use the iterated penalty method to find a solution to the 
        problem given by ``residualForm``, while constraining the test and
        trial functions (``u`` and ``v``) to solenoidal subspaces of 
        ``spline.V``.  Making the ``penalty`` larger can speed up convergence,
        at the cost of worse linear algebra conditioning.  For details and
        analysis, see

        https://epubs.siam.org/doi/10.1137/16M1103117

        NOTE: This algorithm was developed for linear problems, but we also
        apply it in an ad hoc manner to nonlinear problems, by essentially
        performing one multiplier update per Newton step.  This appears to
        be effective in practice.  
        """
        
        # augmented problem
        w = Function(self.V)

        # just penalize directly in parametric domain, because... why not?
        augmentation = penalty*div(u)*div(v)*self.dx \
                       + div(w)*div(v)*self.dx
        residualFormAug = residualForm + augmentation
        JAug = derivative(residualFormAug,u)

        converged = False
        for i in range(0,self.maxIters):
            MTAM,MTb = self.assembleLinearSystem(JAug,residualFormAug)

            currentNorm = norm(MTb)
            if(i==0):
                initialNorm = currentNorm
            relativeNorm = currentNorm/initialNorm
            if(mpirank == 0):
                print("Solver iteration: "+str(i)+" , Relative norm: "\
                      + str(relativeNorm))
            if(currentNorm/initialNorm < self.relativeTolerance):
                converged = True
                break
            du = Function(self.V)
            #du.assign(Constant(0.0)*du)
            self.solveLinearSystem(MTAM,MTb,du)
            #as_backend_type(u.vector()).vec().assemble()
            #as_backend_type(du.vector()).vec().assemble()
            u.assign(u-du)
            w.assign(w+penalty*u)
        if(not converged):
            print("ERROR: Iterated penalty solver failed to converge.")
            exit()

    def divFreeProject(self,toProject,penalty=DEFAULT_RT_PENALTY):
        """
        Project some expression ``toProject`` onto a solenoidal subspace of
        ``spline.V``, using the iterated penalty method with an
        optionally-specified ``penalty``.
        """
        uhat = Function(self.V)
        vhat = TestFunction(self.V)
        u = self.pushforward(uhat)
        v = self.pushforward(vhat)
        res = inner(u-toProject,v)*self.dx
        self.iteratedDivFreeSolve(res,uhat,vhat,penalty)
        return uhat

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
