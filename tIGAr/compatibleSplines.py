from tIGAr.common import *
from tIGAr.BSplines import *
import copy
from numpy import concatenate

DEFAULT_RT_PENALTY = Constant(1e1)

# may want to generate the corresponding scalar fields to append to
# larger lists, in, e.g., monolithic pressure--velocity formulations, or
# monolithic or partially-segregated MHD formulations

# due to similar code structure, combine RT and N generation; pass
# RTorN = "RT" or "N"
def generateFieldsCompat(controlMesh,RTorN,degrees,periodicities=None):
    nvar = len(degrees)
    fields = []
    for i in range(0,nvar):
        knotVectors = []
        scalarDegrees = []
        for j in range(0,nvar):
            degree = degrees[i]
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
        fields += [BSpline(scalarDegrees,knotVectors),]
    return fields


# spline with mixed space for just an RT velocity; assume that some sort of
# iterated penalty type solver is used for the pressure, so there is no
# pressure variable
class BSplineCompat(AbstractMultiFieldSpline):

    # args: controlMesh, RTorN, degrees, periodicities=None
    def customSetup(self,args):
        self.controlMesh = args[0]
        self.RTorN = args[1]
        self.degrees = args[2]
        if(len(args)>3):
            self.periodicities = args[3]
        else:
            self.periodicities = None
        self.fields = generateFieldsCompat(self.controlMesh,self.RTorN,\
                                           self.degrees,\
                                           self.periodicities)
    def getControlMesh(self):
        return self.controlMesh

    def getFieldSpline(self,field):
        return self.fields[field]

    def getNFields(self):
        return len(self.fields)

class ExtractedBSplineRT(ExtractedSpline):

    def pushforward(self,uhat,F=None):
        if(F==None):
            F = self.F
        return cartesianPushforwardRT(uhat,F)
    
    # use the iterated penalty method to get a solenoidal RT velocity field u
    # satisfying residualForm == 0.  v is the test function (couldn't find any
    # way to get it directly from the form...)
    def iteratedDivFreeSolve(self,residualForm,u,v,penalty=DEFAULT_RT_PENALTY):

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
        uhat = Function(self.V)
        vhat = TestFunction(self.V)
        u = self.pushforward(uhat)
        v = self.pushforward(vhat)
        res = inner(u-toProject,v)*self.dx
        self.iteratedDivFreeSolve(res,uhat,vhat,penalty)
        return uhat

class ExtractedBSplineN(ExtractedSpline):

    def pushforward(self,Ahat,F=None):
        if(F==None):
            F = self.F
        return cartesianPushforwardN(Ahat,F)

    # un-determined up to a gradient; need to use an appropriate iterative
    # solver to get an answer
    def projectCurl(self,toProject,applyBCs=False):
        Ahat = TrialFunction(self.V)
        Bhat = TestFunction(self.V)
        u = self.curl(self.pushforward(Ahat))
        v = self.curl(self.pushforward(Bhat))
        res = inner(u-toProject,v)*self.dx
        retval = Function(self.V)
        self.solveLinearVariationalProblem(res,retval,applyBCs)
        return retval
