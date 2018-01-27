# some utility functions for differential geometry and whatnot in UFL

# note that these functions simply help prepare a UFL specification of the
# PDE, which is then compiled into efficient code.  These are not being called
# inside some inner loop over quadrature points, and should therefore be
# optimized for generality/readability rather than speed of execution.

from dolfin import *

# get a metric tensor from a given deformation
def getMetric(F):
    DF = grad(F)
    return DF.T*DF

# get christoffel symbols for a metric g assuming the first index is the
# raised one
# TODO: check/test
def getChristoffel(g):
    a,b,c,d = indices(4)
    return as_tensor\
        (0.5*inv(g)[a,b]\
         *(grad(g)[c,b,d]\
           + grad(g)[d,b,c]\
           - grad(g)[d,c,b]), (a,d,c))

# map a normal vector N forward to n, after deformation F
def mappedNormal(N,F,normalize=True):
    DF = grad(F)
    g = getMetric(F)
    n = DF*inv(g)*N
    if(normalize):
        return n/sqrt(inner(n,n))
    else:
        return n
    # sanity check: consistent w/ Nanson formula for invertible DF,
    # metric = (DF)^T * DF

# pseudo-inverse of DF for change-of-variables
def pinvD(F):
    DF = grad(F)
    g = getMetric(F)
    return inv(g)*DF.T
    
# volume jacobian
def volumeJacobian(g):
    return sqrt(det(g))
    
# surface jacobian
def surfaceJacobian(g,N):
    return sqrt(det(g)*inner(N,inv(g)*N))

# class for tensors in curvilinear coordinates w/ metric g
class CurvilinearTensor:
    def __init__(self,T,g,lowered=None):
        self.T = T
        self.g = g
        if(lowered != None):
            self.lowered = lowered
        else:
            # default: all lowered indices
            self.lowered = []
            for i in range(0,rank(T)):
                self.lowered += [True,]
                
    def __add__(self,other):
        # TODO: add consistency checks on g and lowered
        return CurvilinearTensor(self.T+other.T,self.g,self.lowered)

    def __sub__(self,other):
        return CurvilinearTensor(self.T-other.T,self.g,self.lowered)

    # for scalar coefficients
    def __rmul__(self,other):
        return CurvilinearTensor(other*self.T,self.g,self.lowered)
                
    # not well tested...
    def raiseLowerIndex(self,i):
        n = rank(self.T)
        ii = indices(n+1)
        mat = self.g
        if(self.lowered[i]):
            mat = inv(self.g)
        else:
            mat = self.g
        retval = as_tensor(self.T[ii[0:i]+(ii[i],)+ii[i+1:n]]\
                           *mat[ii[i],ii[n]],\
                           ii[0:i]+(ii[n],)+ii[i+1:n])
        return CurvilinearTensor(retval,self.g,\
                                 self.lowered[0:i]\
                                 +[not self.lowered[i],]+self.lowered[i+1:])
    def raiseIndex(self,i):
        if(self.lowered[i]):
            return self.raiseLowerIndex(i)
        else:
            return self
        
    def lowerIndex(self,i):
        if(not self.lowered[i]):
            return self.raiseLowerIndex(i)
        else:
            return self

    def sharp(self):
        retval = self
        for i in range(0,rank(self.T)):
            retval = retval.raiseIndex(i)
        return retval

    def flat(self):
        retval = self
        for i in range(0,rank(self.T)):
            retval = retval.lowerIndex(i)
        return retval

    def rank(self):
        return rank(self.T)

def curvilinearInner(T,S):
    Tsharp = T.sharp();
    Sflat = S.flat();
    ii = indices(rank(T.T))
    return as_tensor(Tsharp.T[ii]*Sflat.T[ii],())

# Covariant derivative of curvilinear tensor
# TODO: check/test more thoroughly
def covariantDerivative(T):
    n = rank(T.T)
    ii = indices(n+2)
    g = T.g
    gamma = getChristoffel(g)
    retval = grad(T.T)
    for i in range(0,n):
        # use ii[n] as the new index of the covariant deriv
        # use ii[n+1] as dummy index
        if(T.lowered[i]):
            retval -= as_tensor(T.T[ii[0:i]+(ii[n+1],)+ii[i+1:n]]\
                                *gamma[(ii[n+1],ii[i],ii[n])],\
                                ii[0:n+1])
        else:
            retval += as_tensor(T.T[ii[0:i]+(ii[n+1],)+ii[i+1:n]]\
                                *gamma[(ii[i],ii[n+1],ii[n])],\
                                ii[0:n+1])
    newLowered = T.lowered+[True,]
    return CurvilinearTensor(retval,g,newLowered)

# gradient of curvilinear tensor
def curvilinearGrad(T):
    n = rank(T.T)
    ii = indices(n+2)
    g = T.g
    deriv = covariantDerivative(T)
    invg = inv(g)
    # raise last index
    retval = as_tensor(deriv.T[ii[0:n+1]]*invg[ii[n:n+2]],\
                       ii[0:n]+(ii[n+1],))
    return CurvilinearTensor(retval,g,T.lowered+[False,])

# divergence of curvilinear tensor; contracts new lowered index from
# derivative with last raised index of tensor.  error if no raised indices.
def curvilinearDiv(T):
    n = rank(T.T)
    ii = indices(n)
    g = T.g
    j = -1 # last raised index
    for i in range(0,n):
        if(not T.lowered[i]):
            j = i
    if(j == -1):
        print("ERROR: Divergence operator requires at least one raised index.")
        exit()
    deriv = covariantDerivative(T)
    retval = as_tensor(deriv.T[ii[0:n]+(ii[j],)],ii[0:j]+ii[j+1:n])
    return CurvilinearTensor(retval,g,T.lowered[0:j]+T.lowered[j+1:n])

# Cartesian differential operators in deformed configuration
# N.b. that, when applied to tensor-valued f, f is considered to be
# in the Cartesian coordinates of the physical configuration, NOT in the
# local coordinate chart w.r.t. which derivatives are taken by FEniCS
def cartesianGrad(f,F):
    n = rank(f)
    ii = indices(n+2)
    pinvDF = pinvD(F)
    return as_tensor(grad(f)[ii[0:n+1]]\
                     *pinvDF[ii[n],ii[n+1]],\
                     ii[0:n]+(ii[n+1],))
def cartesianDiv(f,F):
    n = rank(f)
    ii = indices(n)
    return as_tensor(cartesianGrad(f,F)[ii+(ii[n-1],)],ii[0:n-1])

# only applies to f w/ rank 1, in 3D
def cartesianCurl(f,F):
    eps = PermutationSymbol(3)
    gradf = cartesianGrad(f,F)
    (i,j,k) = indices(3)
    return as_tensor(eps[i,j,k]*gradf[k,j],(i,))

# pushforwards for compatible spaces; output is in cartesian coordinates for
# physical space

# curl-conserving
def cartesianPushforwardN(u,F):
    DF = grad(F)
    return inv(DF.T)*u

# div-conserving
def cartesianPushforwardRT(v,F):
    DF = grad(F)
    return DF*v/det(DF)

# mass-conserving
def cartesianPushforwardW(phi,F):
    DF = grad(F)
    return phi/det(DF)

# TODO: rename this to ScaledMeasure
# I can't just scale a measure by a Jacobian, so I'll store them separately,
# then overload __rmul__()
class tIGArMeasure:

    # if quadDeg==None, then this works if meas is a FEniCS measure, OR if
    # meas is another tIGAr measure; it's a good idea to set quadDeg
    # if meas is a FEniCS measure, though, since the convoluted expressions
    # for rational splines tend to drive up the automatically-determined
    # quadrature degree
    def __init__(self,J,meas,quadDeg=None,boundaryMarkers=None):
        if(quadDeg != None):
            # TODO: is this reflected in the calling scope?
            meas = meas(metadata={'quadrature_degree': quadDeg})
        if(boundaryMarkers != None):
            meas = meas(subdomain_data=boundaryMarkers)
        self.meas = meas
        self.J = J

    # pass an argument indicating a subdomain marker
    def __call__(self,marker):
        return tIGArMeasure(self.J,self.meas(marker))
        
    def __rmul__(self, other):
        return (other*self.J)*self.meas
