"""
The ``calculusUtils`` module
----------------------------
contains functions and classes to hide the raw UFL involved in referring PDEs
back to the IGA parametric domain.  Note that most functionality in this 
module can also be used in non-tIGAr FEniCS applications.  
"""

# Note that these functions simply help prepare a UFL specification of the
# PDE, which is then compiled into efficient code.  These are not being called
# inside some inner loop over quadrature points, and should therefore be
# optimized for generality/readability rather than speed of execution.

from dolfin import *
from ufl import indices, rank, shape
from ufl.classes import PermutationSymbol

def getMetric(F):
    """
    Returns a metric tensor corresponding to a given mapping ``F`` from 
    parametric to physical space.
    """
    DF = grad(F)
    return DF.T*DF

def getChristoffel(g):
    """
    Returns Christoffel symbols associated with a metric tensor ``g``.  Indices
    are ordered based on the assumption that the first index is the raised one.
    """
    a,b,c,d = indices(4)
    return as_tensor\
        (0.5*inv(g)[a,b]\
         *(grad(g)[c,b,d]\
           + grad(g)[d,b,c]\
           - grad(g)[d,c,b]), (a,d,c))

def mappedNormal(N,F,normalize=True):
    """
    Returns a deformed normal vector corresponding to the area element with
    normal ``N`` in the parametric reference domain.  Deformation given by
    ``F``.  Optionally, the normal can be left un-normalized by setting
    ``normalize = False``.  In that case, the magnitude is the ratio of 
    deformed to reference area elements.
    """
    DF = grad(F)
    g = getMetric(F)
    n = DF*inv(g)*N
    if(normalize):
        return n/sqrt(inner(n,n))
    else:
        return n
    # sanity check: consistent w/ Nanson formula for invertible DF,
    # metric = (DF)^T * DF

def pinvD(F):
    """
    Returns the Moore--Penrose pseudo-inverse of the derivative of the mapping
    ``F``.
    """
    DF = grad(F)
    g = getMetric(F)
    return inv(g)*DF.T
    
def volumeJacobian(g):
    """
    Returns the volume element associated with the metric ``g``.
    """
    return sqrt(det(g))
    
def surfaceJacobian(g,N):
    """
    Returns the surface element associated with the metric ``g``, for a surface
    oriented in the direction given by unit vector ``N``.

    Note:  In version 2018.1, nontrivial boundary integrals produce 
    incorrect results with quad/hex elements.  Use of this in boundary 
    integrals is only robust with simplices.  
    """
    return sqrt(det(g)*inner(N,inv(g)*N))

# class for tensors in curvilinear coordinates w/ metric g
class CurvilinearTensor:
    """
    Class to represent arbitrary tensors in curvilinear coordinates, with a
    mechanism to distinguish between raised and lowered indices.
    """
    def __init__(self,T,g,lowered=None):
        """
        Create a ``CurvilinearTensor`` with components given by the UFL tensor
        ``T``, on a manifold with metric ``g``.  The sequence of Booleans
        ``lowered`` indicates whether or not each index is lowered.  The 
        default is for all indices to be lowered.
        """
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
                
    # mainly for internal use. not well tested...
    def raiseLowerIndex(self,i):
        """
        Flips the raised/lowered status of the ``i``-th index.
        """
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
        """
        Returns an associated tensor with the ``i``-th index raised.
        """
        if(self.lowered[i]):
            return self.raiseLowerIndex(i)
        else:
            return self
        
    def lowerIndex(self,i):
        """
        Returns an associated tensor with the ``i``-th index lowered.
        """
        if(not self.lowered[i]):
            return self.raiseLowerIndex(i)
        else:
            return self

    def sharp(self):
        """
        Returns an associated tensor with all indices raised.
        """
        retval = self
        for i in range(0,rank(self.T)):
            retval = retval.raiseIndex(i)
        return retval

    def flat(self):
        """
        Returns an associated tensor with all indices lowered.
        """
        retval = self
        for i in range(0,rank(self.T)):
            retval = retval.lowerIndex(i)
        return retval

    def rank(self):
        """
        Returns the rank of the tensor.
        """
        return rank(self.T)

def curvilinearInner(T,S):
    """
    Returns the inner product of ``CurvilinearTensor`` objects
    ``T`` and ``S``, inserting factors of the metric and inverse metric
    as needed, depending on the co/contra-variant status of corresponding
    indices.
    """
    Tsharp = T.sharp();
    Sflat = S.flat();
    ii = indices(rank(T.T))
    return as_tensor(Tsharp.T[ii]*Sflat.T[ii],())

# TODO: check/test more thoroughly
def covariantDerivative(T):
    """
    Returns a ``CurvilinearTensor`` that is the covariant derivative of
    the ``CurvilinearTensor`` argument ``T``.
    """
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

def curvilinearGrad(T):
    """
    Returns the gradient of ``CurvilinearTensor`` argument ``T``, i.e., the
    covariant derivative with the last index raised.
    """
    n = rank(T.T)
    ii = indices(n+2)
    g = T.g
    deriv = covariantDerivative(T)
    invg = inv(g)
    # raise last index
    retval = as_tensor(deriv.T[ii[0:n+1]]*invg[ii[n:n+2]],\
                       ii[0:n]+(ii[n+1],))
    return CurvilinearTensor(retval,g,T.lowered+[False,])

def curvilinearDiv(T):
    """
    Returns the divergence of the ``CurvilinearTensor`` argument ``T``, i.e.,
    the covariant derivative, but contracting over the new index and the 
    last raised index.  

    NOTE: This operation is invalid for tensors that do not 
    contain at least one raised index.
    """
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
    """
    The gradient of an arbitrary-rank tensor ``f`` in spatial Cartesian
    coordinates, assuming the parametric domain has been mapped to its
    physical configuration by the mapping ``F``.
    """
    return dot(grad(f),pinvD(F))
    #n = rank(f)
    #ii = indices(n+2)
    #pinvDF = pinvD(F)
    #return as_tensor(grad(f)[ii[0:n+1]]\
    #                 *pinvDF[ii[n],ii[n+1]],\
    #                 ii[0:n]+(ii[n+1],))

def cartesianDiv(f,F):
    """
    The divergence operator corresponding to ``cartesianGrad(f,F)`` that 
    sums on the last two indices.
    """
    n = rank(f)
    ii = indices(n)
    return as_tensor(cartesianGrad(f,F)[ii+(ii[n-1],)],ii[0:n-1])

def cartesianCurl(f,F):
    """
    The curl operator corresponding to ``cartesianGrad(f,F)``.  For ``f`` of 
    rank 1, it returns a vector in 3D or a scalar in 2D.  For ``f`` scalar
    in 2D, it returns a vector.
    """
    n = rank(f)
    gradf = cartesianGrad(f,F)
    if(n==1):
        m = shape(f)[0]
        eps = PermutationSymbol(m)
        if(m == 3):
            (i,j,k) = indices(3)
            return as_tensor(eps[i,j,k]*gradf[k,j],(i,))
        elif(m == 2):
            (j,k) = indices(2)
            return eps[j,k]*gradf[k,j]
        else:
            print("ERROR: Unsupported dimension of argument to curl.")
            exit()
    elif(n==0):
        return as_vector((-gradf[1],gradf[0]))
    else:
        print("ERROR: Unsupported rank of argument to curl.")
        exit()

# pushforwards for compatible spaces; output is in cartesian coordinates for
# physical space

def cartesianPushforwardN(u,F):
    """
    The curl-conserving pushforward of ``u`` by mapping ``F`` (as might be
    used for a Nedelec element, hence "N").  This is only valid for 3D vector
    fields on 3D domains.  
    """
    DF = grad(F)
    return inv(DF.T)*u

    # Since it only really makes sense to use this pushforward in 3D,
    # the 2D-safe pseudo-inverse implementation is overkill.
    #return (pinvD(F).T)*u

def cartesianPushforwardRT(v,F):
    """
    The div-conserving pushforward of ``v`` by mapping ``F`` (as might be
    used for a Raviart--Thomas element, hence "RT").
    """
    DF = grad(F)
    #return DF*v/det(DF)
    
    # Note: the metric is used for the Jacobian here to make sure this
    # formula remains well-defined on manifolds of nonzero co-dimension.
    # In that case det(DF) results in an error.
    
    # TODO: Include a switch somewhere to use the simpler formlua in the
    # co-dimension zero case, to avoid the unnecessary extra operations.

    g = getMetric(F)
    return DF*v/sqrt(det(g))
    
def cartesianPushforwardW(phi,F):
    """
    The mass-conserving pushforward of scalar field ``phi`` by mapping 
    ``F``.  ("W" comes from notation in J.A. Evans's dissertation.)
    """
    #DF = grad(F)
    #return phi/det(DF)
    g = getMetric(F)
    return phi/sqrt(det(g))
    
# TODO: rename this to ScaledMeasure
# I can't just scale a measure by a Jacobian, so I'll store them separately,
# then overload __rmul__()
class tIGArMeasure:

    """
    A UFL object multiplied by a measure produces a ``Form``, which cannot
    then be used conveniently, like a weighted measure.  This class is a 
    way to circumvent that, by storing the weight and measure separately,
    and combining them only once right-multiplied by something else.

    NOTE: Attempting to use subdomain data with the class currently
    behaves erratically.
    """
    
    # if quadDeg==None, then this works if meas is a FEniCS measure, OR if
    # meas is another tIGAr measure; it's a good idea to set quadDeg
    # if meas is a FEniCS measure, though, since the convoluted expressions
    # for rational splines tend to drive up the automatically-determined
    # quadrature degree
    def __init__(self,J,meas,quadDeg=None,boundaryMarkers=None):
        """
        Initializes a weighted measure that will behave like ``J*meas``.
        The optional argument ``quadDeg`` sets the quadrature degree of 
        ``meas``, which is a good idea for integrating the sorts of 
        complicated expressions that come out of rational spline 
        discretizations, because the automatically-computed quadrature rules
        have too many points.  The argument ``boundaryMarkers`` can be used
        to set ``subdomain_data`` of ``meas``, to perform integrals over
        specific sub-domains.
        """
        if(quadDeg != None):
            # TODO: is this reflected in the calling scope?
            meas = meas(metadata={'quadrature_degree': quadDeg})
        if(boundaryMarkers != None):
            meas = meas(subdomain_data=boundaryMarkers)
        self.meas = meas
        self.J = J

    def setMarkers(self,markers):
        """
        Sets the ``subdomain_data`` attribute of ``self.meas`` to 
        ``markers``.  
        """
        self.meas = self.meas(subdomain_data=markers)
        
        
    # TODO: should probably change name of argument so that the measure
    # can be called exactly like spline.dx(subdomain_data=...)
    def __call__(self,marker):
        """
        This allows ``subdomain_data`` of an existing measure to be 
        changed to ``marker`` after the fact, using the same calling
        syntax that one would use to change ``subdomain_data`` of an
        ordinary measure.
        """
        return tIGArMeasure(self.J,self.meas(marker))
        
    def __rmul__(self, other):
        """
        Multiplies ``other`` by ``self.J``, THEN multiplies by ``self.meas``.
        """
        return (other*self.J)*self.meas

def getQuadRule(n):
    """
    Return a list of points and a list of weights for integration over the
    interval (-1,1), using ``n`` quadrature points.  

    NOTE: This functionality is mainly intended
    for use in through-thickness integration of Kirchhoff--Love shell
    formulations, but might also be useful for implementing space--time
    formulations using a mixed element to combine DoFs from various time
    levels.
    """
    if(n==1):
        xi = [Constant(0.0),]
        w = [Constant(2.0),]
        return (xi,w)
    if(n==2):
        xi = [Constant(-0.5773502691896257645091488),
              Constant(0.5773502691896257645091488)]
        w = [Constant(1.0),
             Constant(1.0)]
        return (xi,w)
    if(n==3):
        xi = [Constant(-0.77459666924148337703585308),
              Constant(0.0),
              Constant(0.77459666924148337703585308)]
        w = [Constant(0.55555555555555555555555556),
             Constant(0.88888888888888888888888889),
             Constant(0.55555555555555555555555556)]
        return (xi,w)
    if(n==4):
        xi = [Constant(-0.86113631159405257524),
              Constant(-0.33998104358485626481),
              Constant(0.33998104358485626481),
              Constant(0.86113631159405257524)]
        w = [Constant(0.34785484513745385736),
             Constant(0.65214515486254614264),
             Constant(0.65214515486254614264),
             Constant(0.34785484513745385736)]
        return (xi,w)
    
    # TODO: add more quadrature rules (or, try to find a function in scipy or
    # another common library, to generate arbitrary Gaussian quadrature
    # rules on-demand).
    
    print("ERROR: invalid number of quadrature points requested.")
    exit()

def getQuadRuleInterval(n,L):
    """
    Returns an ``n``-point quadrature rule for the interval 
    (-``L``/2,``L``/2), consisting of a list of points and list of weights.
    """
    xi_hat, w_hat = getQuadRule(n)
    xi = []
    w = []
    for i in range(0,n):
        xi += [L*xi_hat[i]/2.0,]
        w += [L*w_hat[i]/2.0,]
    return (xi,w)
