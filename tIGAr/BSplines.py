"""
The ``BSplines`` module 
-----------------------
provides a self-contained implementation of B-splines
that can be used to generate simple sets of extraction data for 
rectangular domains.
"""

from tIGAr.common import *
import bisect
from numpy import searchsorted


def uniformKnots(p,start,end,N,periodic=False):
    """
    Helper function to generate a uniform open knot vector of degree ``p`` with
    ``N`` elements.  If ``periodic``, end knots are not repeated.  
    Otherwise, they are repeated ``p+1`` times for an open knot vector.
    """
    retval = []
    if(not periodic):
        for i in range(0,p):
            retval += [start,]
    h = (end - start)/float(N)
    for i in range(0,N+1):
        retval += [start+float(i)*h,]
    if(not periodic):
        for i in range(0,p):
            retval += [end,]
    return retval

# need a custom eps for checking knots; dolfin_eps is too small and doesn't
# reliably catch repeated knots
KNOT_NEAR_EPS = 10.0*DOLFIN_EPS


# cProfile identified basis function evaluation as a bottleneck in the
# preprocessing, so i've moved it into an inline C++ routine, using
# dolfin's extension module compilation
basisFuncsCXXString = """
#include <dolfin/common/Array.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace dolfin {

int flatIndex(int i, int j, int N){
  return i*N + j;
}

//void basisFuncsInner(const Array<double> &ghostKnots,
//                     int nGhost,
//                     double u,
//                     int pl,
//                     int i,
//                     const Array<double> &ndu,
//                     const Array<double> &left,
//                     const Array<double> &right,
//                     const Array<double> &ders){

typedef py::array_t<double, py::array::c_style | py::array::forcecast> 
    npArray;

void basisFuncsInner(npArray ghostKnots,
                     int nGhost,
                     double u,
                     int pl,
                     int i,
                     npArray ndu,
                     npArray left,
                     npArray right,
                     npArray ders){

    // Technically results in un-defined behavior:
    //Array<double> *ghostKnotsp = const_cast<Array<double>*>(&ghostKnots);
    //Array<double> *ndup = const_cast<Array<double>*>(&ndu);
    //Array<double> *leftp = const_cast<Array<double>*>(&left);
    //Array<double> *rightp = const_cast<Array<double>*>(&right);
    //Array<double> *dersp = const_cast<Array<double>*>(&ders);

    auto ghostKnotsb = ghostKnots.request();
    auto ndub = ndu.request();
    auto leftb = left.request();
    auto rightb = right.request();
    auto dersb = ders.request();

    double *ghostKnotsp = (double *)ghostKnotsb.ptr;
    double *ndup = (double *)ndub.ptr;
    double *leftp = (double *)leftb.ptr;
    double *rightp = (double *)rightb.ptr;
    double *dersp = (double *)dersb.ptr;

    int N = pl+1;
    ndup[flatIndex(0,0,N)] = 1.0;
    for(int j=1; j<pl+1; j++){
        leftp[j] = u - ghostKnotsp[i-j+nGhost];
        rightp[j] = ghostKnotsp[i+j-1+nGhost]-u;
        double saved = 0.0;
        for(int r=0; r<j; r++){
            ndup[flatIndex(j,r,N)] = rightp[r+1] + leftp[j-r];
            double temp = ndup[flatIndex(r,j-1,N)]
                          /ndup[flatIndex(j,r,N)];
            ndup[flatIndex(r,j,N)] = saved + rightp[r+1]*temp;
            saved = leftp[j-r]*temp;
        } // r
        ndup[flatIndex(j,j,N)] = saved;
    } // j
    for(int j=0; j<pl+1; j++){
        dersp[j] = ndup[flatIndex(j,pl,N)];
    } // j
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("basisFuncsInner",&basisFuncsInner,"");
}
}
"""

#basisFuncsCXXModule = compile_extension_module(basisFuncsCXXString,
#                                               cppargs='-g -O2')
basisFuncsCXXModule = compile_cpp_code(basisFuncsCXXString)

    
# function to eval B-spline basis functions (for internal use)
def basisFuncsInner(ghostKnots,nGhost,u,pl,i,ndu,left,right,ders):
    
    # TODO: Fix C++ module for 2018.1, and restore this (or equivalent)
    # call to a compiled routine.
    #
    #basisFuncsCXXModule.basisFuncsInner(ghostKnots,nGhost,u,pl,i,
    #                                    ndu.flatten(),
    #                                    left,right,ders)
    basisFuncsCXXModule.basisFuncsInner(ghostKnots,nGhost,u,pl,i,
                                        ndu.flatten(),
                                        left,right,ders)
    
    # Backup Python implementation:
    #
    #ndu[0,0] = 1.0
    #for j in range(1,pl+1):
    #    left[j] = u - ghostKnots[i-j+nGhost]
    #    right[j] = ghostKnots[i+j-1+nGhost]-u
    #    saved = 0.0
    #    for r in range(0,j):
    #        ndu[j,r] = right[r+1] + left[j-r]
    #        temp = ndu[r,j-1]/ndu[j,r]
    #        ndu[r,j] = saved + right[r+1]*temp
    #        saved = left[j-r]*temp
    #    ndu[j,j] = saved
    #for j in range(0,pl+1):
    #    ders[j] = ndu[j,pl]


class BSpline1(object):
    """
    Scalar univariate B-spline; this is used construct tensor products, with a
    univariate "tensor product" as a special case; therefore this does not
    implement the ``AbstractScalarBasis`` interface, even though you might
    initially think that it should.
    """
    
    # create from degree, knot data
    def __init__(self,p,knots):
        """
        Creates a univariate B-spline from a degree, ``p``, and an ordered
        collection of ``knots``.
        """
        self.p = p
        self.knots = array(knots)
        self.computeNel()
        
        # needed for mesh generation
        self.uniqueKnots = zeros(self.nel+1)
        self.multiplicities = zeros(self.nel+1,dtype=INDEX_TYPE)
        ct = -1
        lastKnot = None
        for i in range(0,len(self.knots)):
            if(lastKnot == None or (not near(self.knots[i],lastKnot,\
                                             eps=KNOT_NEAR_EPS))):
                ct += 1
                self.uniqueKnots[ct] = knots[i]
            lastKnot = knots[i]
            self.multiplicities[ct] += 1
        self.ncp = self.computeNcp()

        # knot array with ghosts, for optimized access
        self.nGhost = self.p+1
        #self.ghostKnots = []
        #for i in range(-self.nGhost,len(self.knots)+self.nGhost):
        #    self.ghostKnots += [self.getKnot(i),]
        #self.ghostKnots = array(self.ghostKnots)
        self.ghostKnots = self.computeGhostKnots()

    def computeGhostKnots(self):
        """
        Pre-compute ghost knots and return as a numpy array.  (Primarily
        intended for internal use.)
        """
        ghostKnots = []
        for i in range(-self.nGhost,len(self.knots)+self.nGhost):
            ghostKnots += [self.getKnot(i),]
        return array(ghostKnots)
        
    def normalizeKnotVector(self):
        """
        Re-scales knot vector to be from 0 to 1.
        """
        L = self.knots[-1] - self.knots[0]
        self.knots = (self.knots - self.knots[0])/L
        self.uniqueKnots = (self.uniqueKnots - self.uniqueKnots[0])/L
        self.ghostKnots = self.computeGhostKnots()
        
    # if any non-end knot is repeated more than p time, then the B-spline
    # is discontinuous
    def isDiscontinuous(self):
        """
        Returns a Boolean indicating whether or not the B-spline is 
        discontinuous.  
        """
        for i in range(1,len(self.uniqueKnots)-1):
            if(self.multiplicities[i] > self.p):
                return True
        return False
        
    def computeNel(self):
        """
        Returns the number of non-degenerate knot spans.
        """
        self.nel = 0
        for i in range(1,len(self.knots)):
            if(not near(self.knots[i],self.knots[i-1],\
                        eps=KNOT_NEAR_EPS)):
                self.nel += 1

    def getKnot(self,i):
        """
        return a knot, with a (possibly) out-of-range index ``i``.  If ``i``
        is out of range, ghost knots are conjured by looking at the other 
        end of the vector.
        Assumes that the first and last unique knots are duplicates with the
        same multiplicity.
        """
        if(i<0):
            ii = len(self.knots) - self.multiplicities[-1] + i
            return self.knots[0] - (self.knots[-1] - self.knots[ii])
        elif(i>=len(self.knots)):
            ii = i-len(self.knots) + self.multiplicities[0]
            return self.knots[-1] + (self.knots[ii] - self.knots[0])
        else:
            return self.knots[i]
                
    def greville(self,i):
        """
        Returns the Greville parameter associated with the 
        ``i``-th control point.
        """
        retval = 0.0
        for j in range(i,i+self.p):
            retval += self.getKnot(j+1)
        retval /= float(self.p)
        return retval
        
    def computeNcp(self):
        """
        Computes and returns the number of control points.
        """
        return len(self.knots) - self.multiplicities[0]

    def getNcp(self):
        """
        Returns the number of control points.
        """
        return self.ncp

    def getKnotSpan(self,u):
        """
        Given parameter ``u``, return the index of the knot span in which
        ``u`` falls.  (Numbering includes degenerate knot spans.)
        """
        
        # placeholder linear search
        #span = 0
        #nspans = len(self.knots)-1
        #for i in range(0,nspans):
        #    span = i
        #    if(u<self.knots[i+1]+DOLFIN_EPS):
        #        break

        # from docs: should be index of "rightmost value less than x"
        nspans = len(self.knots)-1
        #span = bisect.bisect_left(self.knots,u)-1
        span = searchsorted(self.knots,u)-1
        
        if(span < self.multiplicities[0]-1):
            span = self.multiplicities[0]-1
        if(span > nspans-(self.multiplicities[-1]-1)-1):
            span = nspans-(self.multiplicities[-1]-1)-1
        return span

    def getNodes(self,u):
        """
        Given a parameter ``u``, return a list of the indices of B-spline
        basis functions whose supports contain ``u``.
        """
        nodes = []
        knotSpan = self.getKnotSpan(u)
        for i in range(knotSpan-self.p,knotSpan+1):
            nodes += [i % self.getNcp(),]
        return nodes

    def basisFuncs(self,knotSpan,u):
        """
        Return a list of the ``p+1`` nonzero basis functions evaluated at 
        the parameter ``u`` in the knot span with index ``knotSpan``.
        """
        pl = self.p
        #u_knotl = self.knots
        i = knotSpan+1
        ndu = zeros((pl+1,pl+1))
        left = zeros(pl+1)
        right = zeros(pl+1)
        ders = zeros(pl+1)

        basisFuncsInner(self.ghostKnots,self.nGhost,u,
                        pl,i,ndu,left,right,ders)
        
        #ndu[0,0] = 1.0
        #for j in range(1,pl+1):
        #    left[j] = u - self.getKnot(i-j) #u_knotl[i-j]
        #    right[j] = self.getKnot(i+j-1)-u #u_knotl[i+j-1]-u
        #    saved = 0.0
        #    for r in range(0,j):
        #        ndu[j,r] = right[r+1] + left[j-r]
        #        temp = ndu[r,j-1]/ndu[j,r]
        #        ndu[r,j] = saved + right[r+1]*temp
        #        saved = left[j-r]*temp
        #    ndu[j,j] = saved
        #for j in range(0,pl+1):
        #    ders[j] = ndu[j,pl]

        return ders

# utility functions for indexing (mainly for internal library use)
def ij2dof(i,j,M):
    return j*M + i

def ijk2dof(i,j,k,M,N):
    return k*(M*N) + j*M + i

def dof2ij(dof,M):
    i = dof%M
    j = dof//M
    return (i,j)

def dof2ijk(dof,M,N):
    ij = dof%(M*N)
    i = ij%M
    j = ij//M
    k = dof//(M*N)
    return (i,j,k)

# Use BSpline1 instances to store info about each dimension.  No point in going
# higher than 3, since FEniCS only generates meshes up to dimension 3...
class BSpline(AbstractScalarBasis):

    """
    Class implementing the ``AbstractScalarBasis`` interface, to represent
    a uni-, bi-, or tri-variate B-spline.
    """

    def __init__(self,degrees,kvecs,useRect=USE_RECT_ELEM_DEFAULT,
                 overRefine=0):
        """
        Create a ``BSpline`` with degrees in each direction given by the
        sequence ``degrees``, knot vectors given by the list of 
        sequences ``kvecs``, and an optional Boolean parameter 
        ``useRect``, indicating
        whether or not rectangular elements should be used in the 
        extracted representation.  The optional parameter ``overRefine``
        indicates how many levels of refinement to apply beyond what is
        needed to represent the spline functions; choosing a value greater
        than the default of zero may be useful for 
        integrating functions with fine-scale features.
        
        NOTE: Over-refinement is only supported with simplicial elements.
        """
        self.nvar = len(degrees)
        if(self.nvar > 3 or self.nvar < 1):
            print("ERROR: Unsupported parametric dimension.")
            exit()
        self.splines = []
        for i in range(0,self.nvar):
            self.splines += [BSpline1(degrees[i],kvecs[i]),]
        self.useRect = useRect
        self.overRefine = overRefine
        self.ncp = self.computeNcp()
        self.nel = self.computeNel()

    def normalizeKnotVectors(self):
        """
        Scale knot vectors in all directions to (0,1).
        """
        for s in self.splines:
            s.normalizeKnotVector()
        
    #def getParametricDimension(self):
    #    return self.nvar

    def needsDG(self):
        """
        Returns a Boolean, indicating whether or not the extraction requires
        DG element (due to the function space being discontinuous somewhere).
        """
        for i in range(0,self.nvar):
            if(self.splines[i].isDiscontinuous()):
                return True
        return False

    def useRectangularElements(self):
        """
        Returns a Boolean indicating whether or not the basis should use
        rectangular elements in its extraction.
        """
        return self.useRect

    # non-default implementation, optimized for B-splines
    def getPrealloc(self):
        totalFuncs = 1
        for spline in self.splines:
            # a 1d b-spline should have p+1 active basis functions at any given
            # point; however, when eval-ing at boundaries of knot spans,
            # may pick up at most 2 extra basis functions due to epsilons.
            #
            # TODO: find a way to automatically ignore those extra nearly-zero
            # basis functions.
            #totalFuncs *= (spline.p+1 +2)
            totalFuncs *= (spline.p+1)
        return totalFuncs
    
    def getNodesAndEvals(self,xi):

        if(self.nvar == 1):
            u = xi[0]
            span = self.splines[0].getKnotSpan(u)
            nodes = self.splines[0].getNodes(u)
            ders = self.splines[0].basisFuncs(span,u)
            retval = []
            for i in range(0,len(nodes)):
                retval += [[nodes[i],ders[i]],]
            return retval
        elif(self.nvar == 2):
            u = xi[0]
            v = xi[1]
            uspline = self.splines[0]
            vspline = self.splines[1]
            spanu = uspline.getKnotSpan(u)
            spanv = vspline.getKnotSpan(v)
            nodesu = uspline.getNodes(u)
            nodesv = vspline.getNodes(v)
            dersu = uspline.basisFuncs(spanu,u)
            dersv = vspline.basisFuncs(spanv,v)
            retval = []
            for i in range(0,len(nodesu)):
                for j in range(0,len(nodesv)):
                    retval += [[ij2dof(nodesu[i],nodesv[j],\
                                      uspline.getNcp()),\
                                dersu[i]*dersv[j]],]
            return retval
        else:
            u = xi[0]
            v = xi[1]
            w = xi[2]
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            spanu = uspline.getKnotSpan(u)
            spanv = vspline.getKnotSpan(v)
            spanw = wspline.getKnotSpan(w)
            nodesu = uspline.getNodes(u)
            nodesv = vspline.getNodes(v)
            nodesw = wspline.getNodes(w)
            dersu = uspline.basisFuncs(spanu,u)
            dersv = vspline.basisFuncs(spanv,v)
            dersw = wspline.basisFuncs(spanw,w)
            retval = []
            for i in range(0,len(nodesu)):
                for j in range(0,len(nodesv)):
                    for k in range(0,len(nodesw)):
                        retval += [[ijk2dof\
                                    (nodesu[i],nodesv[j],nodesw[k],\
                                     uspline.getNcp(),vspline.getNcp()),\
                                    dersu[i]*dersv[j]*dersw[k]],]
            return retval
        
    def generateMesh(self,comm=worldcomm):
        if(self.nvar == 1):
            spline = self.splines[0]
            mesh = IntervalMesh(comm,spline.nel,0.0,float(spline.nel))
            x = mesh.coordinates()
            xbar = zeros((len(x),1))
            for i in range(0,len(x)):
                knotIndex = int(round(x[i,0]))
                xbar[i,0] = spline.uniqueKnots[knotIndex]
            mesh.coordinates()[:] = xbar
            #return mesh
        elif(self.nvar == 2):
            uspline = self.splines[0]
            vspline = self.splines[1]
            if(self.useRect):
                cellType = CellType.Type.quadrilateral
            else:
                cellType = CellType.Type.triangle
            mesh = UnitSquareMesh.create(comm,uspline.nel,vspline.nel,cellType)
            #mesh = RectangleMesh(Point(0.0,0.0),\
            #                     Point(uspline.nel,vspline.nel),\
            #                     uspline.nel,vspline.nel)
            x = mesh.coordinates()
            xbar = zeros((len(x),2))
            for i in range(0,len(x)):
                #uknotIndex = int(round(x[i,0]))
                #vknotIndex = int(round(x[i,1]))
                uknotIndex = int(round(x[i,0]*float(uspline.nel)))
                vknotIndex = int(round(x[i,1]*float(vspline.nel)))
                xbar[i,0] = uspline.uniqueKnots[uknotIndex]
                xbar[i,1] = vspline.uniqueKnots[vknotIndex]
            mesh.coordinates()[:] = xbar
            #return mesh
        else:
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            if(self.useRect):
                cellType = CellType.Type.hexahedron
            else:
                cellType = CellType.Type.tetrahedron
            mesh = UnitCubeMesh.create(comm,
                                       uspline.nel,vspline.nel,wspline.nel,
                                       cellType)
            #mesh = BoxMesh(Point(0.0,0.0,0.0),\
            #               Point(uspline.nel,vspline.nel,wspline.nel),\
            #               uspline.nel,vspline.nel,wspline.nel)
            x = mesh.coordinates()
            xbar = zeros((len(x),3))
            for i in range(0,len(x)):
                #uknotIndex = int(round(x[i,0]))
                #vknotIndex = int(round(x[i,1]))
                #wknotIndex = int(round(x[i,2]))
                uknotIndex = int(round(x[i,0]*float(uspline.nel)))
                vknotIndex = int(round(x[i,1]*float(vspline.nel)))
                wknotIndex = int(round(x[i,2]*float(wspline.nel)))
                xbar[i,0] = uspline.uniqueKnots[uknotIndex]
                xbar[i,1] = vspline.uniqueKnots[vknotIndex]
                xbar[i,2] = wspline.uniqueKnots[wknotIndex]
            mesh.coordinates()[:] = xbar

        # Apply any over-refinement specified:
        for i in range(0,self.overRefine):
            mesh = refine(mesh)
        return mesh
        
    def computeNcp(self):
        prod = 1
        for i in range(0,self.nvar):
            prod *= self.splines[i].getNcp()
        return prod

    def getNcp(self):
        return self.ncp

    def getDegree(self):
        deg = 0
        for i in range(0,self.nvar):
            # for simplex elements; take max for rectangles
            if(self.useRect):
                deg = max(deg,self.splines[i].p)
            else:
                deg += self.splines[i].p
        return deg

    def computeNel(self):
        """
        Returns the number of Bezier elements in the B-spline.
        """
        nel = 1
        for spline in self.splines:
            nel *= spline.nel
        return nel
    
    def getSideDofs(self,direction,side,nLayers=1):
        """
        Return the DoFs on a ``side`` (zero or one) that is perpendicular 
        to a parametric ``direction`` (0, 1, or 2, capped at 
        ``self.nvar-1``, obviously).  Can optionally constrain more than
        one layer of control points (e.g., for strongly-enforced clamped BCs
        on Kirchhoff--Love shells) using ``nLayers`` greater than its
        default value of one.
        """
        offsetSign = 1-2*side
        retval = []
        for absOffset in range(0,nLayers):
            offset = absOffset*offsetSign
            if(side == 0):
                i=0
            else:
                i=self.splines[direction].getNcp()-1
            i += offset
            M = self.splines[0].getNcp()
            if(self.nvar == 1):
                retval += [i,]
                continue
            N = self.splines[1].getNcp()
            if(self.nvar == 2):
                dofs = []
                if(direction==0):
                    for j in range(0,N):
                        dofs += [ij2dof(i,j,M),]
                elif(direction==1):
                    for j in range(0,M):
                        dofs += [ij2dof(j,i,M),]
                retval += dofs
                continue
            O = self.splines[2].getNcp()
            if(self.nvar == 3):
                dofs = []
                if(direction==0):
                    for j in range(0,N):
                        for k in range(0,O):
                            dofs += [ijk2dof(i,j,k,M,N),]
                elif(direction==1):
                    for j in range(0,M):
                        for k in range(0,O):
                            dofs += [ijk2dof(j,i,k,M,N),]
                elif(direction==2):
                    for j in range(0,M):
                        for k in range(0,N):
                            dofs += [ijk2dof(j,k,i,M,N),]
                retval += dofs
                continue
        return retval
                
class MultiBSpline(AbstractScalarBasis):
    """
    Several ``BSpline`` instances grouped together.
    """

    # TODO: add a mechanism to merge basis functions (analogous to IPER
    # in the Fortran code) that can be used to merge control points for
    # equal-order interpolations.  (Should be an integer array of length
    # self.ncp, with mostly array[i]=i, except for slave nodes.)
    
    def __init__(self,splines):
        """
        Create a ``MultiBSpline`` from a sequence ``splines`` of individual
        ``BSpline`` instances.  This sequence is assumed to contain at least
        one ``BSpline``, and all elements of ``splines`` are assumed to use
        the same element type, and have the same parametric 
        dimensions as each other.
        """
        self.splines = splines
        self.ncp = self.computeNcp()

        # normalize all knot vectors to (0,1) for each patch, for easy lookup
        # of patch index from coordinates
        for s in self.splines:
            s.normalizeKnotVectors()

        # pre-compute DoF index offsets for each patch
        self.doffsets = []
        ncp = 0
        for s in self.splines:
            self.doffsets += [ncp,]
            ncp += s.getNcp()

        self.nvar = self.splines[0].nvar
        self.useRect = self.splines[0].useRect
        self.overRefine = self.splines[0].overRefine
        self.nPatch = len(self.splines)
        self.nel = self.computeNel()

    def computeNel(self):
        """
        Returns the number of Bezier elements between all patches.
        """
        nel = 0
        for spline in self.splines:
            nel += spline.nel
        return nel
        
    # TODO: this should not need to exist
    def needsDG(self):
        return False

    def useRectangularElements(self):
        """
        Returns a Boolean indicating whether or not the basis should use
        rectangular elements in its extraction.
        """
        return self.useRect

    # non-default implementation, optimized for B-splines
    def getPrealloc(self):
        return self.splines[0].getPrealloc()
    
    def getNodesAndEvals(self,xi):
        patch = self.patchFromCoordinates(xi)
        xi_local = self.localParametricCoordinates(xi,patch)
        localNodesAndEvals = self.splines[patch].getNodesAndEvals(xi_local)
        retval = []
        for pair in localNodesAndEvals:
            retval += [[self.globalDofIndex(pair[0],patch),pair[1]],]
        return retval
        
    def patchFromCoordinates(self,xi):
        return int(xi[0]+0.5)//2
        
    def globalDofIndex(self,localDofIndex,patchIndex):
        return self.doffsets[patchIndex] + localDofIndex

    def localParametricCoordinates(self,xi,patchIndex):
        retval = xi.copy()
        retval[0] = xi[0] - 2.0*float(patchIndex)
        return retval

    def generateMesh(self,comm=worldcomm):

        MESH_FILE_NAME = generateMeshXMLFileName(comm)
        
        if(MPI.rank(comm) == 0):
            fs = '<?xml version="1.0" encoding="UTF-8"?>' + "\n"
            fs += '<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">'+"\n"
            if(self.nvar == 1):
                # TODO
                print("ERROR: Univariate multipatch not yet supported.")
                exit()
            elif(self.nvar == 2):
                if(self.useRect):
                    fs += '<mesh celltype="quadrilateral" dim="2">' + "\n"

                    # TODO: Do indexing more intelligently, so that elements
                    # are connected within each patch.  (This will improve
                    # parallel performance.)
                    
                    nverts = 4*self.nel
                    nel = self.nel
                    fs += '<vertices size="'+str(nverts)+'">' + "\n"
                    vertCounter = 0
                    x00 = 0.0
                    for patch in range(0,self.nPatch):
                        spline = self.splines[patch]
                        uspline = spline.splines[0]
                        vspline = spline.splines[1]
                        for i in range(0,uspline.nel):
                            for j in range(0,vspline.nel):
                                x0 = repr(x00+uspline.uniqueKnots[i])
                                x1 = repr(x00+uspline.uniqueKnots[i+1])
                                y0 = repr(vspline.uniqueKnots[j])
                                y1 = repr(vspline.uniqueKnots[j+1])
                                fs += '<vertex index="'+str(vertCounter)\
                                      +'" x="'+x0+'" y="'+y0+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+1)\
                                      +'" x="'+x1+'" y="'+y0+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+2)\
                                      +'" x="'+x0+'" y="'+y1+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+3)\
                                      +'" x="'+x1+'" y="'+y1+'"/>' + "\n"
                                vertCounter += 4
                        x00 += 2.0
                    fs += '</vertices>' + "\n"
                    fs += '<cells size="'+str(nel)+'">' + "\n"
                    elCounter = 0
                    for patch in range(0,self.nPatch):
                        spline = self.splines[patch]
                        uspline = spline.splines[0]
                        vspline = spline.splines[1]
                        for i in range(0,uspline.nel):
                            for j in range(0,vspline.nel):
                                v0 = str(elCounter*4+0)
                                v1 = str(elCounter*4+1)
                                v2 = str(elCounter*4+2)
                                v3 = str(elCounter*4+3)
                                fs += '<quadrilateral index="'+str(elCounter)\
                                      +'" v0="'+v0+'" v1="'+v1\
                                      +'" v2="'+v2+'" v3="'+v3+'"/>'\
                                      + "\n"
                                elCounter += 1
                    fs += '</cells></mesh></dolfin>'
                else:

                    fs += '<mesh celltype="triangle" dim="2">' + "\n"

                    # TODO: Do indexing more intelligently, so that elements
                    # are connected within each patch.  (This will improve
                    # parallel performance.)

                    # TODO: Reduce amount of redundant copy--pasted code
                    # from quads to tris.
                    
                    nverts = 4*self.nel
                    nel = 2*self.nel # (two triangles per Bezier element)
                    fs += '<vertices size="'+str(nverts)+'">' + "\n"
                    vertCounter = 0
                    x00 = 0.0
                    for patch in range(0,self.nPatch):
                        spline = self.splines[patch]
                        uspline = spline.splines[0]
                        vspline = spline.splines[1]
                        for i in range(0,uspline.nel):
                            for j in range(0,vspline.nel):
                                x0 = repr(x00+uspline.uniqueKnots[i])
                                x1 = repr(x00+uspline.uniqueKnots[i+1])
                                y0 = repr(vspline.uniqueKnots[j])
                                y1 = repr(vspline.uniqueKnots[j+1])
                                fs += '<vertex index="'+str(vertCounter)\
                                      +'" x="'+x0+'" y="'+y0+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+1)\
                                      +'" x="'+x1+'" y="'+y0+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+2)\
                                      +'" x="'+x0+'" y="'+y1+'"/>' + "\n"
                                fs += '<vertex index="'+str(vertCounter+3)\
                                      +'" x="'+x1+'" y="'+y1+'"/>' + "\n"
                                vertCounter += 4
                        x00 += 2.0
                    fs += '</vertices>' + "\n"
                    fs += '<cells size="'+str(nel)+'">' + "\n"
                    elCounter = 0
                    for patch in range(0,self.nPatch):
                        spline = self.splines[patch]
                        uspline = spline.splines[0]
                        vspline = spline.splines[1]
                        for i in range(0,uspline.nel):
                            for j in range(0,vspline.nel):
                                bezElCounter = elCounter//2
                                v0 = str(bezElCounter*4+0)
                                v1 = str(bezElCounter*4+1)
                                v2 = str(bezElCounter*4+3)
                                fs += '<triangle index="'+str(elCounter)\
                                      +'" v0="'+v0+'" v1="'+v1\
                                      +'" v2="'+v2+'"/>'\
                                      + "\n"
                                v0 = str(bezElCounter*4+0)
                                v1 = str(bezElCounter*4+3)
                                v2 = str(bezElCounter*4+2)
                                fs += '<triangle index="'+str(elCounter+1)\
                                      +'" v0="'+v0+'" v1="'+v1\
                                      +'" v2="'+v2+'"/>'\
                                      + "\n"
                                elCounter += 2
                    fs += '</cells></mesh></dolfin>'
            elif(self.nvar == 3):
                # TODO
                print("ERROR: Trivariate multipatch not yet supported.")
                exit()
            else:
                # TO NOT DO...
                print("ERROR: Unsupported parametric dimension: "
                      +str(self.nvar))
                exit()
            f = open(MESH_FILE_NAME,'w')
            f.write(fs)
            f.close()
                
        MPI.barrier(comm)
        mesh = Mesh(comm,MESH_FILE_NAME)

        if(MPI.rank(comm)==0):
            import os
            os.system("rm "+MESH_FILE_NAME)

        # Apply any over-refinement specified:
        for i in range(0,self.overRefine):
            mesh = refine(mesh)
            
        return mesh
    
    def computeNcp(self):
        ncp = 0
        for s in self.splines:
            ncp += s.getNcp()
        return ncp

    def getNcp(self):
        return self.ncp

    def getDegree(self):
        # assumes all splines have same degree
        return self.splines[0].getDegree()

    def getPatchSideDofs(self,patch,direction,side,nLayers=1):
        """
        This is analogous to the ``BSpline`` method ``getSideDofs()``, but
        it has an extra argument ``patch`` to indicate which patch to obtain
        DoFs from.  The returned DoFs are in the global numbering.
        """
        localSideDofs = self.splines[patch].getSideDofs(direction,side,nLayers)
        retval = []
        for dof in localSideDofs:
            retval += [self.globalDofIndex(dof,patch),]
        return retval
    
class ExplicitBSplineControlMesh(AbstractControlMesh):

    """
    A control mesh for a B-spline with identical physical and parametric
    domains.
    """
    
    def __init__(self,degrees,kvecs,extraDim=0,useRect=USE_RECT_ELEM_DEFAULT,
                 overRefine=0):
        """
        Create an ``ExplicitBSplineControlMesh`` with degrees in each direction
        given by the sequence ``degrees`` and knot vectors given by the list
        of sequences ``kvecs``.  The optional Boolean parameter ``useRect``
        indicates whether or not to use rectangular FEs in the extraction.
        """
        self.scalarSpline = BSpline(degrees,kvecs,
                                    useRect=useRect,
                                    overRefine=overRefine)
        # parametric == physical
        self.nvar = len(degrees)
        self.nsd = self.nvar + extraDim

    def getScalarSpline(self):
        return self.scalarSpline
        
    def getHomogeneousCoordinate(self,node,direction):
        # B-spline
        if(direction == self.nsd):
            return 1.0
        # otherwise, get coordinate (homogeneous == ordniary for B-spline)
        # for explicit spline, space directions and parametric directions
        # coincide
        if(direction < self.nvar):
            if(self.nvar == 1):
                directionalIndex = node
            elif(self.nvar == 2):
                directionalIndex \
                    = dof2ij(node,\
                             self.scalarSpline.splines[0].getNcp())\
                             [direction]
            else:
                M = self.scalarSpline.splines[0].getNcp()
                N = self.scalarSpline.splines[1].getNcp()
                directionalIndex = dof2ijk(node,M,N)[direction]

            # use Greville points for explicit spline
            coord = self.scalarSpline.splines[direction].greville\
                    (directionalIndex)
        else:
            coord = 0.0
        return coord

    def getNsd(self):
        return self.nsd
    
# TODO: think about re-organization, as this is NURBS functionality (but
# does not rely on igakit)
class LegacyMultipatchControlMesh(AbstractControlMesh):
    """
    A class to generate a multi-patch NURBS from data given in a legacy 
    ASCII format used by some early prototype IGA codes from the Hughes 
    group at UT Austin.
    """
    
    def __init__(self,prefix,nPatch,suffix,
                 useRect=USE_RECT_ELEM_DEFAULT,
                 overRefine=0):
        """
        Loads a collection of ``nPatch`` files with names of the form
        ``prefix+str(i+1)+suffix``, for ``i in range(0,nPatch)``, where each 
        file contains data for a NURBS patch, in the ASCII format used by 
        J. A. Cottrell's preprocessor.  (The ``+1`` in the file name 
        convention comes from Fortran indexing.)  The optional argument
        ``useRect`` is a Boolean, indicating whether or not to use
        rectangular elements.  The optional argument ``overRefine`` can
        specify a number of global refinements of the FE mesh used for
        extraction.  (This does not refine the IGA space.)  Over-refinement
        is only supported for simplicial elements.

        The parametric dimension is inferred from the contents of the first 
        file, and assumed to be the same for all patches.
        """

        # Accummulate B-splines for each patch's scalar basis here
        splines = []
        # Empty control net, to be filled in with pts in homogeneous coords
        self.bnet = []
        # sentinel value for parametric and physical dimensions
        nvar = -1
        self.nsd = -1
        for i in range(0,nPatch):

            # Read contents of file
            fname = prefix + str(i+1) + suffix
            f = open(fname,'r')
            fs = f.read()
            f.close()
            lines = fs.split('\n')

            # infer parametric dimension from the number of
            # whitespace-delimited tokens on the second line
            if(nvar==-1):
                self.nsd = int(lines[0])
                nvar = len(lines[1].split())

            # Load general info on $\hat{d}$, spline degrees, number of CPs    
            degrees = []
            degStrs = lines[1].split()
            ncps = []
            ncpStrs = lines[2].split()
            for d in range(0,nvar):
                degrees += [int(degStrs[d]),]
                ncps += [int(ncpStrs[d]),]

            # Load knot vector for each parametric dimension
            kvecs = []
            for d in range(0,nvar):
                kvecStrs = lines[3+d].split()
                kvec = []
                for s in kvecStrs:
                    kvec += [float(s),]
                kvecs += [array(kvec),]

            # Use the knot vectors to create a B-spline basis for this patch
            splines += [BSpline(degrees,kvecs,useRect,overRefine),]

            # Load control points
            ncp = 1
            for d in range(0,nvar):
                ncp *= ncps[d]

            # Note: this only works for all parametric dimensions because
            # the ij2dof and ijk2dof functions follow the same convention of
            # having i as the fastest-varying index, j as the next-fastest,
            # and k as the outer loop.
            for pt in range(0,ncp):
                bnetRow = []
                coordStrs = lines[3+nvar+pt].split()
                w = float(coordStrs[self.nsd])
                # NOTE: bnet should be in homogeneous coordinates
                for d in range(0,self.nsd):
                    bnetRow += [float(coordStrs[d])*w,]
                bnetRow += [w,]
                # Note: filling of control pts in global, multi-patch bnet is
                # consistent with the globalDofIndex() method of
                # MultiBSpline
                self.bnet += [bnetRow,]
                
            # TODO: formats for different parametric dimensions diverge
            # after this point, and additional data (element types, etc.)
            # needs to be loaded in an nvar-dependent way.  Ignoring extra
            # data for now...

        # create the scalar spline instance to be used for all components of
        # the control mapping.
        self.scalarSpline = MultiBSpline(splines)

        # Make lookup faster
        self.bnet = array(self.bnet)
        
    # TODO: include some functionality to match up CPs w/in epsilon of
    # each other and construct an IPER array.  (Cf. TODO in MultiBSpline.)
    # Should be able to use the SciPy KD tree to do this in a few lines.
        
    # Required interface for an AbstractControlMesh:
    def getHomogeneousCoordinate(self,node,direction):
        return self.bnet[node,direction]
    def getScalarSpline(self):
        return self.scalarSpline
    def getNsd(self):
        return self.nsd
