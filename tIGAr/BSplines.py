from tIGAr.common import *
import bisect
from numpy import searchsorted

# helper function to generate a uniform open knot vector for degree p with
# N elements.  If periodic, end knots are not repeated.  Otherwise, they
# are repeated p+1 times for an open knot vector
def uniformKnots(p,start,end,N,periodic=False):
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
namespace dolfin {

int flatIndex(int i, int j, int N){
  return i*N + j;
}

void basisFuncsInner(const Array<double> &ghostKnots,
                     int nGhost,
                     double u,
                     int pl,
                     int i,
                     const Array<double> &ndu,
                     const Array<double> &left,
                     const Array<double> &right,
                     const Array<double> &ders){

    Array<double> *ghostKnotsp = &ghostKnots;
    Array<double> *ndup = &ndu;
    Array<double> *leftp = &left;
    Array<double> *rightp = &right;
    Array<double> *dersp = &ders;

    int N = pl+1;
    (*ndup)[flatIndex(0,0,N)] = 1.0;
    for(int j=1; j<pl+1; j++){
        (*leftp)[j] = u - (*ghostKnotsp)[i-j+nGhost];
        (*rightp)[j] = (*ghostKnotsp)[i+j-1+nGhost]-u;
        double saved = 0.0;
        for(int r=0; r<j; r++){
            (*ndup)[flatIndex(j,r,N)] = (*rightp)[r+1] + (*leftp)[j-r];
            double temp = (*ndup)[flatIndex(r,j-1,N)]
                          /(*ndup)[flatIndex(j,r,N)];
            (*ndup)[flatIndex(r,j,N)] = saved + (*rightp)[r+1]*temp;
            saved = (*leftp)[j-r]*temp;
        } // r
        (*ndup)[flatIndex(j,j,N)] = saved;
    } // j
    for(int j=0; j<pl+1; j++){
        (*dersp)[j] = (*ndup)[flatIndex(j,pl,N)];
    } // j
}
}
"""

basisFuncsCXXModule = compile_extension_module(basisFuncsCXXString,
                                               cppargs='-g -O2')

# function to eval B-spline basis functions
def basisFuncsInner(ghostKnots,nGhost,u,pl,i,ndu,left,right,ders):

    basisFuncsCXXModule.basisFuncsInner(ghostKnots,nGhost,u,pl,i,
                                        ndu.flatten(),
                                        left,right,ders)
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


# scalar univariate b-spline; this is used construct tensor products, with a
# univariate "tensor product" as a special case; therefore this does not
# implement the AbstractScalarBasis interface, even though you might think
# that it should.
class BSpline1(object):

    # create from degree, knot data
    def __init__(self,p,knots):
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
        self.ghostKnots = []
        for i in range(-self.nGhost,len(self.knots)+self.nGhost):
            self.ghostKnots += [self.getKnot(i),]
        self.ghostKnots = array(self.ghostKnots)

    # if any non-end knot is repeated more than p time, then the B-spline
    # is discontinuous
    def isDiscontinuous(self):
        for i in range(1,len(self.uniqueKnots)-1):
            if(self.multiplicities[i] > self.p):
                return True
        return False
        
    # get the number of non-degenerate knot spans
    def computeNel(self):
        self.nel = 0
        for i in range(1,len(self.knots)):
            if(not near(self.knots[i],self.knots[i-1],\
                        eps=KNOT_NEAR_EPS)):
                self.nel += 1

    # return a knot, possibly with an out-of-range index, in which case
    # ghost knots are conjured by looking at the other end of the vector.
    # assume that the first and last unique knots are duplicates with the same
    # multiplicity
    def getKnot(self,i):
        if(i<0):
            ii = len(self.knots) - self.multiplicities[-1] + i
            return self.knots[0] - (self.knots[-1] - self.knots[ii])
        elif(i>=len(self.knots)):
            ii = i-len(self.knots) + self.multiplicities[0]
            return self.knots[-1] + (self.knots[ii] - self.knots[0])
        else:
            return self.knots[i]
                
    def greville(self,i):
        retval = 0.0
        for j in range(i,i+self.p):
            retval += self.getKnot(j+1)
        retval /= float(self.p)
        return retval

        
    def computeNcp(self):
        return len(self.knots) - self.multiplicities[0]

    def getNcp(self):
        return self.ncp

    # given a parameter, return the knot span
    def getKnotSpan(self,u):

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

    # given a param u, return a list of indexes of IGA basis functions whose
    # supports contain u
    def getNodes(self,u):
        nodes = []
        knotSpan = self.getKnotSpan(u)
        for i in range(knotSpan-self.p,knotSpan+1):
            nodes += [i % self.getNcp(),]
        return nodes

    # the index knotSpan is the index of a knot span (including zero-thickness
    # ones from repeated knots) starting from zero
    def basisFuncs(self,knotSpan,u):
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

# utility functions for indexing
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

# create a scalar B-spline of parametric dimension 1, 2, or (TODO) 3, using
# BSpline1 instances to store info about each dimension.  No point in going
# higher than 3, since FEniCS only generates meshes up to dimension 3...
class BSpline(AbstractScalarBasis):

    def __init__(self,degrees,kvecs,useRect=USE_RECT_ELEM_DEFAULT):
        self.nvar = len(degrees)
        if(self.nvar > 3 or self.nvar < 1):
            print("ERROR: Unsupported parametric dimension.")
            exit()
        self.splines = []
        for i in range(0,self.nvar):
            self.splines += [BSpline1(degrees[i],kvecs[i]),]
        self.useRect = useRect

        self.ncp = self.computeNcp()
        
    #def getParametricDimension(self):
    #    return self.nvar

    def needsDG(self):
        # check whether any of the univariate splines is discontinuous
        for i in range(0,self.nvar):
            if(self.splines[i].isDiscontinuous()):
                return True
        return False

    def useRectangularElements(self):
        return self.useRect
    
    def getPrealloc(self):
        totalFuncs = 1
        for spline in self.splines:
            # a 1d b-spline should have p+1 active basis functions at any given
            # point; however, when eval-ing at boundaries of knot spans,
            # may pick up at most 2 extra basis functions due to epsilons.
            #
            # TODO: find a way to automatically ignore those extra nearly-zero
            # basis functions.
            totalFuncs *= (spline.p+1 +2)
        return totalFuncs
    
    def getNodesAndEvals(self,xi):

        if(self.nvar == 1):
            span = self.splines[0].getKnotSpan(xi)
            nodes = self.splines[0].getNodes(xi)
            ders = self.splines[0].basisFuncs(span,xi)
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
        
    def generateMesh(self):        
        if(self.nvar == 1):
            spline = self.splines[0]
            mesh = IntervalMesh(spline.nel,0.0,float(spline.nel))
            x = mesh.coordinates()
            xbar = zeros((len(x),1))
            for i in range(0,len(x)):
                knotIndex = int(round(x[i,0]))
                xbar[i,0] = spline.uniqueKnots[knotIndex]
            mesh.coordinates()[:] = xbar
            return mesh
        elif(self.nvar == 2):
            uspline = self.splines[0]
            vspline = self.splines[1]
            if(self.useRect):
                cellType = CellType.Type_quadrilateral
            else:
                cellType = CellType.Type_triangle
            mesh = UnitSquareMesh.create(uspline.nel,vspline.nel,cellType)
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
            return mesh
        else:
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            if(self.useRect):
                cellType = CellType.Type_hexahedron
            else:
                cellType = CellType.Type_tetrahedron
            mesh = UnitCubeMesh.create(uspline.nel,vspline.nel,wspline.nel,
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

    # return the dofs on a side (zero or one) perpendicular to a parametric
    # direction (0, 1, or 2), capped at self.nvar-1, obviously
    def getSideDofs(self,direction,side):
        if(side == 0):
            i=0
        else:
            i=self.splines[direction].getNcp()-1
        M = self.splines[0].getNcp()
        if(self.nvar == 1):
            return [i,]
        N = self.splines[1].getNcp()
        if(self.nvar == 2):
            dofs = []
            if(direction==0):
                for j in range(0,N):
                    dofs += [ij2dof(i,j,M),]
            elif(direction==1):
                for j in range(0,M):
                    dofs += [ij2dof(j,i,M),]
            return dofs
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
                
class ExplicitBSplineControlMesh(AbstractControlMesh):

    def __init__(self,degrees,kvecs,extraDim=0,useRect=USE_RECT_ELEM_DEFAULT):
        self.scalarSpline = BSpline(degrees,kvecs,useRect)
        # parametric = physical
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

            # use Greville abscjklfd for explicit spline
            coord = self.scalarSpline.splines[direction].greville\
                    (directionalIndex)
        else:
            coord = 0.0
        return coord

    def getNsd(self):
        return self.nsd
