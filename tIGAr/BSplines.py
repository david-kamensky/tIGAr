from tIGAr.common import *
#from numba import jit as numbaJit

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

# scalar univariate b-spline; this is used construct tensor products, with a
# univariate "tensor product" as a special case; therefore this does not
# implement the AbstractScalarBasis interface, even though you might think
# that it should.
class BSpline1(object):

    # create from degree, knot data
    def __init__(self,p,knots):
        self.p = p
        self.knots = knots
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

        
    def getNcp(self):
        return len(self.knots) - self.multiplicities[0]

    # given a parameter, return the knot span
    def getKnotSpan(self,u):

        # TODO: binary search

        # placeholder linear search
        #if(u<self.knots[0]-DOLFIN_EPS or u>self.knots[-1]+DOLFIN_EPS):
        #    print u, self.knots[0], self.knots[-1]
        #    return -1
        span = 0
        nspans = len(self.knots)-1
        for i in range(0,nspans):
            span = i
            if(u<self.knots[i+1]+DOLFIN_EPS):
                break
        #if(span < self.p):
        #    span = self.p
        #if(span > nspans-self.p-1):
        #    span = nspans-self.p-1 
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
        #for i in range(knotSpan-self.p,knotSpan+1):
        #    if(i<self.getNcp() and i>=0):
        #        nodes += [i,]
        #start = knotSpan-self.multiplicities[0]+1
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
        
        ndu[0,0] = 1.0
        for j in range(1,pl+1):
            left[j] = u - self.getKnot(i-j) #u_knotl[i-j]
            right[j] = self.getKnot(i+j-1)-u #u_knotl[i+j-1]-u
            saved = 0.0
            for r in range(0,j):
                ndu[j,r] = right[r+1] + left[j-r]
                temp = ndu[r,j-1]/ndu[j,r]
                ndu[r,j] = saved + right[r+1]*temp
                saved = left[j-r]*temp
            ndu[j,j] = saved
        for j in range(0,pl+1):
            ders[j] = ndu[j,pl]

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

    def __init__(self,degrees,kvecs):
        self.nvar = len(degrees)
        if(self.nvar > 3 or self.nvar < 1):
            print("ERROR: Unsupported parametric dimension.")
            exit()
        self.splines = []
        for i in range(0,self.nvar):
            self.splines += [BSpline1(degrees[i],kvecs[i]),]

    #def getParametricDimension(self):
    #    return self.nvar

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
            mesh = RectangleMesh(Point(0.0,0.0),\
                                 Point(uspline.nel,vspline.nel),\
                                 uspline.nel,vspline.nel)
            x = mesh.coordinates()
            xbar = zeros((len(x),2))
            for i in range(0,len(x)):
                uknotIndex = int(round(x[i,0]))
                vknotIndex = int(round(x[i,1]))
                xbar[i,0] = uspline.uniqueKnots[uknotIndex]
                xbar[i,1] = vspline.uniqueKnots[vknotIndex]
            mesh.coordinates()[:] = xbar
            return mesh
        else:
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            mesh = BoxMesh(Point(0.0,0.0,0.0),\
                           Point(uspline.nel,vspline.nel,wspline.nel),\
                           uspline.nel,vspline.nel,wspline.nel)
            x = mesh.coordinates()
            xbar = zeros((len(x),3))
            for i in range(0,len(x)):
                uknotIndex = int(round(x[i,0]))
                vknotIndex = int(round(x[i,1]))
                wknotIndex = int(round(x[i,2]))
                xbar[i,0] = uspline.uniqueKnots[uknotIndex]
                xbar[i,1] = vspline.uniqueKnots[vknotIndex]
                xbar[i,2] = wspline.uniqueKnots[wknotIndex]
            mesh.coordinates()[:] = xbar
            return mesh
        
    def getNcp(self):
        prod = 1
        for i in range(0,self.nvar):
            prod *= self.splines[i].getNcp()
        return prod

    def getDegree(self):
        deg = 0
        for i in range(0,self.nvar):
            # for simplex elements; take max for rectangles
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

    def __init__(self,degrees,kvecs,extraDim=0):
        self.scalarSpline = BSpline(degrees,kvecs)
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
