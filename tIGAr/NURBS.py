# module to read in a previously-written PetIGA object from igakit, and
# generate a NURBS control mesh from it

from tIGAr.common import *
from tIGAr.BSplines import *
from igakit.io import PetIGA

class NURBSControlMesh(AbstractControlMesh):

    def __init__(self,fname,useRect=USE_RECT_ELEM_DEFAULT):

        # get an igakit nurbs object from the file
        ikNURBS = PetIGA().read(fname)

        # create a BSpline scalar space given the knot vector(s)
        self.scalarSpline = BSpline(ikNURBS.degree,ikNURBS.knots,useRect)
        
        # get the control net; already in homogeneous form
        nvar = len(ikNURBS.degree)
        if(nvar==1):
            self.bnet = ikNURBS.control
        elif(nvar==2):
            M = ikNURBS.control.shape[0]
            N = ikNURBS.control.shape[1]
            dim = ikNURBS.control.shape[2]
            self.bnet = zeros((M*N,dim))
            for j in range(0,N):
                for i in range(0,M):
                    self.bnet[ij2dof(i,j,M),:]\
                        = ikNURBS.control[i,j,:]
        else:
            M = ikNURBS.control.shape[0]
            N = ikNURBS.control.shape[1]
            O = ikNURBS.control.shape[2]
            dim = ikNURBS.control.shape[3]
            self.bnet = zeros((M*N*O,dim))
            for k in range(0,O):
                for j in range(0,N):
                    for i in range(0,M):
                        self.bnet[ijk2dof(i,j,k,M,N),:]\
                            = ikNURBS.control[i,j,k,:]
            
    def getScalarSpline(self):
        return self.scalarSpline

    def getHomogeneousCoordinate(self,node,direction):
        return self.bnet[node,direction]

    def getNsd(self):
        return self.bnet.shape[1]-1
