from tIGAr.common import *

# function to get something at an alpha level
def x_alpha(alpha, x, x_old):
    return Constant(alpha)*x + Constant(1.0-alpha)*x_old

# generalized alpha integration for first- or second-order systems of ODEs
class GeneralizedAlphaIntegrator:

    def __init__(self,RHO_INF,DELTA_T,x,oldFunctions,\
                 useFirstOrderAlphaM=False):
        self.RHO_INF = RHO_INF
        self.DELTA_T = DELTA_T
        # infer whether this is a first or second-order ODE system based on
        # the number of functions prescribed for the previous time step
        self.systemOrder = len(oldFunctions)-1
        # always use first-order alpha_m for first order systems, and
        # optionally use it for second-order systems (e.g., in yuri's fsi
        # paper)
        if(useFirstOrderAlphaM or self.systemOrder==1):
            self.ALPHA_M = 0.5*(3.0 - RHO_INF)/(1.0 + RHO_INF)
        else:
            self.ALPHA_M = (2.0 - RHO_INF)/(1.0 + RHO_INF)
        self.ALPHA_F = 1.0/(1.0 + RHO_INF)
        self.GAMMA = 0.5 + self.ALPHA_M - self.ALPHA_F
        self.BETA = 0.25*(1.0 + self.ALPHA_M - self.ALPHA_F)**2
        self.x = x
        self.x_old = oldFunctions[0]
        self.xdot_old = oldFunctions[1]
        if(self.systemOrder == 2):
            self.xddot_old = oldFunctions[2]

    def xdot(self):
        if(self.systemOrder == 1):
            return Constant(1.0/(self.GAMMA*self.DELTA_T))*self.x\
                + Constant(-1.0/(self.GAMMA*self.DELTA_T))*self.x_old\
                + Constant((self.GAMMA-1.0)/self.GAMMA)*self.xdot_old
        else:
            return Constant(self.GAMMA/(self.BETA*self.DELTA_T))*self.x\
                + Constant(-self.GAMMA/(self.BETA*self.DELTA_T))*self.x_old\
                + Constant(1.0 - self.GAMMA/self.BETA)*self.xdot_old\
                + Constant((1.0-self.GAMMA)*self.DELTA_T \
                           - (1.0 - 2.0*self.BETA)*self.DELTA_T\
                           *self.GAMMA/(2.0*self.BETA))*self.xddot_old

    def xddot(self):
        # should never be used for first-order systems
        return Constant(1.0/self.DELTA_T/self.GAMMA)*self.xdot()\
            + Constant(-1.0/self.DELTA_T/self.GAMMA)*self.xdot_old\
            + Constant(-(1.0-self.GAMMA)/self.GAMMA)*self.xddot_old

    def x_alpha(self):
        return x_alpha(self.ALPHA_F,self.x,self.x_old)
    
    def xdot_alpha(self):
        if(self.systemOrder == 1):
            alpha = self.ALPHA_M
        else:
            alpha = self.ALPHA_F
        return x_alpha(alpha,self.xdot(),self.xdot_old)
    
    def xddot_alpha(self):
        # should never be used for first-order systems
        return x_alpha(self.ALPHA_M,self.xddot(),self.xddot_old)

    # set current values to old ones to move to next time step
    def advance(self):
        # must make copies first, to avoid using updated values in
        # self.xdot(), etc., then assign self.xdot, etc., to re-assigned
        # copies
        x_old = Function(self.x.function_space())
        xdot_old = Function(self.x.function_space())
        x_old.assign(self.x)
        xdot_old.assign(self.xdot())
        if(self.systemOrder==2):
            xddot_old = Function(self.x.function_space())
            xddot_old.assign(self.xddot())
        self.x_old.assign(x_old)
        self.xdot_old.assign(xdot_old)
        if(self.systemOrder==2):
            self.xddot_old.assign(xddot_old)
