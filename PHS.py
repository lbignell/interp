# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:17:25 2017

@author: Lindsey Bignell

This package contains a definition of a polyharmonic spline, used for interpolating multidimensional functions.
"""
import numpy as np

class polyharmonic_spline:
    '''
    Define a polyharmonic spline interpolator.
    The default basis is a thin plate spline, though a callable can be passed.
    '''
    def __init__(self, basisfn=None):
        if basisfn is None:
            self.basisfn = self._thinspline
        else:
            self.basisfn = basisfn
        return

    def _thinspline(self, x):
        if x != 0:
            return (x**2)*np.log(x)
        else:
            return 0

    def normfn(self, pt1, pt2):
        '''
        calculate the euclidian norm between two points of arbitrary dimension.
        Optionally add a sample to pass through the origin if there isn't one.
        '''
        thediff = np.subtract(pt1, pt2)
        return np.linalg.norm(thediff)

    def set_coeffs(self, data, passthruorigin=True):
        '''
        Set the polyharmonic spline coefficients.
        The data should be a Nsamples by (Ndimensions + 1) matrix.
        '''
        #Infer the dimension from the input:
        self.data = np.array(data)
        self.nsamples, n2 = np.shape(data)
        self.ndim = n2 - 1

        #infer whether data cointains an origin (so a trial with all shocks
        #equal to 0, passing through 0).
        hasorigin = not np.all([np.any(sample) for sample in data])

        if not hasorigin and passthruorigin:
            self.data = np.append(self.data, [[0]*(self.ndim+1)], axis=0)
            self.nsamples = self.nsamples+1

        self.samplepts = [self.data[i][:-1] for i in range(self.nsamples)]
        yvals = [self.data[i][-1] for i in range(self.nsamples)]

        #Construct the matricies A and V.
        norms = [[self.normfn(self.samplepts[j], self.samplepts[i])
                  for i in range(self.nsamples)] for j in range(self.nsamples)]
        vecbasis = np.vectorize(self.basisfn, [float])
        A = vecbasis(norms).tolist()
        V = [np.ones(self.nsamples).tolist()] +\
            np.transpose(self.samplepts).tolist()
        VT = np.transpose(V).tolist()

        VTrows, VTcols = np.shape(VT)
        MatTop = [A[i] + VT[i] for i in range(len(VT))]
        MatBott = [V[i] + np.zeros(VTcols).tolist() for i in range(len(V))]
        self.Mat = MatTop + MatBott

        self.RHS = yvals + np.zeros(self.ndim + 1).tolist()

        self.coeffs = np.linalg.solve(self.Mat, self.RHS)

        return

    def interp_val(self, x):
        '''
        Return the interpolated value.
        The coefficients must be set already using set_coeff.
        x is the new point in space.
        '''
        #Number of rows in coeffs = self.samplepts + self.ndim + 1
        #First self.samplepts are the basis function weights and the rest are
        #the linear part.
        #This code can probably stand to be sped up...
        basissum = sum([self.coeffs[i]*self.basisfn(self.normfn(x, pt))
                       for i,pt in enumerate(self.samplepts)])

        linearpart = np.dot(self.coeffs[self.nsamples:], [1] + list(x))

        return basissum + linearpart
