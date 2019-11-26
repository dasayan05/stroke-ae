import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from beziermatrix import bezier_matrix

def bij(t, i, n):
    # binomial coefficients
    return comb(n, i) * (t ** i) * ((1-t) ** (n-i))

def draw_bezier(ctrlPoints, nCtrlPoints = 0, nPointsCurve = 100, annotate = True, return_curve=False,
    ctrlPointPlotKwargs = dict(marker='X', color='r', linestyle='--'), curvePlotKwagrs = dict(color='g'),
    draw_axis = plt):
    '''
    Draws a Bezier curve with given control points

    ctrlPoints: shape (n+1, 2) matrix containing all control points
    nCtrlPoints: No. of control points. If 0, infered from 'ctrlPoints', otherwise consideres first 'nCtrlPoints' points from 'ctrlPoints'
    nPointsCurve: granularity of the Bezier curve
    return_curve: returns the points on the curve rather than drawing them

    ctrlPointPlotKwargs: The **kwargs for control point's plot() function
    curvePlotKwagrs: The **kwargs for curve's plot() function
    '''

    def T(ts: 'time points', d: 'degree'):
      # 'ts' is a vector (np.array) of time points
      ts = ts[..., np.newaxis]
      Q = tuple(ts**n for n in range(d, -1, -1))
      return np.concatenate(Q, 1)

    if nCtrlPoints == 0:
        # Infer the no. of control points
        nCtrlPoints, _ = ctrlPoints.shape
    else:
        # If given, pick first `nCtrlPoints` control points
        ctrlPoints = ctrlPoints[0:nCtrlPoints, :]

    # curve = np.zeros((nPointsCurve, 2))
    # for step, t in enumerate(np.linspace(0.0, 1.0, num = nPointsCurve)):
    #   s = np.zeros_like(ctrlPoints[0]) # Basically [0., 0.]
    #   for pointID, point in enumerate(ctrlPoints):
    #      # 'point' has shape (2,)
    #      s += bij(t, pointID, nCtrlPoints-1) * point
    #   curve[step] = s
    ts = np.linspace(0., 1., num = nPointsCurve)
    curve = np.matmul(
      T(ts, nCtrlPoints - 1),
      np.matmul(
        bezier_matrix(nCtrlPoints-1),
        ctrlPoints
      )
    )

    if return_curve: # Return the points of the curve as 'np.array'
      return curve

    # Plot the curve
    draw_axis.plot(ctrlPoints[:,0], ctrlPoints[:,1], **ctrlPointPlotKwargs)
    for n, ctrlPoint in enumerate(ctrlPoints):
      if annotate:
         draw_axis.annotate(str(n), (ctrlPoint[0], ctrlPoint[1]), color=ctrlPointPlotKwargs['color'])

    draw_axis.plot(curve[:,0], curve[:,1], **curvePlotKwagrs)
    for n, curvePoint in enumerate(curve):
      if n % 10 == 0 and annotate:
         draw_axis.annotate(str(n), (curvePoint[0], curvePoint[1]), color=curvePlotKwagrs['color'])

if __name__ == '__main__':
    ## Sample usage of the 'draw_bezier()' function
    
    # few definitions
    degree = 4
    
    # random control points over [-30,30] range
    ctrlPoints = np.random.randint(-30, 30, (degree + 1, 2)).astype(np.float_)

    fig = plt.figure()
    draw_bezier(ctrlPoints, draw_axis=plt.gca())
    plt.show()