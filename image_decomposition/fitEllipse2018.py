"""
Ellipse Fitting Algorithm.

Original code from
http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

but notation changed to that of  www.wikipedia.com

Author: Fabricio Ferrari
Co-Author: Geferson Lucatelli
v1 @ 2018 -- Morfometryka Utils

"""


from __future__ import division
from pylab import *
import astropy.io.fits as pyfits
import numpy as np
from time import sleep
from numpy.linalg import eig, inv
from sys import argv

# import pyff
import numpy as np
import matplotlib.pyplot as plt

def fitEllipse2(x, y):
    ''' cf.
        NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES,
        Halir, Flusser.
    '''
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    D1 = mat(D[:, :3])
    D2 = mat(D[:, 3:])
    S1 = D1.T * D1
    S2 = D1.T * D2
    S3 = D2.T * D2
    C1 = zeros((3, 3))
    C1[2, 0] = 2
    C1[1, 1] = -1
    C1[0, 2] = 2
    C1 = mat(C1)
    M = C1.I * (S1 - S2 * S3.I * S2.T)
    E, V = eig(M)
    VV = asarray(V)
    # print ( VV[:,1]**2 -  4*VV[:,0] * VV[:,2] < 0)

    idx = where(E > 0)[0]
    if len(idx) == 0:
        idx = 0
    a1 = V[:, idx]
    a2 = - S3.I * S2.T * a1

    fit = ravel((a1, a2))
    A, B, C, D, E, F = fit
    Delta = ((A * C - B ** 2) * F + B * E * D / 4 - C * D ** 2 / 4 - A * E ** 2 / 4)
    if C * Delta >= 0:
        print('non real ellipse')
    return fit


#
# http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
#
# also see http://mathworld.wolfram.com/Ellipse.html

def ellipse_center(fit):
    A, B, C, D, E, F = fit
    num1 = B ** 2 - 4 * A * C
    x0 = (2 * C * D - B * E) / num1
    y0 = (2 * A * E - B * D) / num1
    return np.array([x0, y0])


def ellipse_angle(fit):
    A, B, C, D, E, F = fit

    PA = (0.5 * arctan2(B, A - C))
    # if PA<0: PA=PA+pi/2.
    return PA


def ellipse_axis_length(fit):
    A, B, C, D, E, F = fit

    Aq = mat([[A, B / 2., D / 2.], [B / 2., C, E / 2.], [D / 2., E / 2., F]])
    A33 = Aq[:2, :2]

    lam1, lam2 = eigvals(A33)
    a = sqrt(abs(det(Aq) / (lam1 * det(A33))))
    b = sqrt(abs(det(Aq) / (lam2 * det(A33))))
    return array([max(a, b), min(a, b)])


def ellipse_axis_length_orig(fit):
    b, c, d, f, g, a = fit[1] / 2, fit[2], fit[3] / 2, fit[4] / 2, fit[5], fit[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def fitted_params(fit):
    'returns x0, y0, PA, a, b'
    x0, y0 = ellipse_center(fit)
    PA = ellipse_angle(fit)
    a, b = ellipse_axis_length(fit)
    return x0, y0, PA, a, b


def fitted_ellipse_points(fit, N=100):
    ''' returns N points in the fitted ellipse
    Useful for plotting purposes '''

    R = linspace(0, 2 * np.pi, N)

    x0, y0, PA, a, b = fitted_params(fit)

    xe = x0 + a * cos(R) * cos(PA) - b * sin(R) * sin(PA)
    ye = y0 + a * cos(R) * sin(PA) + b * sin(R) * cos(PA)
    return xe, ye


def main_test2(g,Isequence = None,region_split=None,SAVENAME=None):
    # g = pyfits.getdata(name)

    dxc = int(g.shape[0] * 0.25 * 0.5)
    dyc = int(g.shape[1] * 0.25 * 0.5)
    dc = int(g.shape[0] / 2)
    print(dxc, dc, dc - dxc)
    gal = g  # [dc-dyc:dc+dyc,dc-dxc:dc+dxc]

    M, N = gal.shape

    # levels are almost equally spaced in log(log(I)) for a Sersic profile
    R = linspace(0, 2 * np.pi, 100)
    if Isequence is None:
        Isteps = 100
        Imin = (0.1 * np.std(gal))
        Imax = 0.99 * ((gal.max()))
        Isequence = (np.geomspace(Imax, Imin, Isteps))


    Iellipses = zeros_like(gal)

    gal0 = gal.copy()
    II = np.zeros_like((Isequence))
    AA = np.zeros_like((Isequence))
    BB = np.zeros_like((Isequence))
    PPAA = np.zeros_like((Isequence))
    PHI = np.zeros_like((Isequence))
    XX0 = np.zeros_like((Isequence))
    YY0 = np.zeros_like((Isequence))
    XE = np.zeros_like((Isequence))
    YE = np.zeros_like((Isequence))
    for k in range(len(Isequence)):
        I = Isequence[k]
        try:
            delta = 0.5 * I
            y, x = where((gal > I - delta) & (gal < I + delta))

            # remove outliers
            remove_indexes = np.where(np.logical_or(abs(y - y.mean()) >
                                                    abs(y - y.mean()).mean() * 1.5, \
                                                    abs(x - x.mean()) >
                                                    abs(x - x.mean()).mean() * 1.5) == True)[0]
            # remove_indexes = np.where(np.logical_or(abs(y - np.std(y)) >
            #                                         np.mean(abs(y - np.std(y))) * 3.0, \
            #                                         abs(x - np.std(x)) >
            #                                         np.mean(abs(x - np.std(x))) * 3.0) == True)[0]

            x = np.delete(x, remove_indexes)
            y = np.delete(y, remove_indexes)

            if len(x) < 6:
                continue
            try:
                fit = fitEllipse2(x, y)
            except:
                continue
            x0, y0 = ellipse_center(fit)
            phi = ellipse_angle(fit)
            if phi < 0:
                phi = phi + np.pi
            a, b = ellipse_axis_length(fit)

            xe = x0 + a * cos(R) * cos(phi) - b * sin(R) * sin(phi)
            ye = y0 + a * cos(R) * sin(phi) + b * sin(R) * cos(phi)


            PA = np.rad2deg(phi)
            II[k] = I  # Note that II is not on the same size as Isequence.
            AA[k] = a
            BB[k] = b
            PPAA[k] = PA
            PHI[k] = phi
            XX0[k] = x0
            YY0[k] = y0
            XE = XE
            YE = YE


            Ifit = mean(gal[y, x])
            Iellipses[y, x] = Ifit
            # rr = np.sqrt(xe**2.0+ye**2.0)

            loopc = 0
            fit2 = array([0, 0, 0, 0, 0, 0])
            dpoints = 1

            while dpoints != 1 and loopc < 50:
                # print mean(fit-fit2)
                ### Distance matrix

                m = sqrt((x - x0) ** 2 / a ** 2 + (y - y0) ** 2 / b ** 2)
                mmean = m.mean()
                mstd = m.std()

                # Q based
                # idxin = where( qq<(qqmean+1.0*qqstd) )[0]
                # m based
                idxin = where((m > (mmean - 2.0 * mstd)) & (m < (mmean + 2.0 * mstd)))[0]
                if len(idxin) < 6:
                    break

                plt.plot(x, y, '.r', ms=4)
                plt.plot(x[idxin], y[idxin], '+b', ms=4)
                plt.plot(xe, ye, '-k', lw=2)
                plt.imshow(arcsinh(gal))
                plt.show()
                fit2 = fitEllipse2(x[idxin], y[idxin])
                xe2, ye2 = fitted_ellipse_points(fit2)
                plt.plot(xe2,ye2, '-b', lw=2)

                dpoints = abs(len(x) - len(x[idxin]))

                x = x[idxin]
                y = y[idxin]

                xlim(0, N)
                ylim(0, M)
                draw()
                loopc += 1

            print('ctr=(%8.2f %8.2f)     I=%8.2f     q=%8.2f     PA=%8.2f' % (x0, y0, Ifit, b / a, rad2deg(phi)))
        except:
            pass

    if region_split is None:
        rsplit = int(len(II) * 0.5)

    if region_split is not None:
        rsplit = region_split



    qmedian = np.nanmedian(BB / AA)
    PAmedian = np.nanmedian(PPAA)
    qmi = np.nanmedian((BB / AA)[:rsplit])
    qmo = np.nanmedian((BB / AA)[rsplit:])
    PAmi = np.nanmedian(PPAA[:rsplit])
    PAmo = np.nanmedian(PPAA[rsplit:])
    x0median = np.nanmedian(XX0)
    y0median = np.nanmedian(YY0)
    x0median_i = np.nanmedian(XX0[:rsplit])
    y0median_i = np.nanmedian(YY0[:rsplit])
    x0median_o = np.nanmedian(XX0[:rsplit])
    y0median_o = np.nanmedian(YY0[:rsplit])

    try:
        plt.figure()
        plt.imshow(np.log(gal), origin='lower')
        # plt.plot(XE, YE, '-b', lw=0.4,alpha=0.6)
        plt.scatter(XE[0:rsplit:5], YE[0:rsplit:5], s=0.3, color='blue')
        plt.scatter(XE[rsplit:-1:5], YE[rsplit:-1:5], s=0.3, color='red')
        if SAVENAME is not None:
            plt.savefig(SAVENAME, dpi=300, bbox_inches='tight')
        plt.clf()
    except:
        pass
    return (qmi, qmo, PAmi, PAmo, qmedian, PAmedian,
            x0median,y0median,x0median_i,y0median_i,x0median_o,y0median_o)

# if __name__ == '__main__':
#     main_test2()
