import numpy as np
import scipy.ndimage as nd
from astropy.constants import c
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table, Column
import warnings


# TODO: use unit stuff more now it is better developped
def vbyc(v):
    return (Quantity(v, unit=u.km/u.s)/c).to(1).value


def doppler(v):
    return 1.+vbyc(v)


def doppler_undo(v):
    return 1./doppler(v)

doppler_logw = vbyc


class Fit1(object):
    def __init__(self, w, f, e, mw, mf, npol, doppler):
        self.w = w
        self.f = f
        self.e = e
        self.mw = mw
        self.mf = mf
        self.npol = npol
        self.doppler = doppler
        # pre-calculate f/e and polynomial bases
        self.fbye = f/self.e
        self.wmean = w.mean()
        # Vp[:,0]=1, Vp[:,1]=w, .., Vp[:,npol]=w**npol
        self.Vp = np.polynomial.polynomial.polyvander(w/self.wmean-1., npol)
        self.rcond = len(f)*np.finfo(f.dtype).eps
        self.sol = None

    def __call__(self, v):
        mfi = np.interp(self.w, self.mw*self.doppler(v), self.mf)
        # V[:,0]=mfi/e, Vp[:,1]=mfi/e*w, .., Vp[:,npol]=mfi/e*w**npol
        V = self.Vp*(mfi/self.e)[:,np.newaxis]
        # normalizes different powers
        scl = np.sqrt((V*V).sum(0))
        sol, resids, rank, s = np.linalg.lstsq(V/scl, self.fbye, self.rcond)
        self.sol = (sol.T/scl).T
        if rank != self.npol:
            msg = "The fit may be poorly conditioned"
            warnings.warn(msg)
        fit = np.dot(V, self.sol)*self.e
        chi2 = np.sum(((self.f-fit)/self.e)**2)
        return chi2, fit, mfi


def fittable(obs, model, *args, **kwargs):
    return fit(obs['w'], obs['flux'], obs['err'], model['w'], model['flux'],
               *args, **kwargs)


def fit(w, f, e, mw, mf, vgrid, npol,
        sigrange=None, vrange=None, doppler=doppler, plot=False):

    vgrid = Quantity(vgrid, u.km/u.s)
    chi2 = Table([vgrid.value, np.zeros_like(vgrid.value)], names=['v','chi2'])
    chi2['v'].units = vgrid.unit

    fit1 = Fit1(w, f, e, mw, mf, npol, doppler)

    chi2['chi2'] = np.array([fit1(v)[0] for v in vgrid])

    chi2.meta['ndata'] = len(f)
    chi2.meta['npar'] = npol+1+1
    chi2.meta['ndof'] = chi2.meta['ndata']-chi2.meta['npar']

    if plot:
        import matplotlib.pylab as plt
        plt.scatter(chi2['v'], chi2['chi2'])

    if vrange is None and sigrange is None or len(vgrid) < 3:
        ibest = chi2['chi2'].argmin()
        vbest, bestchi2 = chi2[ibest]
        chi2.meta['vbest'] = vbest
        chi2.meta['verr'] = 0.
        chi2.meta['bestchi2'] = bestchi2
    else:
        vbest, verr, bestchi2 = minchi2(chi2, vrange, sigrange, plot=plot)

    _, fit, mfi = fit1(vbest)
    chi2.meta['wmean'] = fit1.wmean
    chi2.meta['continuum'] = fit1.sol
    return chi2, fit, mfi


def minchi2(chi2, vrange=None, sigrange=None,
            fitcol='chi2fit', plot=False):
    assert vrange is not None or sigrange is not None
    if sigrange is None:
        sigrange = 1e10
    if vrange is None:
        vrange = 1e10

    iminchi2 = chi2['chi2'].argmin()
    ndof = float(chi2.meta['ndof'])
    iok = np.where((chi2['chi2'] <
                    chi2['chi2'][iminchi2]*(1.+sigrange**2/ndof)) &
                   (abs(chi2['v']-chi2['v'][iminchi2]) <= vrange))

    p = np.polynomial.Polynomial.fit(chi2['v'][iok], chi2['chi2'][iok],
                                     2, domain=[])

    if plot:
        import matplotlib.pylab as plt
        plt.scatter(chi2['v'][iok], chi2['chi2'][iok], c='g')
        plt.plot(chi2['v'], p(chi2['v']))

    vbest = -p.coef[1]/2./p.coef[2]
    # normally, get sigma from where chi2 = chi2min+1, but best to scale
    # errors, so look for chi2 = chi2min*(1+1/ndof) ->
    # a verr**2 = chi2min/ndof -> verr = sqrt(chi2min/ndof/a)
    verr = np.sqrt(p(vbest)/p.coef[2]/ndof)
    chi2.meta['vbest'] = vbest
    chi2.meta['verr'] = verr
    chi2.meta['bestchi2'] = p(vbest)
    if fitcol is not None:
        if fitcol in chi2.colnames:
            chi2[fitcol] = p(chi2['v'])
        else:
            chi2.add_column(Column(data=p(chi2['v']), name=fitcol))

    return chi2.meta['vbest'], chi2.meta['verr'], chi2.meta['bestchi2']


def observe(model, wgrid, slit, seeing, overresolve, offset=0.):
    """Convolve a model with a seeing profile, truncated by a slit, & pixelate

    Parameters
    ----------
    model: Table (or dict-like)
       Holding wavelengths and fluxes in columns 'w', 'flux'
    wgrid: array
       Wavelength grid to interpolate model on
    slit: float
       Size of the slit in wavelength units
    seeing: float
       FWHM of the seeing disk in wavelength units
    overresolve: int
       Factor by which detector pixels are overresolved in the wavelength grid
    offset: float, optional
       Offset of the star in the slit in wavelength units (default 0.)

    Returns
    -------
    Convolved model: Table
       Holding wavelength grid and interpolated, convolved fluxes
       in columns 'w', 'flux'
    """
    # make filter
    wgridres = np.min(np.abs(np.diff(wgrid)))
    filthalfsize = np.round(slit/2./wgridres)
    filtgrid = np.arange(-filthalfsize,filthalfsize+1)*wgridres
    # sigma ~ seeing-fwhm/sqrt(8*ln(2.))
    filtsig = seeing/np.sqrt(8.*np.log(2.))
    filt = np.exp(-0.5*((filtgrid-offset)/filtsig)**2)
    filt /= filt.sum()
    # convolve with pixel width
    filtextra = int((overresolve-1)/2+0.5)
    filt = np.hstack((np.zeros(filtextra), filt, np.zeros(filtextra)))
    filt = nd.convolve1d(filt, np.ones(overresolve)/overresolve)
    mint = np.interp(wgrid, model['w'], model['flux'])
    mconv = nd.convolve1d(mint, filt)
    return Table([wgrid, mconv], names=('w','flux'), meta={'filt': filt})
