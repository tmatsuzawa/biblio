import numpy as np
import sys
import turbulence.mdata


'''This module gives the JHTDdata class for loading and handling Hopkins data chunks'''


class JHTDdata:
    """class for handling Johns Hopkins Turbulence Database data
    """
    def __init__(self, data, param, N=None):
        ''''''
        if N is None:
            N = len(data.keys())

        dim = data[data.keys()[0]].shape
        tup = tuple(dim[:-2]) + (N,)

        #   print(tup)

        self.t = np.zeros(N)
        self.x0 = np.zeros(N)
        self.y0 = np.zeros(N)
        self.z0 = np.zeros(N)

        self.Ux = np.zeros(tup)
        self.Uy = np.zeros(tup)
        self.Uz = np.zeros(tup)

        self.def_axis(data, param, dim)

        self.fx = 1
        self.ft = 1

        self.load_data(data, param, N=N)
        #   self.filename =
        self.Id = turbulence.mdata.Id.Id(S=None, typ='Numerics', who='JHTDB')

    def load_data(self, data, param, N=None):
        """Load the data for the dataset chunk"""
        keys = np.sort(data.keys())
        for i, key in enumerate(keys[:N]):
            self.t[i] = param[key]['t0']
            self.x0[i] = param[key]['x0']
            self.y0[i] = param[key]['y0']
            self.z0[i] = param[key]['z0']

            data_np = np.asarray(data[key])

            self.Ux[..., i] = data_np[..., 0, 2]
            self.Uy[..., i] = data_np[..., 0, 1]
            self.Uz[..., i] = data_np[..., 0, 0]

        indices = np.argsort(self.t)

        self.t = self.t[indices]
        self.x0 = self.x0[indices]
        self.y0 = self.y0[indices]
        self.z0 = self.x0[indices]

        self.Ux = self.Ux[..., indices]
        self.Uy = self.Uy[..., indices]
        self.Uz = self.Uz[..., indices]

    def def_axis(self, data, param, dim, d=2):
        key = data.keys()[0]
        # first dimension is x instead of z
        self.x = np.asarray([[[k for i in np.arange(param[key]['xl'])] for j in range(param[key]['yl'])] for k in
                             range(param[key]['zl'])])[..., 0]
        #   print(self.x)
        self.y = np.asarray([[[j for i in np.arange(param[key]['xl'])] for j in range(param[key]['yl'])] for k in
                             range(param[key]['zl'])])[..., 0]
        self.z = np.asarray([[[i for i in np.arange(param[key]['xl'])] for j in range(param[key]['yl'])] for k in
                             range(param[key]['zl'])])[..., 0]
        #   print(self.x.shape)
        #   print(self.y.shape)
        #   print(self.z.shape)
    #        self.x = np.ones(tuple(dim[:-2]))
    #        self.y = np.ones(tuple(dim[:-2]))
    #        self.z = np.ones(tuple(dim[:-2]))

    def shape(self):
        return self.Ux.shape

    #############################
    # Measurements ##############
    #############################
    def get(self, field, **kwargs):

        if field == 'U':
            # return both component in a vectorial format
            Ux = self.get('Ux')
            Uy = self.get('Uy')
            data = np.transpose(np.asarray([Ux, Uy]), (1, 2, 3, 0))
            return data
            # if (not hasattr(self,field)) or (compute):
            #       vgradient.compute(M,field,Dt=Dt_filt,**kwargs)
        if not hasattr(self, field):
            if 'Dt_filt' in kwargs and kwargs['Dt_filt'] > 1:
                print('Filtering of the data : irreversible')
            self.compute(field)
        #            setattr(self,field,)
        if hasattr(self, field):
            return getattr(self, field)
        else:
            return None

    def get_cut(self):
        print("JHTDdata.get_cut() has not been implemented")
        sys.exit()

    def compute(self, field, **kwargs):
        import turbulence.analysis.vgradient as vgradient
        return vgradient.compute(self, field, **kwargs)

    def measure(self, name, function, force=False, *args, **kwargs):
        if (not hasattr(self, name)) or force:
            print("Compute " + name)
            val = function(self, *args, **kwargs)
            setattr(self, name, val)
        else:
            print("Already computed")
