# -----------------------------------------------------------------------------
# Copyright (c) 2025, NeXpy Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING, distributed with this software.
# -----------------------------------------------------------------------------
import inspect
import re

import numpy as np
from lmfit import Model
from lmfit import __version__ as lmfit_version
from nexusformat.nexus import (NeXusError, NXdata, NXentry, NXfield, NXnote,
                               NXparameters, NXprocess)

from ..gui.utils import load_models


def get_models():
    """
    Return a dictionary of LMFIT models.

    This function returns a dictionary of LMFIT models, including those
    defined in the LMFIT package and those defined in the
    ``nexpy.models`` package. Additional models can also be defined in
    the ``~/.nexpy/models`` directory or in another installed package,
    which declares the entry point ``nexpy.models``. The models are
    returned as a dictionary where the keys are the names of the models
    and the values are the classes defining the models.
    """
    from lmfit.models import lmfit_models
    models = lmfit_models
    if 'Expression' in models:
        del models['Expression']
    if 'Gaussian-2D' in models:
        del models['Gaussian-2D']

    nexpy_models = load_models()

    for model in nexpy_models:
        try:
            models.update(
                dict((n.strip('Model'), m)
                for n, m in inspect.getmembers(nexpy_models[model],
                                               inspect.isclass)
                if issubclass(m, Model) and n != 'Model'))
        except ImportError:
            pass

    return models


all_models = get_models()


def get_methods():
    """Return a dictionary of minimization methods in LMFIT."""
    methods = {'leastsq': 'Levenberg-Marquardt',
               'least_squares': 'Least-Squares minimization, '
                                'using Trust Region Reflective method',
               'differential_evolution': 'differential evolution',
               'nelder': 'Nelder-Mead',
               'lbfgsb': ' L-BFGS-B',
               'powell': 'Powell',
               'cg': 'Conjugate-Gradient',
               'newton': 'Newton-CG',
               'cobyla': 'Cobyla',
               'bfgs': 'BFGS',
               'tnc': 'Truncated Newton',
               'trust-ncg': 'Newton-CG trust-region',
               'trust-exact': 'nearly exact trust-region',
               'trust-krylov': 'Newton GLTR trust-region',
               'trust-constr': 'trust-region for constrained optimization',
               'dogleg': 'Dog-leg trust-region',
               'slsqp': 'Sequential Linear Squares Programming'}
    return methods


all_methods = get_methods()


class NXFit:
    """Class defining the data, parameters, and results of a least-squares fit.

    Attributes
    ----------
    x : ndarray
        x-values of data.
    y : ndarray
        y-values of data.
    e : ndarray, optional
        standard deviations of the y-values.
    use_errors : bool
        set to True if the errors are to be used in the fit.
    data : NXdata
        NXdata group containing the signal and axis.
    models : list of models
        Models to be used in constructing the fit model.
    fit
        Results of the fit.
    """

    def __init__(self, data=None, models=None, use_errors=True):
        if ((isinstance(data, NXentry) or isinstance(data, NXprocess))
            and 'data' in data):
            group = data
            self.initialize_data(group['data'])
        elif isinstance(data, NXdata):
            self.initialize_data(data)
            group = None
        else:
            raise NeXusError("Must be an NXdata group")
        self.initialize_models(models)
        self.fit = None
        self.fitted = False
        self.use_errors = use_errors

    def __repr__(self):
        return f'NXFit({self.data.nxpath})'

    def initialize_data(self, data):
        """
        Initialize the data to be fitted.

        Parameters
        ----------
        data : NXdata
            The data to be fitted.

        Raises
        ------
        NeXusError
            If the data is not one-dimensional.
        """
        if isinstance(data, NXdata):
            if len(data.shape) > 1:
                raise NeXusError(
                    "Fitting only possible on one-dimensional arrays")
            self._data = NXdata()
            self._data['signal'] = data.nxsignal
            self._data.nxsignal = self._data['signal']
            if data.nxaxes[0].size == data.nxsignal.size + 1:
                self._data['axis'] = data.nxaxes[0].centers()
            elif data.nxaxes[0].size == data.nxsignal.size:
                self._data['axis'] = data.nxaxes[0]
            else:
                raise NeXusError("Data has invalid axes")
            self._data.nxaxes = [self._data['axis']]
            if data.nxerrors:
                self._data.nxerrors = data.nxerrors
                self.poisson_errors = False
            else:
                self.poisson_errors = True
            self._data['title'] = data.nxtitle
        else:
            raise NeXusError("Must be an NXdata group")

    def initialize_models(self, models):
        """Initialize the list of models."""
        self.all_models = all_models
        self.models = {}
        if models is not None:
            for model in models:
                if model in self.all_models:
                    self.add_model(model)
                else:
                    raise NeXusError(f"Model {model} not installed")

    @property
    def data(self):
        """The data to be fitted."""
        return self._data

    @property
    def signal(self):
        """
        The data to be fitted as a one-dimensional array.

        If the data is masked, the mask is removed before returning the
        data.
        """
        signal = self.data['signal']
        if signal.mask:
            return signal.nxdata.compressed().astype(np.float64)
        else:
            return signal.nxdata.astype(np.float64)

    @property
    def axis(self):
        """
        The x-axis values of the data to be fitted.

        If the data is masked, the mask is removed before returning the
        axis values.
        """
        data = self.data
        signal = data['signal'].nxdata
        axis = data['axis'].nxdata.astype(np.float64)
        if isinstance(signal, np.ma.MaskedArray):
            return np.ma.masked_array(axis, mask=signal.mask).compressed()
        else:
            return axis

    @property
    def errors(self):
        """
        The data errors as a one-dimensional array.

        If the data is masked, the mask is removed before returning the
        errors. If the data has no errors, returns None.
        """
        data = self.data
        if data.nxerrors:
            errors = data.nxerrors.nxdata.astype(np.float64)
            signal = data['signal'].nxdata
            if isinstance(signal, np.ma.MaskedArray):
                return np.ma.masked_array(
                    errors, mask=signal.mask).compressed()
            else:
                return errors
        else:
            return None

    @property
    def weights(self):
        """
        The data weights as a one-dimensional array.

        If the data is masked, the mask is removed before returning the
        weights. If the data has no errors, returns None.
        """
        if self.errors is not None and np.all(self.errors):
            return 1.0 / self.errors
        else:
            return None

    def add_model(self, model):
        """Add a model to the list of models."""
        if isinstance(model, Model):
            self.models[model._name] = model
        elif model.capitalize() in self.all_models:
            model = model.capitalize()
            self.models[model] = self.all_models[model]()
        else:
            raise NeXusError(f"Model {model} not installed")
        self.

    def get_model(self, x=None, m=None):
        """Returns the value of the model.

        Parameters
        ----------
        x : ndarray, optional
            x-values where the model is calculated. Defaults to `self.x`
        m : Model, optional
            Model to be included in the fit model. Defaults to all the
            models.

        Returns
        -------
        model : ndarray
            values of the model at the requested x-varlues.
        """
        if x is None:
            x = self.x
        model = np.zeros(x.shape, np.float64)
        if m:
            model = m.module.values(x, [p.value for p in m.parameters])
        else:
            for m in self.models:
                model += m.module.values(x, [p.value for p in m.parameters])
        return model

    def eval_model(self, composite_text):
        """
        Evaluates a composite model.

        Parameters
        ----------
        composite_text : str
            The composite model as a string.

        Returns
        -------
        model : lmfit.model.Model
            The evaluated composite model.
        """
        models = {m['name']: m['model'] for m in self.models}
        text = composite_text
        for m in models:
            text = text.replace(m, f"models['{m}']")
        try:
            return eval(text)
        except Exception as error:
            raise NeXusError(str(error))

    def parse_model_name(self, name):
        """
        Parse a model name.

        The model name is expected to be of the form
        <model_name>_<number>, where <model_name> is the name of the
        model and <number> is the number of the model. The _ is
        optional. The function returns a tuple (name, number), where
        name is the name of the model and number is the number of the
        model. If the model name does not match the expected format, the
        function returns (None, None).

        Parameters
        ----------
        name : str
            The model name to be parsed.

        Returns
        -------
        name : str
            The name of the model.
        number : str
            The number of the model.
        """
        match = re.match(r'([a-zA-Z0-9_-]*)_(\d*)$', name)
        if match:
            return match.group(1).replace('_', ' '), match.group(2)
        try:
            match = re.match(r'([a-zA-Z]*)(\d*)', name)
            return match.group(1), match.group(2)
        except Exception:
            return None, None

    def get_model_instance(self, model_class, model_name):
        """
        Returns an instance of the specified model class.

        Parameters
        ----------
        model_class : str
            The name of the model class.
        model_name : str
            The name of the model.

        Returns
        -------
        model : Model
            An instance of the specified model class.
        """
        if self.all_models[model_class].valid_forms:
            return self.all_models[model_class](prefix=model_name+'_',
                                                form=self.form_combo.selected)
        else:
            return self.all_models[model_class](prefix=model_name+'_')

    def load_fit(self, group):
        """
        Load a fit from a NeXus NXprocess group.

        Parameters
        ----------
        group : NXprocess
            The NeXus NXprocess group containing the fit.
        """
        self.model = None
        self.models = []
        if 'fit' in group.entries or 'model' in group.entries:
            for name in group.entries:
                if ('parameters' in group[name] and
                        'model' in group[name]['parameters'].attrs):
                    model_class = group[name]['parameters'].attrs['model']
                    model_name = name
                else:
                    model_class, model_index = self.parse_model_name(name)
                    model_name = model_class + '_' + model_index
                    if (model_class and model_class not in self.all_models and
                            model_class+'Model' in self.all_models):
                        model_class = model_class + 'Model'
                if model_class in self.all_models:
                    model = self.get_model_instance(model_class, model_name)
                    parameters = model.make_params()
                    saved_parameters = group[name]['parameters']
                    for mp in parameters:
                        p = mp.replace(model.prefix, '')
                        if p in saved_parameters:
                            parameter = parameters[mp]
                            parameter.value = saved_parameters[p].nxvalue
                            parameter.min = float(
                                saved_parameters[p].attrs['min'])
                            parameter.max = float(
                                saved_parameters[p].attrs['max'])
                            if 'vary' in saved_parameters[p].attrs:
                                parameter.vary = (
                                    saved_parameters[p].attrs['vary'])
                            if 'expr' in saved_parameters[p].attrs:
                                parameter.expr = (
                                    saved_parameters[p].attrs['expr'])
                            else:
                                parameter.expr = None
                            if 'error' in saved_parameters[p].attrs:
                                error = saved_parameters[p].attrs['error']
                                if error:
                                    parameter.stderr = float(
                                        saved_parameters[p].attrs['error'])
                    self.models.append({'name': model_name,
                                        'class': model_class,
                                        'model': model,
                                        'parameters': parameters})
            self.parameters = self.parameters

            def idx(model):
                return int(re.match('.*?([0-9]+)$', model['name']).group(1))
            self.models = sorted(self.models, key=idx)
            for model_index, model in enumerate(self.models):
                if model_index == 0:
                    self.model = model['model']
                    self.composite_model = model['name']
                else:
                    self.model += model['model']
                    self.composite_model += '+' + model['name']
            if 'composite_model' in group.attrs:
                composite_model = group.attrs['composite_model']
            elif 'model' in group and isinstance(group['model'], NXfield):
                composite_model = group['model'].nxvalue
            else:
                composite_model = None
            if composite_model is not None:
                self.model = self.eval_model(composite_model)
                self.composite_model = composite_model
            else:
                self.model = self.eval_model(self.composite_model)

    def fit_data(self):
        """Fit the data with the current model and parameters."""
        self.fit = self.model.fit(self.signal,
                                  params=self.parameters,
                                  weights=self.weights,
                                  x=self.axis,
                                  method=self.method,
                                  nan_policy='omit')
        if self.fit:
            self.parameters = self.fit.params
            self.fitted = True
        else:
            self.fitted = False

    def save_fit(self):
        """Save the results of a fit in a NXprocess group."""
        if self.fit is None:
            raise NeXusError('Fit not available for saving')
        group = NXprocess()
        group.attrs['composite_model'] = self.composite_model
        group['data'] = self.data
        for m in self.models:
            group[m['name']] = self.get_model(m['model'])
            parameters = NXparameters(attrs={'model': m['class']})
            for name in m['parameters']:
                p = self.fit.params[name]
                name = name.replace(m['model'].prefix, '')
                parameters[name] = NXfield(p.value, error=p.stderr,
                                           initial_value=p.init_value,
                                           min=str(p.min), max=str(p.max),
                                           vary=p.vary, expr=p.expr)
            group[m['name']].insert(parameters)
        group['program'] = 'lmfit'
        group['program'].attrs['version'] = lmfit_version
        group['title'] = 'Fit Results'
        group['fit'] = self.get_model(fit=True)
        fit = NXparameters()
        fit.nfev = self.fit.result.nfev
        fit.chisq = self.fit.result.chisqr
        fit.redchi = self.fit.result.redchi
        fit.message = self.fit.result.message
        group['statistics'] = fit
        group.note = NXnote(
            self.fit.result.message,
            f'Chi^2 = {self.fit.result.chisqr}\n'
            f'Reduced Chi^2 = {self.fit.result.redchi}\n'
            f'No. of Function Evaluations = {self.fit.result.nfev}\n'
            f'No. of Variables = {self.fit.result.nvarys}\n'
            f'No. of Data Points = {self.fit.result.ndata}\n'
            f'No. of Degrees of Freedom = {self.fit.result.nfree}\n'
            f'{self.fit.fit_report()}')
        return group
