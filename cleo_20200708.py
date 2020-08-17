import random
import json
import time

import numpy as np
from scipy import optimize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class CLEO:

    def __init__(self, param):
        self.param = param
        self.delta_max = param['delta_max']
        self.eta_1 = param['eta_1']
        self.eta_2 = param['eta_2']
        self.gamma = param['gamma']
        
        self.delta = param['delta_init']
        self.x = np.array(param['x_init'])
        self.x_new = np.array(param['x_init'])
        self.s = self.x_new - self.x
        self.obj_appro = 0.0
        self.obj_appro_h = 0.0

        self.niter = param['niter']
        self.stop_tol = param['stop_tol']
        self.status = 'accept'
        self.his = {'x': [], 'obj': [], 'delta': []}

    def demand_func(self, x):
        if type(x) == list or x.ndim == 1:
            tmp = np.exp(self.param['k'] - self.param['m'] * x[0])
        else:
            tmp = np.exp(self.param['k'] - self.param['m'] * x[:, 0])
        w = self.param['a'] * (tmp / (1 + tmp)) + self.param['c_demand']
        return w

    def data_generator(self, num_x=None):
        if num_x is None:
            num_x = int(self.param['num_x'])
        else:
            num_x = int(num_x)
        sample_x = np.random.uniform(low=self.param['x_min'], high=self.param['x_max'],
                                     size=(num_x, len(self.param['x_init'])))
        sample_w = self.demand_func(sample_x)
        return sample_x, sample_w

    def demand_func_appro(self, x, residual=None, mode='linear'):
        if mode == 'linear':
            return np.dot(self.k1, x) + self.k0
        elif mode == 'empirical':
            return np.dot(self.k1, x) + residual

    def manufact_cost_func(self, x):
        if type(x) == list or x.ndim == 1:
            return self.param['c_manu_1'] * (x[1] - self.param['c_manu_x']) ** 2 + self.param['c_manu_0']
        else:
            return self.param['c_manu_1'] * (x[:, 1] - self.param['c_manu_x']) ** 2 + self.param['c_manu_0']

    def obj_func_h(self, x, w):
        return - x[0] * w + self.param['c_add'] * np.maximum(w - x[1], 0) + \
               self.param['c_storage'] * np.maximum(x[1] - w, 0)

    def obj_func_w(self, x, w):
        return self.manufact_cost_func(x) * x[1] + self.obj_func_h(x, w)

    def obj_func(self, x, sample=None, mode='linear'):
        if mode == 'linear':
            w = self.demand_func_appro(x, mode='linear')
            obj = self.obj_func_w(x, w)
            return obj

        elif mode == 'empirical':
            c_manu = self.manufact_cost_func(x)
            sample_x, sample_w = sample
            residuals = sample_w - np.dot(sample_x, self.k1)
            w = self.demand_func_appro(x, residual=residuals, mode='empirical')
            obj = c_manu * x[1] + np.mean(self.obj_func_h(x, w))
            return obj

        elif mode == 'valid':
            w = self.demand_func(x)
            c_manu = self.manufact_cost_func(x)
            obj = c_manu * x[1] - x[0] * w + \
                  self.param['c_add'] * np.maximum(w - x[1], 0) + self.param['c_storage'] * np.maximum(x[1] - w, 0)

            cost = {'product': c_manu * x[1],
                    'add': self.param['c_add'] * np.maximum(w - x[1], 0),
                    'storage':  self.param['c_storage'] * np.maximum(x[1] - w, 0)}
            cost['a+s'] = cost['add'] + cost['storage']
            cost['total'] = cost['product'] + cost['add'] + cost['storage']
            return obj, w, - x[0] * w, cost

    def grad_func(self, x):
        pass

    ####

    def generate_sample(self, x=None, delta=None, mode='uniform'):
        if x is None:
            x = self.x
        if delta is None:
            delta = self.delta
        if mode == 'uniform':
            sample = x + np.random.uniform(low=-delta/2, high=delta/2,
                                           size=(int(np.ceil(self.param['sample_density'] * delta)) + 10, len(x)))
            idx_valid = np.where(np.sum((sample < self.param['x_min']) + (sample > self.param['x_max']), axis=1) == 0)[0]
            sample_valid = sample[idx_valid, :]
            return sample_valid
        else:
            raise ValueError('invalid sample mode, "uniform" only')

    def fit_lr(self, X, y):
        X_1 = np.hstack((X, np.ones([len(X), 1])))
        K= np.linalg.lstsq(X_1, y, rcond=None)[0]
        self.k1 = K[:-1]
        self.k0 = K[-1]

    def update(self):
        delta_s = np.linalg.norm(self.s) / 2
        sample_x = self.generate_sample(delta=delta_s, mode='uniform')
        sample_w = sample_w = self.demand_func(sample_x)
        sample_x_h = self.generate_sample(x=self.x_new, delta=delta_s, mode='uniform')
        sample_w_h = self.demand_func(sample_x_h)
        v = self.obj_func(self.x, sample=[sample_x, sample_w], mode='empirical')
        v_h = self.obj_func(self.x_new, sample=[sample_x_h, sample_w_h], mode='empirical')
        rho = (v_h - v) / (self.obj_appro_h - self.obj_appro)

        if rho > self.eta_1 and True:
            self.x = self.x_new
            self.delta = min(self.gamma * self.delta, self.delta_max)
            self.status = 'accept'
        else:
            self.delta = self.delta / self.gamma
            self.status = 'reject'

    def run(self, mode = 'empirical', verbose=True):
        for k in range(self.niter):
            self.his['x'].append(self.x)
            self.his['obj'].append(self.obj_func(self.x, mode='valid')[0])
            self.his['delta'].append(self.delta)

            sample_x = self.generate_sample(mode='uniform')
            sample_w = self.demand_func(sample_x)
            self.nsample = len(sample_x)
            self.fit_lr(sample_x, sample_w)

            # self.obj_func(self.x, sample=None, mode='linear')
            # self.obj_func(self.x, sample=[sample_x, sample_w], mode='empirical')

            self.x_max = np.min(np.vstack((self.x + self.delta/2, self.param['x_max'])), axis=0)
            self.x_min = np.max(np.vstack((self.x - self.delta/2, self.param['x_min'])), axis=0)

            x_bound = optimize.Bounds(self.x_min, self.x_max)
            optimizer = optimize.minimize(self.obj_func, self.x, args=([sample_x, sample_w], mode),
                                          method='L-BFGS-B', bounds=x_bound)
            self.x_new = optimizer.x
            self.s = self.x_new - self.x
            self.obj_appro = self.obj_func(self.x, sample=[sample_x, sample_w], mode=mode)
            self.obj_appro_h = self.obj_func(self.x_new, sample=[sample_x, sample_w], mode=mode)

            obj_gt, _, _, _ = self.obj_func(self.x, sample=None, mode='valid')
            obj_gt_new, _, _, _ = self.obj_func(self.x_new, sample=None, mode='valid')

            if verbose is True:
                print(self.x_new, obj_gt, obj_gt_new, self.status, self.delta)

            if self.obj_appro_h - self.obj_appro < -self.stop_tol:
                self.update()
            else:
                break

    ####

    def obj_plot(self):
        x = [np.linspace(0, 50, 50), np.linspace(0, 50, 50)]

        X_0, X_1 = np.meshgrid(x[0], x[1])
        obj, w, income, cost = self.obj_func([X_0, X_1], mode='valid')

        fig_cost = plt.figure(figsize=(12, 8))
        # ax = fig_cost.gca(projection='3d')
        ax = fig_cost.add_subplot(3, 2, 1, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, w, alpha=0.3,
                                      cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, w, zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Demand')

        ax = fig_cost.add_subplot(3, 2, 2, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, obj, alpha=0.3,
                                      cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, obj, zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Obj with min at ' +
                      str(np.unravel_index(obj.argmin(), obj.T.shape)) + ' ' + str(np.min(obj)))

        ax = fig_cost.add_subplot(3, 2, 3, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, income, alpha=0.3,
                                      cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, income, zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Income')

        ax = fig_cost.add_subplot(3, 2, 4, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, cost['total'], alpha=0.3,
                                      cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, cost['total'], zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Total Cost')

        ax = fig_cost.add_subplot(3, 2, 5, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, cost['product'], alpha=0.3,
                                    cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, cost['product'], zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Product Cost')

        ax = fig_cost.add_subplot(3, 2, 6, projection='3d')
        surf_cost = ax.plot_surface(X_0, X_1, cost['a+s'], alpha=0.3,
                                    cmap=cm.coolwarm, linewidth=2, antialiased=False)
        cset = ax.contourf(X_0, X_1, cost['a+s'], zdir='z', offset=np.min(w), cmap=cm.coolwarm)
        ax.set_xlabel('Price')
        ax.set_ylabel('Amount')
        ax.set_zlabel('Add and Storage Cost')

        plt.show()


def run_plot(param):
    cleo = CLEO(param)

    n_sample = 1e2
    sample_x, sample_w = cleo.data_generator(n_sample)
    x_init = sample_x[0, :]

    cleo.x = x_init
    cleo.run(mode='empirical', verbose=False)

    quad = LinearRegression(fit_intercept=False)
    poly = PolynomialFeatures(degree=1)
    tmp = poly.fit_transform(sample_x)
    quad.fit(tmp, sample_w)

    svr = SVR(gamma='scale', C=5e1, epsilon=0.2)
    svr.fit(sample_x, sample_w)

    obj = {}
    w = {}
    N = 50
    x = [np.linspace(0, N, N), np.linspace(0, N, N)]
    X_0, X_1 = np.meshgrid(x[0], x[1])
    obj['cleo'], w['cleo'], _, _ = cleo.obj_func([X_0, X_1], mode='valid')
    X = np.array([X_0, X_1]).reshape(2, -1).T
    w['quad'] = quad.predict(poly.fit_transform(X)).reshape(N, N)
    obj['quad'] = cleo.obj_func_w([X_0, X_1], w['quad'])
    w['svr'] = svr.predict(X).reshape(N, N)
    obj['svr'] = cleo.obj_func_w([X_0, X_1], w['svr'])

    fig_cost = plt.figure(figsize=(12, 4))
    # ax = fig_cost.gca(projection='3d')
    ax = fig_cost.add_subplot(1, 2, 1, projection='3d')
    surf_cost = ax.plot_surface(X_0, X_1, w['cleo'], alpha=0.5,
                                cmap=cm.PuBu, linewidth=2, antialiased=False)
    surf_cost = ax.plot_surface(X_0, X_1, w['quad'], alpha=0.5,
                                cmap=cm.Greens, linewidth=2, antialiased=False)
    surf_cost = ax.plot_surface(X_0, X_1, w['svr'], alpha=0.5,
                                cmap=cm.Reds, linewidth=2, antialiased=False)
    # cset = ax.contourf(X_0, X_1, w['cleo'], zdir='z', offset=np.min(w['cleo']), cmap=cm.coolwarm)
    ax.set_xlabel('Price')
    ax.set_ylabel('Amount')
    ax.set_zlabel('Demand')

    ax = fig_cost.add_subplot(1, 2, 2, projection='3d')
    surf_cost = ax.plot_surface(X_0, X_1, obj['cleo'], alpha=0.5,
                                cmap=cm.PuBu, linewidth=2, antialiased=False)
    surf_cost = ax.plot_surface(X_0, X_1, obj['quad'], alpha=0.5,
                                cmap=cm.Greens, linewidth=2, antialiased=False)
    surf_cost = ax.plot_surface(X_0, X_1, obj['svr'], alpha=0.5,
                                cmap=cm.Reds, linewidth=2, antialiased=False)
    # cset = ax.contourf(X_0, X_1, obj['cleo'], zdir='z', offset=np.min(w['cleo']), cmap=cm.coolwarm)
    ax.set_xlabel('Price')
    ax.set_ylabel('Amount')
    ax.set_zlabel('Obj with min of ' + str(np.min(obj['cleo'])))

    plt.show()


def run_experiment(param, outfile):
    model_types = ['quad', 'cleo', 'svr']
    x = {key: {} for key in model_types}
    obj = {key: {} for key in model_types}
    obj_mean = {key: {} for key in model_types}
    obj_std = {key: {} for key in model_types}

    x_hat = {key: {} for key in model_types}
    obj_hat = {key: {} for key in model_types}
    obj_hat_mean = {key: {} for key in model_types}
    obj_hat_std = {key: {} for key in model_types}

    exp_time = {key: {} for key in model_types}
    exp_time_mean = {key: {} for key in model_types}

    for n_sample in param['n_sample_set']:
        for key in model_types:
            x[key][str(n_sample)] = []
            obj[key][str(n_sample)] = []

            x_hat[key][str(n_sample)] = []
            obj_hat[key][str(n_sample)] = []

            exp_time[key][str(n_sample)] = []

        for idx_exp in range(param['num_exp']):

            print('n_sample:', n_sample, 'idx_exp:', idx_exp)
            cleo = CLEO(param)

            sample_x, sample_w = cleo.data_generator(n_sample)
            # x_init = sample_x[0, :]
            # cleo.x = x_init

            start = time.time()
            cleo.run(mode='empirical', verbose=False)
            end = time.time()
            exp_time['cleo'][str(n_sample)].append(end - start)

            x['cleo'][str(n_sample)].append(cleo.his['x'][-1])
            obj['cleo'][str(n_sample)].append(cleo.his['obj'][-1])

            start = time.time()

            quad = LinearRegression(fit_intercept=False)
            poly = PolynomialFeatures(degree=1)
            tmp = poly.fit_transform(sample_x)
            quad.fit(tmp, sample_w)
            func = lambda x: cleo.obj_func_w(x, quad.predict(poly.fit_transform(x.reshape(1, -1))))

            x_bound = optimize.Bounds(param['x_min'], param['x_max'])
            optimizer = optimize.minimize(func, sample_x[0:1, :], method='L-BFGS-B', bounds=x_bound)
            x['quad'][str(n_sample)].append(optimizer.x)
            obj['quad'][str(n_sample)].append(cleo.obj_func(optimizer.x, mode='valid')[0])

            end = time.time()
            exp_time['quad'][str(n_sample)].append(end - start)

            start = time.time()

            svr = SVR(gamma='scale', C=3e1, epsilon=0.2)
            svr.fit(sample_x, sample_w)
            func = lambda x: cleo.obj_func_w(x, svr.predict(x.reshape(1, -1)))
            x_bound = optimize.Bounds(param['x_min'], param['x_max'])
            optimizer = optimize.minimize(func, sample_x[0:1, :], method='L-BFGS-B', bounds=x_bound)
            x['svr'][str(n_sample)].append(optimizer.x)
            obj['svr'][str(n_sample)].append(cleo.obj_func(optimizer.x, mode='valid')[0])

            end = time.time()
            exp_time['svr'][str(n_sample)].append(end - start)

            N = 50
            X_0, X_1 = np.meshgrid(np.linspace(0, N, N), np.linspace(0, N, N))
            w = {}
            obj_mesh = {}
            obj_h, w['cleo'], _, _ = cleo.obj_func([X_0, X_1], mode='valid')
            obj_hat['cleo'][str(n_sample)].append(np.min(obj_h))

            X = np.array([X_0, X_1]).reshape(2, -1).T

            key = 'quad'
            w[key] = quad.predict(poly.fit_transform(X)).reshape(N, N)
            obj_mesh[key] = cleo.obj_func_w([X_0, X_1], w[key])
            x_h = np.array(np.unravel_index(obj_mesh[key].argmin(), obj_mesh[key].T.shape))[::-1]
            x_hat[key][str(n_sample)].append(x_h)
            obj_hat[key][str(n_sample)].append(cleo.obj_func(x_h, mode='valid')[0])

            key = 'svr'
            w[key] = svr.predict(X).reshape(N, N)
            obj_mesh[key] = cleo.obj_func_w([X_0, X_1], w[key])
            x_h = np.array(np.unravel_index(obj_mesh[key].argmin(), obj_mesh[key].shape))[::-1]
            x_hat[key][str(n_sample)].append(x_h)
            obj_hat[key][str(n_sample)].append(cleo.obj_func(x_h, mode='valid')[0])

        for key in model_types:
            obj_mean[key][str(n_sample)] = np.mean(obj[key][str(n_sample)])
            obj_std[key][str(n_sample)] = np.std(obj[key][str(n_sample)])

            obj_hat_mean[key][str(n_sample)] = np.mean(obj_hat[key][str(n_sample)])
            obj_hat_std[key][str(n_sample)] = np.std(obj_hat[key][str(n_sample)])

            exp_time_mean[key][str(n_sample)] = np.mean(exp_time[key][str(n_sample)])

    print(obj_mean)
    print(obj_std)
    print(obj_hat_mean)
    print(obj_hat_std)
    print('done')

    res = {'om': obj_mean, 'os': obj_std, 'ohm': obj_hat_mean, 'ohs': obj_hat_std, 'ts': exp_time_mean}
    # with open(outfile, 'w') as outfile:
    #     json.dump(res, outfile)
    # outfile.close()

    return res


def run_plot_curve(param, res_file):

    with open(res_file, 'r') as f:
        res = json.load(f)
    f.close()

    data = {}
    for k1 in ['om', 'os', 'ohm', 'ohs']:
        for k2 in ['quad', 'svr', 'cleo']:
            data[k1 + '_' + k2] = np.array([[float(item[0]), item[1]] for item in res[k1][k2].items()])

    plt.figure(figsize=(12, 8))
    for k1 in ['om', 'ohm']:
        k0 = k1[:-1] + 's'
        for k2 in ['quad', 'svr', 'cleo']:
            # plt.plot(data[k1 + '_' + k2][:, 0], data[k1 + '_' + k2][:, 1], label=k1+'_'+k2)
            plt.errorbar(data[k1 + '_' + k2][:, 0], data[k1 + '_' + k2][:, 1],
                         yerr=data[k0 + '_' + k2][:, 1], fmt='o-', elinewidth=2, capsize=4, label=k1+'_'+k2)
    plt.legend(loc="upper right")
    plt.show()


def run_plot_curve_2(param, res_file):

    with open(res_file, 'r') as f:
        res = json.load(f)
    f.close()

    data = {}
    for k1 in ['om', 'os', 'ohm', 'ohs']:
        for k2 in ['quad', 'svr', 'cleo']:
            data[k1 + '_' + k2] = np.array([[float(item[0]), item[1]] for item in res[k1][k2].items()])

    # plt.figure(figsize=(12, 8))
    # for k1 in ['om', 'ohm']:
    #     k0 = k1[:-1] + 's'
    #     for k2 in ['quad', 'svr', 'cleo']:
    #         # plt.plot(data[k1 + '_' + k2][:, 0], data[k1 + '_' + k2][:, 1], label=k1+'_'+k2)
    #         plt.errorbar(data[k1 + '_' + k2][:, 0], data[k1 + '_' + k2][:, 1],
    #                      yerr=data[k0 + '_' + k2][:, 1], fmt='o-', elinewidth=2, capsize=4, label=k1+'_'+k2)
    # plt.legend(loc="upper right")

    # data['om_'][:, 0]


    plt.figure(figsize=(8, 6))

    ratio = 3
    bias = 800

    plt.plot(data['ohm_cleo'][:, 0], data['ohm_cleo'][:, 1] + bias, '-', color='k', lw=2, label='Global Minimum')

    plt.errorbar(data['om_svr'][:, 0], data['om_svr'][:, 1] + bias,
                 yerr=data['os_svr'][:, 1] / ratio, fmt='o-', color='darkorange', elinewidth=2, capsize=4, label='GPR+LBFGS')
    plt.errorbar(data['ohm_svr'][:, 0], data['ohm_svr'][:, 1] + bias,
                 yerr=data['ohs_svr'][:, 1] / ratio, fmt='o-', color='forestgreen', elinewidth=2, capsize=4, label='GPR+Global Min')

    plt.errorbar(data['om_quad'][:, 0], data['om_quad'][:, 1] + bias,
                 yerr=data['os_quad'][:, 1]/ratio, fmt='o-', color='royalblue', elinewidth=2, capsize=4, label='QR+LBFGS')
    plt.errorbar(data['ohm_quad'][:, 0], data['ohm_quad'][:, 1] + bias,
                 yerr=data['ohs_quad'][:, 1] / ratio, fmt='o-', color='purple', elinewidth=2, capsize=4, label='QR+Global Min')

    plt.errorbar(data['om_cleo'][:, 0], data['om_cleo'][:, 1] + bias,
                 yerr=data['os_cleo'][:, 1] / ratio, fmt='o-', color='indianred', elinewidth=2, capsize=4, label='CLEO')

    plt.xlim([0, 510])
    plt.ylim([-2500, 900])
    plt.xlabel('Number of samples', fontsize=15)
    plt.ylabel('Objective function value', fontsize=15)
    plt.legend(loc="upper right", fontsize=12)
    # plt.title('test')

    # plt.savefig('curve_gpr_cleo_2.pdf')

    plt.show()


if __name__ == "__main__":

    random.seed(23)

    param = {
        # data generation
        'a': 4.7e2,
        'k': -1e0,
        'm': 1e-1,
        'c_demand': 2.5,

        # objective function
        'c_manu_1': 1e-2,
        'c_manu_0': 10,
        'c_manu_x': 55,

        'c_add': 2e1,
        'c_storage': 8e0,

        # CLEO model
        'x_init': [25, 25],
        'x_min': [0, 0],
        'x_max': [5e1, 5e1],
        'sample_density': 10,
        'delta_init': 4.0,
        'delta_max': 1.0,

        'eta_1': 0.99,
        'eta_2': 0.5,
        'gamma': 0.95,
        'niter': 10,
        'stop_tol': 1e-3,

        # experiment
        'num_x': 1e4,
        'num_exp': 10,
        # 'n_sample_set': [1e1, 2e1, 4e1, 1e2, 2e2, 3e2, 4e2, 6e2, 1e3],
        'n_sample_set': [1e1, 1e2, 1e3, 2e3, 3e3, 1e4]
    }

    outfile = 'res.json'
    res = run_experiment(param, outfile)
    # run_plot_curve(param, outfile)
    # run_plot_curve_2(param, outfile)

    # run_plot(param)
    # cleo = CLEO(param)
    # cleo.obj_plot()
