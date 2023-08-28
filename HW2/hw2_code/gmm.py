import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        max_vals = np.amax(logit, axis = 1)
        max_vals = max_vals.flatten()
        max_vals = max_vals[:, np.newaxis]
        diff = np.subtract(logit, max_vals)
        sum_exp = np.sum(np.exp(diff), axis = 1, keepdims = True)
        sum_exp = sum_exp.flatten()
        sum_exp = sum_exp[:, np.newaxis]
        prob = np.exp(diff) / sum_exp
        return prob

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        max_vals = np.amax(logit, axis = 1)
        max_vals = max_vals.flatten()
        max_vals = max_vals[:, np.newaxis]
        logit_new = np.subtract(logit, max_vals)
        sum_exp = np.log(np.sum(np.exp(logit_new), axis = 1, keepdims = True))
        sum_exp = sum_exp.reshape(sum_exp.shape[0], 1)
        s = np.add(sum_exp, max_vals)
        return s

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        var = np.diagonal(sigma_i)
        top = -(np.square(np.subtract(points, mu_i))) / (2 * var)
        num = np.exp(top)
        den = np.sqrt(2 * np.pi * var)
        pdf = np.prod(num / den, axis = 1)
        return pdf

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        raise NotImplementedError



    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed

        # pi = np.ones(self.K)/self.K
        # mu = self.points[np.random.choice(self.N, size=self.K, replace=True)]
        # sigma = []
        # for i in range(self.K):
        #     sigma.append(np.eye(self.D))

        # sigma = np.array(sigma)
        # return pi, mu, sigma

        pi = np.ones(self.K) / self.K
        N, D = self.N, self.D
        mu = self.points[np.random.uniform(0, N-1, self.K).astype(int)] 
        covar = np.eye(D)
        covar = covar.reshape((1, D, D))
        sigma = np.repeat(covar, self.K, axis = 0)
        return pi, mu, sigma


    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        if full_matrix is False:
            
            ll = np.ones((self.N, self.K))
            for k in range(self.K):
                pdf = self.normalPDF(self.points, mu[k], sigma[k])
                ll[:, k] = np.log(pi[k] + 1e-32) +  np.log(pdf + 1e-32)
            return ll

        

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        if full_matrix is False:
            return self.softmax(self._ll_joint(pi, mu, sigma, full_matrix))

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        if full_matrix is False:
            n_k = np.sum(gamma, axis=0)
            mu_new = np.dot(gamma.T, self.points) /  n_k[:, None]
            sigma_new = np.ones((self.K, self.D, self.D))
            for k in range(self.K):
                diff = self.points - mu_new[k, :]
                sigma_new[k] = np.diag((np.sum((gamma[:, k].reshape(self.N, 1) * diff).T* diff.T, axis=1)) / n_k[k])

            pi_new = n_k / self.N
            return pi_new, mu_new, sigma_new


    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

