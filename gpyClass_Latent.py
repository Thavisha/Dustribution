import torch
import gpytorch
import pyro

from dens_integTorch import integ_allSource


"""
Class file holding the GP model we use

"""


class LatentDensityGPModel(gpytorch.models.ApproximateGP):

    
    #Define the GP RBF kernel, variational distribution and the GP strategy 
    def __init__(self, source_dists, source_l, source_b, l_bounds, b_bounds, d_bounds, threeDGrid_l, threeDGrid_b, l_ind, b_ind, d_ind,
                    inducing_points, train_gpu, name_prefix="density_gp_model"):
        """
        Sets up the gaussian process it self. e.g: kernel and it's hyperparameters, the input data and the training grid
        """

        #some parts of this are the same as our existing __init__(), but we also need to define some things for pyro to do variational inference
        #These are: inducting_points, variational_distribution, variational_strategy

        self.name_prefix = name_prefix

        #We input all the information needed by the dens_integ function here and convert them to tensors as required by torch
        self.train_gpu = train_gpu

        if train_gpu: #Run code on GPU

            self.source_dists = torch.Tensor(source_dists).double().cuda()
            self.source_l = torch.Tensor(source_l).double().cuda()
            self.source_b = torch.Tensor(source_b).double().cuda()
            self.l_bounds = torch.Tensor(l_bounds).double().cuda()
            self.b_bounds = torch.Tensor(b_bounds).double().cuda()
            self.d_bounds = torch.Tensor(d_bounds).double().cuda()
            self.threeDGrid_l = torch.Tensor(threeDGrid_l).double().cuda()
            self.threeDGrid_b = torch.Tensor(threeDGrid_b).double().cuda()
            self.l_ind = torch.from_numpy(l_ind).cuda() 
            self.b_ind = torch.from_numpy(b_ind).cuda() 
            self.d_ind = torch.from_numpy(d_ind).cuda() 

        else: #Run code on CPU

            self.source_dists = torch.Tensor(source_dists)
            self.source_l = torch.Tensor(source_l)
            self.source_b = torch.Tensor(source_b)
            self.l_bounds = torch.Tensor(l_bounds)
            self.b_bounds = torch.Tensor(b_bounds)
            self.d_bounds = torch.Tensor(d_bounds)
            self.threeDGrid_l = torch.Tensor(threeDGrid_l)
            self.threeDGrid_b = torch.Tensor(threeDGrid_b)
            self.l_ind = torch.from_numpy(l_ind)
            self.b_ind = torch.from_numpy(b_ind)
            self.d_ind = torch.from_numpy(d_ind)


        # Define the variational distribution and strategy of the GP
        # We initialize the inducing points to lie on a grid from 0 to T
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        #variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution) #Learn the inducing point locations
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True) #Keep IP locations set and do not learn them


        # Define model
        super().__init__(variational_strategy=variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3 ), 
                                                            lengthscale_constraint=gpytorch.constraints.Interval(lower_bound=10, upper_bound=50) #These bounds don't matter as they are not used in any way, only given as function requires some sort of input
                                                        )

    #Define mean and covariance modules 
    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x) 

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
    #Guide function
    def guide(self, x, y, yerr):

        function_dist = self.pyro_guide(x)

        # Use a plate to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    
    #Define the GP model and carry out comparison to input extinctions
    def model(self, x, y, yerr):
        
        """
        - This is the function that does the real work
        - First it draws samples from the GP Prior
        - Then it transforms them (in our case, the link function integrates over 10**samples for specific lines of sight)
        - Then it computes the likelihood. To first order, we can reuse a lot of the code we've already written, but some of it will have to be transformed to use pytorch instead of numpy.
        """

        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)
            
            #Convert latent function to density
            density_samples = 10**function_samples #This is where we force the positivity in density!
            
            #Now we integrate
            extinction_samples = integ_allSource(self.source_dists, self.source_l, self.source_b, self.l_bounds, self.b_bounds, self.d_bounds, 
                                                    self.threeDGrid_l, self.threeDGrid_b, self.l_ind, self.b_ind, self.d_ind, density_samples.T, self.train_gpu)

            #Calculate the log-likelihood
            return pyro.sample(self.name_prefix + ".y",
                                pyro.distributions.Normal(extinction_samples, yerr),  
                                obs=y
                                )

















