import torch
import vbll
import torch.nn as nn
from tqdm import tqdm
from nn_helpers import HapticDataset, HapDatasetFromTwoPos, HapOnePos
from vbllnet import RegNet
from torch.utils.data import DataLoader
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import numpy as np
from pyro.infer import Predictive

class BNN(PyroModule):
    def __init__(self, device, input_size=8, output_size=3, prior_scale=10.):
        super(BNN, self).__init__()
        # Define the layers
        self.pretrained = RegNet(input_size=8)
        self.pretrained.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/vbllnet_1pos_best_model.pth'))
        
        self.activation = nn.ReLU()
        self.fc1 = PyroModule[nn.Linear](input_size, 256)
        self.fc2 = PyroModule[nn.Linear](256, 128)
        self.fc3 = PyroModule[nn.Linear](128, 64)
        self.fc4 = PyroModule[nn.Linear](64, output_size)
        
        self.fc1.weight = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([256, input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([256]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([128, 256]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([128]).to_event(1))
        self.fc3.weight = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([64, 128]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([64]).to_event(1))
        self.fc4.weight = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([output_size, 64]).to_event(2))
        self.fc4.bias = PyroSample(dist.Normal(0., torch.tensor(prior_scale, device=device)).expand([output_size]).to_event(1))
        

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        mu = self.fc4(x)
        
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)).to(x.device)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma*sigma).to_event(1), obs=y)
        return mu


def BNN_pretrained(x, y = None):
    pretrained_model = RegNet(input_size=8, output_size=3)
    pretrained_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/RegNetOnePos_model.pth'))
    pretrained_model = pretrained_model.to(x.device)
    prior_scale = 0.01
    fc1_weight_prior = dist.Normal(pretrained_model.fc1.weight, torch.ones_like(pretrained_model.fc1.weight) * prior_scale)
    fc1_bias_prior = dist.Normal(pretrained_model.fc1.bias, torch.ones_like(pretrained_model.fc1.bias) * prior_scale)
    fc2_weight_prior = dist.Normal(pretrained_model.fc2.weight, torch.ones_like(pretrained_model.fc2.weight) * prior_scale)
    fc2_bias_prior = dist.Normal(pretrained_model.fc2.bias, torch.ones_like(pretrained_model.fc2.bias) * prior_scale)
    fc3_weight_prior = dist.Normal(pretrained_model.fc3.weight, torch.ones_like(pretrained_model.fc3.weight) * prior_scale)
    fc3_bias_prior = dist.Normal(pretrained_model.fc3.bias, torch.ones_like(pretrained_model.fc3.bias) * prior_scale)
    fc4_weight_prior = dist.Normal(pretrained_model.fc4.weight, torch.ones_like(pretrained_model.fc4.weight) * prior_scale)
    fc4_bias_prior = dist.Normal(pretrained_model.fc4.bias, torch.ones_like(pretrained_model.fc4.bias))
    fc1_weight = pyro.sample('fc1_weight', fc1_weight_prior)
    fc1_bias = pyro.sample('fc1_bias', fc1_bias_prior)
    fc2_weight = pyro.sample('fc2_weight', fc2_weight_prior)
    fc2_bias = pyro.sample('fc2_bias', fc2_bias_prior)
    fc3_weight = pyro.sample('fc3_weight', fc3_weight_prior)
    fc3_bias = pyro.sample('fc3_bias', fc3_bias_prior)
    fc4_weight = pyro.sample('fc4_weight', fc4_weight_prior)
    fc4_bias = pyro.sample('fc4_bias', fc4_bias_prior)
    

    # Manually apply the sampled weights and biases
    logits = torch.relu(torch.matmul(x, fc1_weight.t()) + fc1_bias)
    logits = torch.relu(torch.matmul(logits, fc2_weight.t()) + fc2_bias)
    logits = torch.matmul(logits, fc3_weight.t()) + fc3_bias
    logits = torch.relu(logits)
    logits = torch.matmul(logits, fc4_weight.t()) + fc4_bias
    
    # print(logits[0])
    # print(y[0])

    # Condition on the observed datas
    sigma = pyro.sample("sigma", dist.HalfCauchy(2)).to(x.device)
    with pyro.plate('data', x.size(0)):
        # Likelihood of the observed data
        pyro.sample('obs', dist.Normal(logits, sigma*sigma).to_event(1), obs=y)
    
def train_bnn(train_loader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BNN(device).to(device)
    model = BNN_pretrained
    NUTS = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(NUTS, num_samples=1000, warmup_steps=200)
    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)
    mcmc.run(x, y)
    
    posterior_samples = mcmc.get_samples()
    torch.save(posterior_samples, '/media/okemo/extraHDD31/samueljin/Model/bnn3_best_model.pth')
    
    from pyro.infer import Predictive
    # model.eval()
    Total_error = [0,0,0]
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    for i, (x, y) in enumerate(train_loader):
        if i == 1:
            break
    x = x.to(device)
    y = y.to(device)
    samples = predictive(x)
    mean = samples['obs'].mean(0)
    std = samples['obs'].std(0)
    for i in range(len(x)):
        error = torch.abs(mean[i] - y[i]).cpu().detach().numpy()
        Total_error += error
        if i % 100 == 0:
            print(f"Mean: {mean[i]}")
            print(f"error: {error}")
            print(f"Std: {std[i]}")
    

    print(f"Average error: {Total_error/len(x)}")    
    # import arviz as az    
    # inference_data = az.from_pyro(mcmc)
    # ess = az.ess(inference_data)
    # print("Effective Sample Size (ESS):")
    # print(ess)
def eval_and_graph(data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load('/media/okemo/extraHDD31/samueljin/Model/bnn3_best_model.pth')
    
    deterministic_model = RegNet(input_size=8, output_size=3)
    deterministic_model.fc1.weight = nn.parameter.Parameter(weights['fc1_weight'].mean(0))
    deterministic_model.fc1.bias = nn.parameter.Parameter(weights['fc1_bias'].mean(0))
    deterministic_model.fc2.weight = nn.parameter.Parameter(weights['fc2_weight'].mean(0))
    deterministic_model.fc2.bias = nn.parameter.Parameter(weights['fc2_bias'].mean(0))
    deterministic_model.fc3.weight = nn.parameter.Parameter(weights['fc3_weight'].mean(0))  
    deterministic_model.fc3.bias = nn.parameter.Parameter(weights['fc3_bias'].mean(0))
    deterministic_model.fc4.weight = nn.parameter.Parameter(weights['fc4_weight'].mean(0))
    deterministic_model.fc4.bias = nn.parameter.Parameter(weights['fc4_bias'].mean(0))
    deterministic_model = deterministic_model.to(device)
    deterministic_model.eval()
    
    Pred = Predictive(model=BNN_pretrained, posterior_samples=weights)
    x_error,y_error,z_error = [],[],[]
    x_std, y_std, z_std = [],[],[]
    
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        outputs = Pred(x)
        mean = deterministic_model(x)
        std = outputs['obs'].std(0).cpu().detach().numpy()
        error = torch.abs(mean - y).cpu().detach().numpy()
        from sklearn.isotonic import IsotonicRegression

        # Apply isotonic regression for calibration
        # iso_reg = IsotonicRegression(out_of_bounds='clip')
        
        x_std = std[:,0]
        y_std = std[:,1]
        z_std = std[:,2]
        x_error = error[:,0]
        y_error = error[:,1]
        z_error = error[:,2]
        # x_std = iso_reg.fit_transform(x_std, x_error)
        # y_std = iso_reg.fit_transform(y_std, y_error)
        # z_std = iso_reg.fit_transform(z_std, z_error)
        
    import matplotlib.pyplot as plt
    #plot error against std
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(x_error, x_std)
    # plt.ylim(1.0, 2.5)
    plt.xlabel('Error')
    plt.ylabel('Std')
    
    plt.subplot(1,3,2)
    plt.scatter(y_error, y_std)
    # plt.ylim(1.0, 2.5)
    plt.xlabel('Error')
    plt.ylabel('Std')
    
    plt.subplot(1,3,3)
    plt.scatter(z_error, z_std)
    # plt.ylim(1.0, 2.5)
    plt.xlabel('Error')
    plt.ylabel('Std')
    
    plt.savefig('bnn_error_vs_std.png')
    print('Average error: ', np.mean(x_error), np.mean(y_error), np.mean(z_error))
    
        
def diagnoistic_bnn():
    import arviz as az
    posterior_samples = torch.load('/media/okemo/extraHDD31/samueljin/Model/bnn2_best_model.pth')
    
    
    # Convert the posterior samples to a format that ArviZ expects
    num_chains = 1  # Assuming you used a single chain for simplicity
    num_samples = 1000  # Number of samples after warmup
    posterior_dict = {}

    for k, v in posterior_samples.items():
        # Ensure the shape is (chains, draws, *shape)
        reshaped = v.view(num_chains, num_samples, *v.shape[1:])
        reshaped = reshaped.cpu().numpy()

        # Handle NaNs by replacing them with a value or removing
        reshaped = np.nan_to_num(reshaped)

        posterior_dict[k] = reshaped

    # Create the InferenceData object
    inference_data = az.from_dict(posterior=posterior_dict)
    # # Trace plots
    # az.plot_trace(inference_data)

    # # Autocorrelation plots
    # az.plot_autocorr(inference_data)

    # Effective Sample Size (ESS)
    ess = az.ess(inference_data)
    print("Effective Sample Size (ESS):")
    print(ess)
    # Extract individual ESS values for each parameter
    ess_dict = {}
    for var_name, var_data in ess.data_vars.items():
        ess_dict[var_name] = var_data.values

    print(ess_dict)
    import pandas as pd
    ess_df = pd.DataFrame({k: v.flatten() for k, v in ess_dict.items()}) 
    ess_df.to_csv('ess.csv')

    # # R-hat statistic
    # rhat = az.rhat(inference_data)
    # print("R-hat statistic:")
    # print(rhat)

    # # Summary of diagnostics
    # summary = az.summary(inference_data)
    # print(summary)

    
    
    
if __name__ == '__main__':
    
    # diagnoistic_bnn()
    train_loader = DataLoader(HapOnePos('/media/okemo/extraHDD31/samueljin/data2'), batch_size=19000, shuffle=True)
    # print(len(train_loader), len(test_loader))
    # train_bnn(train_loader)
    # bnn = BNN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # bnn.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/vbllnet_1pos_best_model.pth'))
    eval_and_graph(train_loader)