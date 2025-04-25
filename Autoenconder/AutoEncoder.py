import numpy as np
import torch as th

class MarketAutoencoder(th.nn.Module):
    def __init__(self):
        self.dim = 40
        self.compress = 3
        self.learning_rate = 1e-4
        super().__init__()

        self.encoder = None
        self.decoder = None
        self.setup()
        self.double()
        
    def setup(self):
        self.encoder = th.nn.Sequential(    # Probably way overkill
            th.nn.Linear(self.dim, self.dim),
            th.nn.LeakyReLU(),
            th.nn.Linear(self.dim, self.dim*2),
            th.nn.LeakyReLU(),
            th.nn.Linear(self.dim*2, 36),
            th.nn.LeakyReLU(),
            th.nn.Linear(36,18),
            th.nn.LeakyReLU(),
            th.nn.Linear(18, 15),
            th.nn.LeakyReLU(),
            th.nn.Linear(15, 12),
            th.nn.LeakyReLU(),
            th.nn.Linear(12, 9),
            th.nn.Tanh(),
            th.nn.Linear(9,self.compress),
            th.nn.Tanh()
        )

        self.decoder = th.nn.Sequential(    # Probably way overkill
            th.nn.Linear(self.compress, 9),
            th.nn.LeakyReLU(),
            th.nn.Linear(9, 12),
            th.nn.LeakyReLU(),
            th.nn.Linear(12,15),
            th.nn.LeakyReLU(),
            th.nn.Linear(15, 18),
            th.nn.LeakyReLU(),
            th.nn.Linear(18, 36),
            th.nn.LeakyReLU(),
            th.nn.Linear(36, self.dim*2),
            th.nn.LeakyReLU(),
            th.nn.Linear(self.dim*2, self.dim),
            th.nn.LeakyReLU(),
            th.nn.Linear(self.dim,self.dim),
            th.nn.LeakyReLU()
        )
    

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def preprocess(self, sample):
        sample[0:20] = np.log(sample[0:20] + 0.01)
        sample[20:] = sample[20:]*2 - 1
        return sample
        
    def deprocess(self,sample):
        sample.detach().numpy()
        sample[0:20] = np.exp(sample[0:20]) - 0.01
        sample[20:] = (sample[20:] + 1)/2
        return sample
    
    def train(self, inputPaths):
        loss_fn = th.nn.HuberLoss()

        deGroupedSamples = []
        for path in inputPaths:
            for t in range(len(path.t_s)):
                if path.t_s[t] < 5:
                    Swaptions = [0.0] + [path.Swaptions[i][t] for i in range(0,19)]
                    Q_s = [path.Q_s[i][t] for i in range(1,21)]
                    sample = np.concat([Swaptions,Q_s])
                    sample = self.preprocess(sample)
                    deGroupedSamples.append(sample)
            
        training_loader = th.utils.data.DataLoader(deGroupedSamples, batch_size = 125, shuffle=True)

        running_loss = 0
        optimizer = th.optim.Adam(self.parameters(), lr = 0.0025)
        
        n_epochs = 500
        for epoch in range(0,n_epochs):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] - 0.0025*(1/n_epochs)
            for i, data in enumerate(training_loader):
                inputs = data
                labels = data
                #print(inputs)
                
                # Zero Grads
                optimizer.zero_grad()

                # Predict
                outputs = self(inputs)

                # Get Loss and grads
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust weights
                optimizer.step()

                # Report training progress
                running_loss += loss.item()
                if i % 2000 == 999:
                    last_loss = running_loss/(i%2000 + 1) 
                    print('Epoch{}, Batch {} loss: {}'.format(epoch+1,i+1,last_loss))
                    running_loss = 0

    # Right now unused but in future case if you regenerate autoencoder using this in the dev_env instead of forward ?might? be faster
    @th.no_grad
    def compress(self,sample):
        sample = sample.copy()
        sample = self.preprocess(sample)
        z = self.encoder(sample)
        return z