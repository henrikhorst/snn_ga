import torch
import powerlaw
import random
import snntorch as snn
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np

class SNN_GA:
    def __init__(self, population_size, num_neurons, num_steps, beta = 0.8, initial_mask_prob = 0.05) -> None:
        self.population_size = population_size
        self.num_neurons = num_neurons
        self.num_steps = num_steps
        self.num_inputs = num_neurons+1
        self.beta = beta
        self.mask = torch.bernoulli(initial_mask_prob*torch.ones((self.population_size, self.num_neurons, self.num_inputs)))
        self.generate_input()
    def generate_input(self):
        #generate constant input, here just 1
        self.spk_in = torch.ones((self.num_steps, 1, self.num_neurons))
        #print(f"Dimensions of spk_in: {spk_in.size()}")
        

    def mutate(self, masks, mutate_prob= 0.001):
        index=torch.bernoulli(mutate_prob*torch.ones_like(masks)).bool()
        mutation_elems= masks[index]
        mutation_elems = 1- mutation_elems
        masks[index]=mutation_elems
        return masks
    
    def avalanche(self, spk_data):
        ava_data=torch.sum(spk_data, dim=-1)
        avalanche_data = []
        for i in range(self.population_size):

            avalanche_sizes = []
            curr_avalanche = False
            for elem in ava_data[i]:
                if elem > 0:
                    if curr_avalanche == False:
                        avalanche_sizes.append(elem)
                        curr_avalanche = True
                    else:
                        avalanche_sizes[-1] += elem
                else:
                    curr_avalanche = False
            avalanche_data.append(avalanche_sizes)
        
        return avalanche_data

    def plot_single_powerlaw(self, datapoint):
        result = powerlaw.Fit(datapoint, discrete=True, xmin=1)
        print(result.power_law.alpha)
        print(result.power_law.xmin)
        print(result.power_law.sigma)
        print(result.power_law.xmax)
        result.plot_pdf(color='r', linestyle='--')
    
    def fitness(self, avalanche_data):
        sigmas=[]
        for i in range(len(avalanche_data)):
            result = powerlaw.Fit(avalanche_data[i], discrete=True, xmin=1)
            sigmas.append((i,result.power_law.sigma))
        s=[elem for elem in sigmas if elem[-1]>0]
        s=sorted(s, key=lambda x: x[-1])
        return s

    def mate(self, s, mask):
        new_masks= torch.zeros_like(mask)
        for i in range(int(0.1*self.population_size)):
            for j in range(self.population_size//10):
                
                index= i*10+j

                not_again1=[s[i][0]]
                not_again2=[s[i][0]]

                if j<self.population_size//100:
                    new_masks[index] = self.mutate(mask[s[i][0]], mutate_prob= 0.001)

                elif j<self.population_size//20 + 0.5*self.population_size//100:
                    new_masks[index, 0:int(self.num_neurons/2)]= mask[s[i][0], 0:int(self.num_neurons/2)]
                    
                    while True:
                        rand= random.choice(s)
                        if rand[0] not in not_again1:
                            not_again1.append(rand[0])
                            break
                    new_masks[index, int(self.num_neurons/2):]= mask[rand[0], int(self.num_neurons/2):]
                else:
                    new_masks[index, int(self.num_neurons/2):]= mask[s[i][0], int(self.num_neurons/2):]
                    while True:
                        rand= random.choice(s)
                        if rand[0] not in not_again2:
                            not_again2.append(rand[0])
                            break
                    new_masks[index, 0:int(self.num_neurons/2)]= mask[rand[0], 0:int(self.num_neurons/2)]
        
        return new_masks

    def simulate(self, new_masks):
        fcs, lifs = [], []
        for i in range(self.population_size):
            fcs.append([nn.Linear(self.num_inputs, 1, bias=False) for n in range(self.num_neurons)])
            lifs.append([snn.Leaky(beta=self.beta) for n in range(self.num_neurons)])

        with torch.no_grad():
            for elem in fcs:
                for layer in elem:
                    layer.weight = nn.Parameter(0.5*torch.ones_like(layer.weight))

        for i,elem in enumerate(fcs):
            for c,layer in enumerate(elem):
                prune.custom_from_mask(layer, name='weight', mask=new_masks[i,c].unsqueeze(dim =0))

        mem = torch.zeros(self.population_size, self.num_steps+1, self.num_neurons)
        spk = torch.zeros_like(mem)

        with torch.no_grad():
            # network simulation
            for j in range(self.population_size):

                for step in range(1,self.num_steps+1):
                    #print(step)
                    for i in range(self.num_neurons):
                    #print(i)
                        cur = fcs[j][i](torch.cat((self.spk_in[step-1,:,i].unsqueeze(0),spk[j,step-1].unsqueeze(0)), dim=1)) 
                        spk[j,step,i], mem[j,step,i] = lifs[j][i](cur, mem[j,step-1,i])

        return mem, spk
    
    def run(self, new_mask = None):
        if new_mask is None:
            mem, spk = self.simulate(self.mask)
        else:
            mem, spk = self.simulate(new_mask)
        avalanche_data = self.avalanche(spk)
        fit = self.fitness(avalanche_data)
        new_mask=self.mate(fit, self.mask)
        new_mask= self.mutate(new_mask)
        self.mask = new_mask
