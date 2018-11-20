#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:10:52 2017

@author: rodrigodelafuente
"""

import simpy 
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict, OrderedDict
from numpy.random import RandomState
from sklearn.externals.joblib import Parallel, delayed
import xarray as xr

class Warehouse(object):
    def __init__(self, env, order_cut_off, order_target, streams, verbose=False):
        """
        Parameters:
            env: A simpy environment that provides a scheduler
            order_cut_off: The reorder point
            order_target: The desired initial inventory level
            streams: a stream of seed to manage random numbers
            verbose: True prints intermediate information for debuging
            
        Methods:
            behavior: Implements the custom behavior of the warehouse
            update_order: Implements the updating of variables when an order is received
            observe: Collects statistical results and time series information
        """
        ## Pass initial parameters
        self.order_cutoff = order_cut_off
        self.order_target = order_target
        self.env = env
        self.verbose = verbose
        self.on_hand_inventory = self.order_target
        
        ### initialize varaibles
        self.num_ordered = 0
        self.number_ordered = 0
        self.backlog = 0
        self.ordering_cost = 0
        self.num_of_orders = 0
        self.units_ordered = 0
        self.WIP = 0
        self.inventory_position = self.order_target + self.num_ordered - self.backlog
        self.ordering_cost = 0
        self.holding_cost = 0
        
        ### initialize tallys
        self.state_stat_on_hand = 0
        self.state_stat_backlog = 0
        self.t_last_on_hand = 0
        self.t_last_backlog = 0 
        self.last_on_hand = self.on_hand_inventory
        self.last_backlog = self.backlog

        ### initialize methods
        self.env.process(self.behavior(streams))
        self.env.process(self.review(streams))
        self.env.process(self.observe())
        
    def behavior(self,S):
        """
        Parameters:
            S: An object containing all distributions used to generate randomness
            
        Return:
            Void
        """
        
        ## Initialize data collection
        self.tally_service_level = []
        
        while True:
            ################
            # Order Processing Behavior
            ################
            
            # Generate interarrival exponential(1./10)
            interarrival = S.generate_interarrival() 
            yield self.env.timeout(interarrival)
            
            # Generate demand 
            demand = S.generate_demand()
            
            ###################
            # Implement a tally Stat MIN(1, on_hand_inventory/demand)
            self.tally_service_level.append(min(1,self.on_hand_inventory/demand))
            ###################
            
            ###################
            # State Statistic Collection
            time_now = self.env.now
            self.state_stat_on_hand += self.last_on_hand * (time_now - self.t_last_on_hand)
            self.t_last_on_hand = time_now
            
            ##################
            # Begin SIMIO Block
            if demand <= self.on_hand_inventory:
                self.on_hand_inventory -= demand
                self.last_on_hand = self.on_hand_inventory
                if self.verbose:
                    print("{:.2f} sold {}".format(self.env.now,demand))
            else:
                ###################
                # State Statistic Collection
                time_now_b = self.env.now
                self.state_stat_backlog += self.last_backlog * (time_now_b - self.t_last_backlog)
                self.backlog += demand - self.on_hand_inventory
                self.last_backlog = self.backlog
                self.t_last_backlog = time_now_b
                self.on_hand_inventory = 0
                self.last_on_hand = self.on_hand_inventory
                if self.verbose:
                    print("{:.2f} sold {} and backlog {}".format(self.env.now,
                              self.on_hand_inventory, self.backlog))        
            # End SIMIO Block
            ###################
            
            if self.verbose:
                print("{:.2f} inventory_position {}".format(self.env.now,
                                                      self.inventory_position))
                print("{:.2f} on hand {}".format(self.env.now,
                                                      self.on_hand_inventory))
            
    ######################
    # Review behavior
    ######################
    def review(self,S):
        while True:
            ## TODO: pass review period as property
            yield self.env.timeout(1)
            self.inventory_position = self.on_hand_inventory + self.num_ordered - self.backlog
            if self.inventory_position < self.order_cutoff:
                self.number_ordered = max(0,self.order_target-self.inventory_position)
                self.num_of_orders += 1
                self.WIP += self.number_ordered
                self.units_ordered += self.number_ordered
                Supplier(self.env,self.num_of_orders,self.number_ordered,self.update_inventory,S)
                
    def update_inventory(self,num_ordered):
        ###################
        # State Statistic Collection on hand
        time_now = self.env.now
        self.state_stat_on_hand += self.last_on_hand * (time_now - self.t_last_on_hand)
        self.t_last_on_hand = time_now
        self.on_hand_inventory = max(0,self.on_hand_inventory+num_ordered-self.backlog)
        self.last_on_hand = self.on_hand_inventory
        
        self.WIP -= num_ordered
        
        ###################
        # State Statistic Collection backlog
        time_now_b = self.env.now
        self.state_stat_backlog += self.last_backlog * (time_now_b - self.t_last_backlog)
        self.backlog = max(0, self.backlog-num_ordered)
        self.last_backlog = self.backlog
        self.t_last_backlog = time_now_b
        
        if self.verbose:
            print("{:.2f} received order for {}, {} in on hand inventory".
                   format(self.env.now,num_ordered,self.on_hand_inventory))
    
    def observe(self):
        mov_avg_time = 10 
        results = defaultdict(list)
        results["obs_time"]
        results["on_hand_inventory"]
        results["backlog"]
        results["inventory_position"]
        results["state_stat_on_hand"]
        results["state_stat_backlog"]
        results["number_ordered"]
        results["number_of_orders"]
        results["avg_service_level"]
        results["avg_service_level_ma"]
        results["total_cost"]
        self.results = OrderedDict(results)

        ##
        while True:
            ##########
            # Observe period State Stats
            
            # On hand
            time_now = self.env.now
            self.state_stat_on_hand += self.last_on_hand * (time_now - self.t_last_on_hand)
            self.t_last_on_hand = time_now
            
            # Backlog
            time_now_b = self.env.now
            self.state_stat_backlog += self.last_backlog * (time_now_b - self.t_last_backlog)
            self.t_last_backlog = time_now_b
            ##########
            
            self.results["obs_time"].append(self.env.now)
            
            if time_now >= 1:
                self.results["state_stat_on_hand"].append(self.state_stat_on_hand/time_now)
                self.results["state_stat_backlog"].append(self.state_stat_backlog*5/time_now)
                self.results["number_ordered"].append(self.units_ordered*3/time_now)
                self.results["number_of_orders"].append(self.num_of_orders*32/time_now)
                self.results["avg_service_level"].append(np.mean(self.tally_service_level))
                self.results["total_cost"].append(self.state_stat_on_hand/time_now+
                self.state_stat_backlog*5/time_now+self.units_ordered*3/time_now+
                self.num_of_orders*32/time_now)
            else:
                self.results["state_stat_on_hand"].append(self.state_stat_on_hand)
                self.results["state_stat_backlog"].append(self.state_stat_backlog)
                self.results["number_ordered"].append(self.units_ordered)
                self.results["number_of_orders"].append(self.num_of_orders)
                self.results["avg_service_level"].append(1)
                self.results["total_cost"].append(self.state_stat_on_hand+
                self.state_stat_backlog*5+self.units_ordered*3+
                self.num_of_orders*32)
            
            if len(self.tally_service_level) >= mov_avg_time:
                self.results["avg_service_level_ma"].append(np.mean(self.tally_service_level[-mov_avg_time:]))
            else:
                self.results["avg_service_level_ma"].append(1)

            self.results["on_hand_inventory"].append(self.on_hand_inventory)
            self.results["backlog"].append(self.backlog)
            self.results["inventory_position"].append(self.inventory_position)

            yield self.env.timeout(1)
                
class Supplier(object): 
    """
    This is the supplier object. It has implemented two methos: __init__()
    and execute_order(). The later uses a resource to model the lead time
    """
    def __init__(self,env,id_,number_ordered,fun_helper,
                     streams):
        """
        
        """
        self.env = env
        self.id = id_
        self.num_ordered = number_ordered
        self.env.process(self.execute_order(fun_helper,streams))
        
    def execute_order(self,update,S,verbose=False):
        res = R(self.env)
        res.activate()
        with res.server.request() as request:
            if verbose:
                print ("{:.2f} place order for {}".format(self.env.now,self.num_ordered))
            lead_time = S.generate_leadTime()
            if verbose:
                print("Scheduled to arrive on {:.2f}".format(self.env.now+lead_time))
            yield request
            yield self.env.timeout(lead_time)
            update(self.num_ordered)

##### Streams #####
class Streams(object):
    """
    Description: This class takes care of the creation of all distributions
    inside the simulation. 
    """
    def __init__(self,demand_seed,inter_seed,lead_seed):
        """
        This is the constructor of the class.
        
        Parameters:
            demand_seed: A seed to implement CRN on the demand
            inter_seed: A seed to implement CRN on the interarrivals
            lead_seed: A seed to implement CRN on the lead time
        """
        self.demand_rand = RandomState()
        self.demand_rand.seed(demand_seed)
        self.inter_rand = RandomState()
        self.inter_rand.seed(inter_seed)
        self.leadTime_rand = RandomState()   
        self.leadTime_rand.seed(lead_seed)
                      
    def generate_interarrival(self):
        """
        Implements random.exponential() with lambda(1/10)
        """
        return self.inter_rand.exponential(1./10)

    def generate_leadTime(self):
        """
        Implements random.uniform() with a=0.5 and b=1
        """
        return self.leadTime_rand.uniform(0.5,1)

    def generate_demand(self):
        """
        Implements a discrete distribution to characterize the demand
        """
        demand_vals = np.arange(1,5)
        demand_prob = (0.17, 0.33, 0.33, 0.17)
        return self.demand_rand.choice(demand_vals,p=demand_prob)          

##### Resources #####
class R(object):
    """
    This is a resource used to implement the lead time of the demand
    """
    def __init__(self,env):
        """
        Parameters: 
            env: Pass a Simpy environment 
        """
        self.env = env
        
    def activate(self):
        self.server = simpy.Resource(self.env, capacity=4)
    
class Model(object):
    """
    This is the canvas equivalent found in all proprietary software. It consolidate
    all objects and information needed to run a single scenario with just one replicate
    
    Methods:
        __init__()
        run()
    """
    def __init__(self,seeds,order_cut_off,order_target, simulation_time):
        """
        This is the constructor of the Model object:
            
        Parameters:
            demand_seed: A seed to control randomness in the damand
            inter_seed: A seed to control randomness in the interarrivals
            lead_seed: A seed to control randomness in the lead time
            order_cut_off: The reorder point
            order_target: The maximum  inventory target
            simulation_time: the length of the simulation
        """
        self.demand_seed = seeds[0]
        self.inter_seed = seeds[1]
        self.lead_seed = seeds[2]
        self.order_cut_off = order_cut_off
        self.order_target = order_target
        self.simulation_time = simulation_time
        
    def run(self):
        """
        This is the run method that integrates environment, streams and the
        simulation model
        
        Return:
            All the information collected by the observe method implemented in
            the Warehouse object
        """
        env = simpy.Environment()
        S = Streams(self.demand_seed,self.inter_seed,self.lead_seed)
        warehouse = Warehouse(env, self.order_cut_off, self.order_target,S)
        env.run(until=self.simulation_time)
        return warehouse.results
        
class Replicator(object):
    """
    This object implements a replicator to run the same experiment with different seeds
    """
    def __init__(self, seeds):
        """
        This is the constructor of the class and requires a list of integers
        
        Parameters:
            seeds: a list of integers
        """
        self.seeds = seeds
    
    def run(self,params):
        """
        This method instantiate a Model object with a specific set of parameters and
        runs it sevel times with different seeds
        
        Parameters:
            params: A tuple containing all parameters needed to instantiate a Model object
            
        Return:
            A list of list containing all the information provided by the observe
            method implemented in observe method of the Model object.
        """
        return [Model(seeds,*params).run() for seeds in self.seeds], params

class Experiment(object):
    """
    This class implements a parallelized process to perform experimental design 
    evaluation. It has two method implemented
        
        Methods:
            __init__()
            run()
    """
    def __init__(self,num_replics,scenarios):
        """
        This is the construtor of the class.
        
        Parameters:
            num_replics: An int value specifying the number of replicates to 
                        perform
            scenarios: A list of tuples, where each tuple contains the basic
                        information to instantiate a Model object
        """
        self.seeds = list(zip(*3*[iter([i for i in range(num_replics*3)])]))
        self.scenarios = scenarios
        
    def run(self):
        """
        This is the run method that does most of the heavy lifting in the model.
        Currently, the implementation forces the experiments to run in parallel
        forcing all cpus to work.
        
        Returns:
            Returns the results across all scenarios and replications
        """
        cpu = mp.cpu_count()
        self.results = Parallel(n_jobs=cpu, verbose=5)(delayed(Replicator(self.seeds).run)(scenario) for scenario in self.scenarios)
     
    def process_results(self, results_key):
        '''
        results: "obs_time","on_hand_inventory","backlog","inventory_position",
        "state_stat_on_hand", "state_stat_backlog","number_ordered","number_of_orders".
        "avg_service_level"
        '''
        
        ## Prepare container for the results
        S = list(set([element[1] for element in self.scenarios]))
        s = list(set([element[0] for element in self.scenarios]))
        S_len = len(S)
        s_len = len(s)
        S_ = [str(e) for e in S]
        s_ = [str(e) for e in s]
        time = self.scenarios[0][2]
        
        output = {}
        output['means'] = {}
        output['stds'] = {}

        ## Processing the results
        for key in results_key:
            print('key =',key)
            data_container_means = xr.DataArray(np.zeros((S_len,s_len,time)), 
                                    coords={'S':S_, 's':s_},dims=('S', 's', 't'))
            data_container_stds = xr.DataArray(np.zeros((S_len,s_len,time)), 
                                    coords={'S':S_, 's':s_},dims=('S', 's', 't'))
            for r in self.results:
                rep = [r[0][i][key] for i in range(len(r[0]))]
                avg = [float(sum(col))/len(col) for col in zip(*rep)]
                std = [np.std(col,ddof=1) for col in zip(*rep)]
                data_container_means.loc[str(r[1][1]), str(r[1][0])] = avg
                data_container_stds.loc[str(r[1][1]), str(r[1][0])] = std
            output['means'][key] = data_container_means
            output['stds'][key] = data_container_stds
        return output

if __name__ == '__main__':
    time = 120
    scenarios = [(x, y, time) for x in range(20,40) for y in range(40,79)]
    exp = Experiment(30,scenarios)
    exp.run()
    '''
    result = exp.process_results(["state_stat_on_hand", "state_stat_backlog"])
    result1 = exp.process_results(["number_ordered", "number_of_orders"])
    result2 = exp.process_results(["avg_service_level"])

      #                            "number_ordered","number_of_orders",
      #                            "avg_service_level","total_cost"])
      
    service_level = result2["means"]["avg_service_level"]
    sl=service_level.to_pandas()
    pdFrame=sl.to_frame()
    writer = pd.ExcelWriter('C:/Users/rodel/Desktop/Book1.xlsx')
    pdFrame.to_excel(writer,'Sheet1')
    service_level = result2["stds"]["avg_service_level"]
    sl=service_level.to_pandas()
    pdFrame_std=sl.to_frame()
    pdFrame_std.to_excel(writer,'Sheet2')
    writer.save()

    for name in ['means','stds']:
        means = result2[name]['avg_service_level_ma']
        indices = [(int(np.array(S).tolist()),int(np.array(s).tolist())) for 
             S in means.S for s in means.s]
    
        results = [list(np.array(means.loc[{'S':str(S),'s':str(s)}])) for S,s in indices]
    
        time_slice = [str(i+1) for i in range(120)]
        features = ['S','s']
        features.extend(time_slice)
    
        for i, j in enumerate(indices):
            results[i].insert(0,j[1])
            results[i].insert(0,j[0])
        collected_data = pd.DataFrame(results, columns=features)
        name_ = 'output_'+ name +'_ma10'
        collected_data.to_csv('E:/Proyectos/Personales/Inventory/'+name_+'.csv')
    
        experiment = [(20,48),(21,58),(22,68),(23,78),(24,40),(25,50),(26,60),(27,70),
                  (28,56),(29,42),(30,52),(31,62),(32,72),(33,66),(34,44),(35,54),
                  (36,64),(37,74),(38,76),(39,46)]
        experiment = [(element[1],element[0]) for element in experiment]
    
        results_exp = [list(np.array(means.loc[{'S':str(S),'s':str(s)}])) for S,s in experiment]
    
        time_slice = [str(i+1) for i in range(120)]
        features = ['S','s']
        features.extend(time_slice)
    
        for i, j in enumerate(experiment):
            results_exp[i].insert(0,j[1])
            results_exp[i].insert(0,j[0])
        collected_data = pd.DataFrame(results_exp, columns=features)
        name_ = 'output_exp_'+ name + '_ma10'
        collected_data.to_csv('E:/Proyectos/Personales/Inventory/'+ name_ + '.csv')
        
    '''