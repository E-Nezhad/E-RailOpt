# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:06:37 2024

@author: z3547138
"""
import gurobipy as gp
import numpy as np
import openpyxl
import os
import pandas as pd
import pyomo.environ as pe
import pyomo.opt
import random
import scipy as sp
import shutil
import time
import timeit
import xlrd
import xlsxwriter
import zipfile
from datetime import datetime, date
from openpyxl import Workbook
from pathlib import Path
from pyomo.environ import *
from pyomo.environ import Constraint
from xls2xlsx import XLS2XLSX


print ('Running functions...')


'''The following model assumes the charge rate is always a constant.'''
def Model_constant_rate_v1(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge, penalty_fix_cost, penalty_delay, M):

    '''Start modelling'''
    time_start = time.time()
    model_constant_rate_v1 = pe.ConcreteModel()
    print ('Defining indices and sets...')
    # Sets and indices
    model_constant_rate_v1.i = pe.Set(initialize = set(Stations.keys()))
    model_constant_rate_v1.i1 = pe.Set(initialize = set(Stations.keys()))
    model_constant_rate_v1.i2 = pe.Set(initialize = set(Stations.keys()))
    model_constant_rate_v1.j = pe.Set(initialize = set(Trains.keys()))
    model_constant_rate_v1.k = pe.Set(initialize = set(Containers.keys()))
    model_constant_rate_v1.k1 = pe.Set(initialize = set(Containers.keys()))
    model_constant_rate_v1.k2 = pe.Set(initialize = set(Containers.keys()))
    
    # Variables
    print ('Defining variables...')
    model_constant_rate_v1.D = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, within = pe.NonNegativeReals)
    model_constant_rate_v1.S_arrive = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_constant_rate_v1.S_depart = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_constant_rate_v1.T_arrive = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, within = pe.NonNegativeReals)
    model_constant_rate_v1.T_depart = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, within = pe.NonNegativeReals)
    model_constant_rate_v1.X = pe.Var(model_constant_rate_v1.i, within = pe.Binary)
    model_constant_rate_v1.Y = pe.Var(model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.Binary)
    model_constant_rate_v1.Z_c = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.Binary)
    model_constant_rate_v1.Z_s = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.Binary)
    model_constant_rate_v1.T_c = pe.Var(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    
    # preprocessing
    print ('Preprocessing')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_constant_rate_v1.
    for j in model_constant_rate_v1.j:
        for k in model_constant_rate_v1.k:
            if k not in Trains[j]['containers']:
                model_constant_rate_v1.Y[j,k].fix(0)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_constant_rate_v1.j:
        model_constant_rate_v1.T_arrive['origin',j].fix(0)
        model_constant_rate_v1.T_depart['origin',j].fix(0)
    
    
    # objective function
    print ('Reading objective...')
    def obj_rule(model_constant_rate_v1):
        cost_fix = sum(Stations[i]['cost_fix'] * model_constant_rate_v1.X[i] for i in model_constant_rate_v1.i)
        cost_delay = sum(sum((model_constant_rate_v1.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_constant_rate_v1.j) for i in model_constant_rate_v1.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_constant_rate_v1.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    print ('a...')
    def delay_define_rule(model_constant_rate_v1, i, j):
        return model_constant_rate_v1.D[i,j] >= model_constant_rate_v1.T_depart[i,j] - model_constant_rate_v1.T_arrive[i,j]
    model_constant_rate_v1.delay_define_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    print ('b...')
    def deploy_swap_charge_rule(model_constant_rate_v1, i):
        return sum(sum((model_constant_rate_v1.Z_c[i,j,k] + model_constant_rate_v1.Z_s[i,j,k]) \
                       for k in model_constant_rate_v1.k if k in Trains[j]['containers']) \
                   for j in model_constant_rate_v1.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_constant_rate_v1.X[i]
    model_constant_rate_v1.deploy_swap_charge_rule = pe.Constraint(model_constant_rate_v1.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    print ('c...')
    def noboth_swap_charge_rule(model_constant_rate_v1, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_constant_rate_v1.Z_s[i,j,k1] + model_constant_rate_v1.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.noboth_swap_charge_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k1, model_constant_rate_v1.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    print ('d...')
    def wait_passenger_rule(model_constant_rate_v1, i, j):
        return model_constant_rate_v1.T_depart[i,j] - model_constant_rate_v1.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_constant_rate_v1.wait_passenger_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    print ('e...')
    def max_number_batteries_station_rule(model_constant_rate_v1, i, j):
        return sum(model_constant_rate_v1.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_constant_rate_v1.max_number_batteries_station_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    print ('f...')
    def max_number_chargers_station_rule(model_constant_rate_v1, i, j):
        return sum(model_constant_rate_v1.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_constant_rate_v1.max_number_chargers_station_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    print ('g...')
    def power_rule(model_constant_rate_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_constant_rate_v1.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_constant_rate_v1.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.power_rule = pe.Constraint(model_constant_rate_v1.i1, model_constant_rate_v1.i2, model_constant_rate_v1.j, rule = power_rule)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    print ('h...')
    def soc_increase_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart[i,j,k] - model_constant_rate_v1.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_increase_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    print ('i...')
    def swap_full_soc_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart[i,j,k] >= model_constant_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.swap_full_soc_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = swap_full_soc_rule)     
    
    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    print ('j...')
    def time_battery_swap_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.T_depart[i,j] - model_constant_rate_v1.T_arrive[i,j] >= hour_battery_swap * model_constant_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.time_battery_swap_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    print ('k...')
    def soc_time_charge_rule_a(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart[i,j,k] <= model_constant_rate_v1.S_arrive[i,j,k] + rate_charge * model_constant_rate_v1.T_c[i,j,k] + M * model_constant_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_time_charge_rule_a = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_time_charge_rule_a)        
    
    print ('l...')
    def soc_time_charge_rule_b(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return rate_charge * model_constant_rate_v1.T_c[i,j,k] <= 1 - model_constant_rate_v1.S_arrive[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_time_charge_rule_b = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_time_charge_rule_b)
    
    print ('m...')
    def soc_time_charge_rule_c(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.T_c[i,j,k] <= model_constant_rate_v1.T_depart[i,j] - model_constant_rate_v1.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_time_charge_rule_c = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_time_charge_rule_c)
    
    print ('n...')
    def soc_time_charge_rule_d(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.T_c[i,j,k] <= M * model_constant_rate_v1.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_time_charge_rule_d = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_time_charge_rule_d)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    print ('o...')
    def no_swap_charge_soc_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart[i,j,k] - model_constant_rate_v1.S_arrive[i,j,k] <= model_constant_rate_v1.Z_c[i,j,k] + model_constant_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.no_swap_charge_soc_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    print ('o2...')
    def soc_depart_origin_rule(model_constant_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart['origin',j,k] >= model_constant_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_depart_origin_rule = pe.Constraint(model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    print ('o3...')
    def soc_arrive_origin_rule(model_constant_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_arrive['origin',j,k] >= model_constant_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.soc_arrive_origin_rule = pe.Constraint(model_constant_rate_v1.j, model_constant_rate_v1.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    print ('o4...')
    def origin_no_chargeswap_rule(model_constant_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.Z_c['origin',j,k] + model_constant_rate_v1.Z_s['origin',j,k] <= 2 * model_constant_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.origin_no_chargeswap_rule = pe.Constraint(model_constant_rate_v1.j, model_constant_rate_v1.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    print ('o5...')
    def T_traveltime_rule(model_constant_rate_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_constant_rate_v1.T_arrive[i2,j] == model_constant_rate_v1.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.T_traveltime_rule = pe.Constraint(model_constant_rate_v1.i1, model_constant_rate_v1.i2, model_constant_rate_v1.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    print ('o6...')
    def nobattery_nochargeswap_rule(model_constant_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return sum(model_constant_rate_v1.Z_c[i,j,k] for i in model_constant_rate_v1.i) + sum(model_constant_rate_v1.Z_s[i,j,k] for i in model_constant_rate_v1.i) <= 2*len(Stations) * model_constant_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.nobattery_nochargeswap_rule = pe.Constraint(model_constant_rate_v1.j, model_constant_rate_v1.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    print ('o7...')
    def nobattery_zerosoc_rule(model_constant_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return sum(model_constant_rate_v1.S_arrive[i,j,k] for i in model_constant_rate_v1.i) + sum(model_constant_rate_v1.S_depart[i,j,k] for i in model_constant_rate_v1.i) <= 2*len(Stations) * model_constant_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.nobattery_zerosoc_rule = pe.Constraint(model_constant_rate_v1.j, model_constant_rate_v1.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    print ('p...')
    def ub_soc_arrive_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.ub_soc_arrive_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = ub_soc_arrive_rule)
    
    print ('q...')
    def ub_soc_depart_rule(model_constant_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_constant_rate_v1.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_constant_rate_v1.ub_soc_depart_rule = pe.Constraint(model_constant_rate_v1.i, model_constant_rate_v1.j, model_constant_rate_v1.k, rule = ub_soc_depart_rule)
    
    
    # solve the model
    print('Solving...')
    opt= pyomo.opt.SolverFactory("cplex")
    optimality_gap = 0.05
    opt.options["mip_tolerances_mipgap"] = optimality_gap
    opt.options["timelimit"] = 300
    results=opt.solve(model_constant_rate_v1, tee=True, keepfiles=True)
    results.write()
    
    time_end = time.time()
    processing_time = time_end - time_start
    
    
    '''Record variables'''
    print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    
    for i in Stations:
        X.update({i: model_constant_rate_v1.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        for j in Trains:
            D[i].update({j: model_constant_rate_v1.D[i,j].value})
            T_arrive[i].update({j: model_constant_rate_v1.T_arrive[i,j].value})
            T_depart[i].update({j: model_constant_rate_v1.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_constant_rate_v1.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_constant_rate_v1.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_constant_rate_v1.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_constant_rate_v1.Z_s[i,j,k].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_constant_rate_v1.Y[j,k].value})
    
    
    return D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s
    













'''The following model assumes the charge rate decreases linearly with SOC. We use piecewise linear approximation to approximate charge rate.
 The model is problematic and needs to be revised!!!'''
def Model_change_rate_v1(Stations, Trains, Containers, Power_train, TravelTime_train, penalty_fix_cost, penalty_delay, \
                         hour_battery_swap, rate_charge_empty, number_segments_hour, Segments_hour, M):
    '''Start modelling'''
    time_start = time.time()
    model_change_rate_v1 = pe.ConcreteModel()
    print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v1.i = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v1.i1 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v1.i2 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v1.j = pe.Set(initialize = set(Trains.keys()))
    model_change_rate_v1.k = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v1.k1 = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v1.k2 = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v1.l_eta = pe.Set(initialize = set(range(0,number_segments_hour+1)))
    model_change_rate_v1.l_tau = pe.Set(initialize = set(range(0,number_segments_hour)))

    
    # Variables
    print ('Defining variables...')
    model_change_rate_v1.D = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, within = pe.NonNegativeReals)
    model_change_rate_v1.S_arrive = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v1.S_depart = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v1.T_arrive = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, within = pe.NonNegativeReals)
    model_change_rate_v1.T_depart = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, within = pe.NonNegativeReals)
    model_change_rate_v1.X = pe.Var(model_change_rate_v1.i, within = pe.Binary)
    model_change_rate_v1.Y = pe.Var(model_change_rate_v1.j, model_change_rate_v1.k, within = pe.Binary)
    model_change_rate_v1.Z_c = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, within = pe.Binary)
    model_change_rate_v1.Z_s = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, within = pe.Binary)
    model_change_rate_v1.T_c = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    model_change_rate_v1.tau = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, model_change_rate_v1.l_tau, within = pe.Binary)  # variable for piecewise linear approximation
    model_change_rate_v1.eta = pe.Var(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, model_change_rate_v1.l_eta, within = pe.NonNegativeReals)  # variable for piecewise linear approximation
    
    
    # preprocessing
    print ('Preprocessing')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v1.
    for j in model_change_rate_v1.j:
        for k in model_change_rate_v1.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v1.Y[j,k].fix(0)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v1.j:
        model_change_rate_v1.T_arrive['origin',j].fix(0)
        model_change_rate_v1.T_depart['origin',j].fix(0)
    
    
    # objective function
    print ('Reading objective...')
    def obj_rule(model_change_rate_v1):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v1.X[i] for i in model_change_rate_v1.i)
        cost_delay = sum(sum((model_change_rate_v1.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v1.j) for i in model_change_rate_v1.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v1.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    print ('a...')
    def delay_define_rule(model_change_rate_v1, i, j):
        return model_change_rate_v1.D[i,j] >= model_change_rate_v1.T_depart[i,j] - model_change_rate_v1.T_arrive[i,j]
    model_change_rate_v1.delay_define_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v1, i):
        return sum(sum((model_change_rate_v1.Z_c[i,j,k] + model_change_rate_v1.Z_s[i,j,k]) \
                       for k in model_change_rate_v1.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v1.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v1.X[i]
    model_change_rate_v1.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v1.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v1, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v1.Z_s[i,j,k1] + model_change_rate_v1.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k1, model_change_rate_v1.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    print ('d...')
    def wait_passenger_rule(model_change_rate_v1, i, j):
        return model_change_rate_v1.T_depart[i,j] - model_change_rate_v1.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v1.wait_passenger_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v1, i, j):
        return sum(model_change_rate_v1.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v1.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v1, i, j):
        return sum(model_change_rate_v1.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v1.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    print ('g...')
    def power_rule(model_change_rate_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v1.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v1.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.power_rule = pe.Constraint(model_change_rate_v1.i1, model_change_rate_v1.i2, model_change_rate_v1.j, rule = power_rule)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    print ('h...')
    def soc_increase_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_depart[i,j,k] - model_change_rate_v1.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_increase_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    print ('i...')
    def swap_full_soc_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_depart[i,j,k] >= model_change_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.swap_full_soc_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = swap_full_soc_rule)     
    
    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    print ('j...')
    def time_battery_swap_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.T_depart[i,j] - model_change_rate_v1.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.time_battery_swap_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = time_battery_swap_rule)     
    
    # The relationship between S_depart, S_arrive, and T_c. This constraint is different from soc_time_charge_rule_a in Model_constant_rate_v1.
    print ('k...')
    def soc_time_charge_rule_a(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return (1-rate_charge_empty) * model_change_rate_v1.S_depart[i,j,k] + rate_charge_empty * model_change_rate_v1.S_arrive[i,j,k] - 1 \
                    + (1-rate_charge_empty) * sum((1-rate_charge_empty)**Segments_hour[l]['hour'] * model_change_rate_v1.eta[i,j,k,l] \
                                                  for l in model_change_rate_v1.l_eta) \
                    - (1-rate_charge_empty) * M * model_change_rate_v1.Z_s[i,j,k]  \
                    <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a)        
    
    print ('k1...')
    def soc_time_charge_rule_a_PLA_1(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_hour[l]['hour'] * model_change_rate_v1.eta[i,j,k,l]) for l in model_change_rate_v1.l_eta) \
                    - model_change_rate_v1.T_c[i,j,k] \
                    == 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_1 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a_PLA_1)        

    print ('k2...')
    def soc_time_charge_rule_a_PLA_2(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v1.eta[i,j,k,l] for l in model_change_rate_v1.l_eta) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_2 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a_PLA_2)        

    print ('k3...')
    def soc_time_charge_rule_a_PLA_3(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.eta[i,j,k,0] <= model_change_rate_v1.tau[i,j,k,0]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_3 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a_PLA_3)        

    print ('k4...')
    def soc_time_charge_rule_a_PLA_4(model_change_rate_v1, i, j, k, l_eta):
        if k in Trains[j]['containers'] and l_eta < max(Segments_hour) and l_eta != 0:
            return model_change_rate_v1.eta[i,j,k,l_eta] <= model_change_rate_v1.tau[i,j,k,l_eta-1] + model_change_rate_v1.tau[i,j,k,l_eta]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_4 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, model_change_rate_v1.l_eta, rule = soc_time_charge_rule_a_PLA_4)        

    print ('k5...')
    def soc_time_charge_rule_a_PLA_5(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.eta[i,j,k,max(Segments_hour)] <= model_change_rate_v1.tau[i,j,k,max(Segments_hour)-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_5 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a_PLA_5)        

    print ('k6...')
    def soc_time_charge_rule_a_PLA_6(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v1.tau[i,j,k,l] for l in model_change_rate_v1.l_tau if l != max(Segments_hour)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_a_PLA_6 = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_a_PLA_6)        
    
    # print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v1.T_c[i,j,k] <= 1 - model_change_rate_v1.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v1.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_b)
    
    print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.T_c[i,j,k] <= model_change_rate_v1.T_depart[i,j] - model_change_rate_v1.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_c)
    
    print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.T_c[i,j,k] <= M * model_change_rate_v1.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_time_charge_rule_d)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_depart[i,j,k] - model_change_rate_v1.S_arrive[i,j,k] <= model_change_rate_v1.Z_c[i,j,k] + model_change_rate_v1.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_depart['origin',j,k] >= model_change_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_depart_origin_rule = pe.Constraint(model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_arrive['origin',j,k] >= model_change_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v1.j, model_change_rate_v1.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.Z_c['origin',j,k] + model_change_rate_v1.Z_s['origin',j,k] <= 2 * model_change_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v1.j, model_change_rate_v1.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    print ('o5...')
    def T_traveltime_rule(model_change_rate_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v1.T_arrive[i2,j] == model_change_rate_v1.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.T_traveltime_rule = pe.Constraint(model_change_rate_v1.i1, model_change_rate_v1.i2, model_change_rate_v1.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v1.Z_c[i,j,k] for i in model_change_rate_v1.i) + sum(model_change_rate_v1.Z_s[i,j,k] for i in model_change_rate_v1.i) <= 2*len(Stations) * model_change_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v1.j, model_change_rate_v1.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v1, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v1.S_arrive[i,j,k] for i in model_change_rate_v1.i) + sum(model_change_rate_v1.S_depart[i,j,k] for i in model_change_rate_v1.i) <= 2*len(Stations) * model_change_rate_v1.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v1.j, model_change_rate_v1.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = ub_soc_arrive_rule)
    
    print ('q...')
    def ub_soc_depart_rule(model_change_rate_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v1.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v1.ub_soc_depart_rule = pe.Constraint(model_change_rate_v1.i, model_change_rate_v1.j, model_change_rate_v1.k, rule = ub_soc_depart_rule)
    
    
    # solve the model
    print('Solving...')
    opt= pyomo.opt.SolverFactory("cplex")
    optimality_gap = 0.05
    opt.options["mip_tolerances_mipgap"] = optimality_gap
    opt.options["timelimit"] = 300
    results=opt.solve(model_change_rate_v1, tee=True, keepfiles=True)
    results.write()
    
    time_end = time.time()
    processing_time = time_end - time_start
    
    
    '''Record variables'''
    print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    eta = {}
    tau = {}
    
    for i in Stations:
        X.update({i: model_change_rate_v1.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        eta.update({i: {}})
        tau.update({i: {}})
        for j in Trains:
            D[i].update({j: model_change_rate_v1.D[i,j].value})
            T_arrive[i].update({j: model_change_rate_v1.T_arrive[i,j].value})
            T_depart[i].update({j: model_change_rate_v1.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            eta[i].update({j: {}})
            tau[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_change_rate_v1.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_change_rate_v1.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_change_rate_v1.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_change_rate_v1.Z_s[i,j,k].value})
                T_c[i][j].update({k: model_change_rate_v1.T_c[i,j,k].value})
                eta[i][j].update({k: {}})
                tau[i][j].update({k: {}})
                for l in Segments_hour:
                    eta[i][j][k].update({l: model_change_rate_v1.eta[i,j,k,l].value})
                    if l != number_segments_hour:
                        tau[i][j][k].update({l: model_change_rate_v1.tau[i,j,k,l].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_change_rate_v1.Y[j,k].value})
    
    
    return D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, eta, tau
    














'''The following model assumes the charge rate is linear to SOC.
This model is same as Model_constant_rate_v1 except constraint soc_time_charge_rule_a.
The model is non-linear because due to constraints soc_time_charge_rule_a, as there is (1-S_arrive) * (1-r)^{T_c}, (which is a product of a variable and an exponential).'''
def Model_change_rate_v2(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, M, epsilon, buffer_soc_estimate):

    time_start = time.time()
    model_change_rate_v2 = pe.ConcreteModel()
    print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v2.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_change_rate_v2.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v2.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v2.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    
    # Variables
    print ('Defining variables...')
    model_change_rate_v2.D = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, within = pe.NonNegativeReals)
    model_change_rate_v2.S_arrive = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v2.S_depart = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v2.T_arrive = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, within = pe.NonNegativeReals)
    model_change_rate_v2.T_depart = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, within = pe.NonNegativeReals)
    model_change_rate_v2.X = pe.Var(model_change_rate_v2.i, within = pe.Binary)
    model_change_rate_v2.Y = pe.Var(model_change_rate_v2.j, model_change_rate_v2.k, within = pe.Binary)
    model_change_rate_v2.Z_c = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, within = pe.Binary)
    model_change_rate_v2.Z_s = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, within = pe.Binary)
    model_change_rate_v2.T_c = pe.Var(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    
    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v2.
    for j in model_change_rate_v2.j:
        for k in model_change_rate_v2.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v2.Y[j,k].fix(0)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v2.j:
        model_change_rate_v2.T_arrive['origin',j].fix(0)
        model_change_rate_v2.T_depart['origin',j].fix(0)
    
    
    # objective function
    print ('Reading objective...')
    def obj_rule(model_change_rate_v2):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v2.X[i] for i in model_change_rate_v2.i)
        cost_delay = sum(sum((model_change_rate_v2.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v2.j) for i in model_change_rate_v2.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v2.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    print ('a...')
    def delay_define_rule(model_change_rate_v2, i, j):
        return model_change_rate_v2.D[i,j] >= model_change_rate_v2.T_depart[i,j] - model_change_rate_v2.T_arrive[i,j]
    model_change_rate_v2.delay_define_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v2, i):
        return sum(sum((model_change_rate_v2.Z_c[i,j,k] + model_change_rate_v2.Z_s[i,j,k]) \
                       for k in model_change_rate_v2.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v2.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v2.X[i]
    model_change_rate_v2.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v2.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v2, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v2.Z_s[i,j,k1] + model_change_rate_v2.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k1, model_change_rate_v2.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    print ('d...')
    def wait_passenger_rule(model_change_rate_v2, i, j):
        return model_change_rate_v2.T_depart[i,j] - model_change_rate_v2.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v2.wait_passenger_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    print ('e0...')
    def consecutive_battery_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            return sum(model_change_rate_v2.Y[j,kp] \
                       for kp in model_change_rate_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                   <= sum(1 for kp in model_change_rate_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                      * model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.consecutive_battery_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v2, i, j):
        return sum(model_change_rate_v2.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v2.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v2, i, j):
        return sum(model_change_rate_v2.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v2.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    print ('g...')
    def power_rule(model_change_rate_v2, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v2.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v2.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.power_rule = pe.Constraint(model_change_rate_v2.i1, model_change_rate_v2.i2, model_change_rate_v2.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    print ('g1...')
    def battery_sequential_rule(model_change_rate_v2, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            k1_id, k2_id = int(((k1.split('container '))[1].split(' in')[0])), int(((k2.split('container '))[1].split(' in')[0]))
            if k1_id < k2_id:
                return model_change_rate_v2.S_arrive[i,j,k1] * (model_change_rate_v2.S_arrive[i,j,k2]-1) == 0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.battery_sequential_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k1, model_change_rate_v2.k2, rule = battery_sequential_rule)          

    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    print ('h...')
    def soc_increase_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] - model_change_rate_v2.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_increase_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    print ('i...')
    def swap_full_soc_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] >= model_change_rate_v2.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.swap_full_soc_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_change_rate_v2, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return model_change_rate_v2.S_arrive[i2,j,k] - model_change_rate_v2.S_depart[i1,j,k] <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v2.i1, model_change_rate_v2.i2, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    print ('j...')
    def time_battery_swap_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.T_depart[i,j] - model_change_rate_v2.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v2.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.time_battery_swap_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    print ('k0...')
    def soc_time_charge_rule_a_Zs(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] \
                   - model_change_rate_v2.S_arrive[i,j,k] \
                   - (1-model_change_rate_v2.S_arrive[i,j,k]) * (1 - (1-rate_charge_empty)**(1/float(buffer_soc_estimate) * model_change_rate_v2.T_c[i,j,k])) \
                   - 2 * model_change_rate_v2.Z_s[i,j,k] \
                   <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_time_charge_rule_a_Zs = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_a_Zs)        
    
    print ('k1...')
    def soc_time_charge_rule_a_Zc(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] \
                   - model_change_rate_v2.S_arrive[i,j,k] \
                   - (1-model_change_rate_v2.S_arrive[i,j,k]) * (1 - (1-rate_charge_empty)**(1/float(buffer_soc_estimate) * model_change_rate_v2.T_c[i,j,k])) \
                   + epsilon * model_change_rate_v2.Z_c[i,j,k] \
                   >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_time_charge_rule_a_Zc = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_a_Zc)        

    
    print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v2.T_c[i,j,k] <= 1 - model_change_rate_v2.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v2.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_b)
    
    print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.T_c[i,j,k] <= model_change_rate_v2.T_depart[i,j] - model_change_rate_v2.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_c)
    
    print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.T_c[i,j,k] <= M * model_change_rate_v2.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_d)
    
    print ('n1...')
    def soc_time_charge_rule_e(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.Z_c[i,j,k] <= M * model_change_rate_v2.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_time_charge_rule_e)

    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] - model_change_rate_v2.S_arrive[i,j,k] <= model_change_rate_v2.Z_c[i,j,k] + model_change_rate_v2.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart['origin',j,k] >= model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_depart_origin_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_arrive['origin',j,k] >= model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.Z_c['origin',j,k] + model_change_rate_v2.Z_s['origin',j,k] <= 2 * model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    print ('o5...')
    def T_traveltime_rule(model_change_rate_v2, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v2.T_arrive[i2,j] == model_change_rate_v2.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.T_traveltime_rule = pe.Constraint(model_change_rate_v2.i1, model_change_rate_v2.i2, model_change_rate_v2.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2.Z_c[i,j,k] for i in model_change_rate_v2.i) + sum(model_change_rate_v2.Z_s[i,j,k] for i in model_change_rate_v2.i) <= 2*len(Stations) * model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v2, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2.S_arrive[i,j,k] for i in model_change_rate_v2.i) + sum(model_change_rate_v2.S_depart[i,j,k] for i in model_change_rate_v2.i) <= 2*len(Stations) * model_change_rate_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = ub_soc_arrive_rule)
    
    print ('q...')
    def ub_soc_depart_rule(model_change_rate_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.ub_soc_depart_rule = pe.Constraint(model_change_rate_v2.i, model_change_rate_v2.j, model_change_rate_v2.k, rule = ub_soc_depart_rule)
    
    print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v2.
    def preprocessing_Y0_rule(model_change_rate_v2, j, k):
        if k not in Trains[j]['containers']:
            return model_change_rate_v2.Y[j,k] == 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2.preprocessing_Y0_rule = pe.Constraint(model_change_rate_v2.j, model_change_rate_v2.k, rule = preprocessing_Y0_rule)
       
    print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_change_rate_v2, j):
        return model_change_rate_v2.T_arrive['origin',j] + model_change_rate_v2.T_depart['origin',j] == 0
    model_change_rate_v2.preprocessing_T0_rule = pe.Constraint(model_change_rate_v2.j, rule = preprocessing_T0_rule)
      
    
        
    # solve the model
    print('Solving...')
    # opt= pyomo.opt.SolverFactory("cplex")
    # optimality_gap = 0.05
    # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 300
    # results=opt.solve(model_change_rate_v2, tee=True, keepfiles=True)
    # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_change_rate_v2, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_change_rate_v2)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # import os
    # from pyomo.environ import *
    couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    opt = SolverFactory("couenne") 
    # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # opt.options['logFile'] = 'couenne_log.txt'
    results=opt.solve(model_change_rate_v2, timelimit=300, logfile = 'mylog.txt', tee=True)
    results.write()
    
    
    time_end = time.time()
    processing_time = time_end - time_start
    
    
    '''Record variables'''
    # print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    
    for i in Stations:
        X.update({i: model_change_rate_v2.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        for j in Trains:
            D[i].update({j: model_change_rate_v2.D[i,j].value})
            T_arrive[i].update({j: model_change_rate_v2.T_arrive[i,j].value})
            T_depart[i].update({j: model_change_rate_v2.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_change_rate_v2.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_change_rate_v2.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_change_rate_v2.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_change_rate_v2.Z_s[i,j,k].value})
                T_c[i][j].update({k: model_change_rate_v2.T_c[i,j,k].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_change_rate_v2.Y[j,k].value})
    
    
    return D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, results
    














'''This model is same as Model_change_rate_v2 except that this model has all X and Y variables fixed:
    - If a train is selected, then fix X=1. Otherwise fix X=0.
    - Make each train carry batteries in all its containers Y=1'''
def Model_change_rate_v3(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, \
                         penalty_delay, penalty_fix_cost, M, epsilon, buffer_soc_estimate, stations_deploy_list):

    time_start = time.time()
    model_change_rate_v3 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v3.i = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v3.i1 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v3.i2 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v3.j = pe.Set(initialize = set(Trains.keys()))
    model_change_rate_v3.k = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v3.k1 = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v3.k2 = pe.Set(initialize = set(Containers.keys()))
    
    # Variables
    # print ('Defining variables...')
    model_change_rate_v3.D = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, within = pe.NonNegativeReals)
    model_change_rate_v3.S_arrive = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v3.S_depart = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v3.T_arrive = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, within = pe.NonNegativeReals)
    model_change_rate_v3.T_depart = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, within = pe.NonNegativeReals)
    model_change_rate_v3.X = pe.Var(model_change_rate_v3.i, within = pe.Binary)
    model_change_rate_v3.Y = pe.Var(model_change_rate_v3.j, model_change_rate_v3.k, within = pe.Binary)
    model_change_rate_v3.Z_c = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, within = pe.Binary)
    model_change_rate_v3.Z_s = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, within = pe.Binary)
    model_change_rate_v3.T_c = pe.Var(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    
    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3.
    for j in model_change_rate_v3.j:
        for k in model_change_rate_v3.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v3.Y[j,k].fix(0)
            else:
                # Make each train carry batteries in all its containers
                model_change_rate_v3.Y[j,k].fix(1)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v3.j:
        model_change_rate_v3.T_arrive['origin',j].fix(0)
        model_change_rate_v3.T_depart['origin',j].fix(0)
    # If a train is selected, then fix X=1. Otherwise fix X=0
    for i in model_change_rate_v3.i:
        if i in stations_deploy_list:
            model_change_rate_v3.X[i].fix(1)
        else:
            model_change_rate_v3.X[i].fix(0)
    
    
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_change_rate_v3):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v3.X[i] for i in model_change_rate_v3.i)
        cost_delay = sum(sum((model_change_rate_v3.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v3.j) for i in model_change_rate_v3.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v3.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_change_rate_v3, i, j):
        return model_change_rate_v3.D[i,j] >= model_change_rate_v3.T_depart[i,j] - model_change_rate_v3.T_arrive[i,j]
    model_change_rate_v3.delay_define_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v3, i):
        return sum(sum((model_change_rate_v3.Z_c[i,j,k] + model_change_rate_v3.Z_s[i,j,k]) \
                       for k in model_change_rate_v3.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v3.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v3.X[i]
    model_change_rate_v3.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v3.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v3, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v3.Z_s[i,j,k1] + model_change_rate_v3.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k1, model_change_rate_v3.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_change_rate_v3, i, j):
        return model_change_rate_v3.T_depart[i,j] - model_change_rate_v3.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v3.wait_passenger_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            return sum(model_change_rate_v3.Y[j,kp] \
                       for kp in model_change_rate_v3.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                   <= sum(1 for kp in model_change_rate_v3.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                      * model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.consecutive_battery_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v3, i, j):
        return sum(model_change_rate_v3.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v3.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v3, i, j):
        return sum(model_change_rate_v3.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v3.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_change_rate_v3, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v3.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v3.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.power_rule = pe.Constraint(model_change_rate_v3.i1, model_change_rate_v3.i2, model_change_rate_v3.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule(model_change_rate_v3, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            k1_id, k2_id = int(((k1.split('container '))[1].split(' in')[0])), int(((k2.split('container '))[1].split(' in')[0]))
            if k1_id < k2_id:
                return model_change_rate_v3.S_arrive[i,j,k1] * (model_change_rate_v3.S_arrive[i,j,k2]-1) == 0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.battery_sequential_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k1, model_change_rate_v3.k2, rule = battery_sequential_rule)          

    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] - model_change_rate_v3.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_increase_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] >= model_change_rate_v3.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.swap_full_soc_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_change_rate_v3, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return model_change_rate_v3.S_arrive[i2,j,k] - model_change_rate_v3.S_depart[i1,j,k] <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v3.i1, model_change_rate_v3.i2, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.T_depart[i,j] - model_change_rate_v3.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v3.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.time_battery_swap_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_a_Zs(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] \
                   - model_change_rate_v3.S_arrive[i,j,k] \
                   - (1-model_change_rate_v3.S_arrive[i,j,k]) * (1 - (1-rate_charge_empty)**(1/float(buffer_soc_estimate) * model_change_rate_v3.T_c[i,j,k])) \
                   - 2 * model_change_rate_v3.Z_s[i,j,k] \
                   <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_time_charge_rule_a_Zs = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_a_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_a_Zc(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] \
                   - model_change_rate_v3.S_arrive[i,j,k] \
                   - (1-model_change_rate_v3.S_arrive[i,j,k]) * (1 - (1-rate_charge_empty)**(1/float(buffer_soc_estimate) * model_change_rate_v3.T_c[i,j,k])) \
                   + epsilon * model_change_rate_v3.Z_c[i,j,k] \
                   >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_time_charge_rule_a_Zc = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_a_Zc)        
    
    # print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v3.T_c[i,j,k] <= 1 - model_change_rate_v3.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v3.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.T_c[i,j,k] <= model_change_rate_v3.T_depart[i,j] - model_change_rate_v3.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.T_c[i,j,k] <= M * model_change_rate_v3.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.Z_c[i,j,k] <= M * model_change_rate_v3.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_time_charge_rule_e)

    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] - model_change_rate_v3.S_arrive[i,j,k] <= model_change_rate_v3.Z_c[i,j,k] + model_change_rate_v3.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart['origin',j,k] >= model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_depart_origin_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_arrive['origin',j,k] >= model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.Z_c['origin',j,k] + model_change_rate_v3.Z_s['origin',j,k] <= 2 * model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_change_rate_v3, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v3.T_arrive[i2,j] == model_change_rate_v3.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.T_traveltime_rule = pe.Constraint(model_change_rate_v3.i1, model_change_rate_v3.i2, model_change_rate_v3.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3.Z_c[i,j,k] for i in model_change_rate_v3.i) + sum(model_change_rate_v3.Z_s[i,j,k] for i in model_change_rate_v3.i) <= 2*len(Stations) * model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v3, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3.S_arrive[i,j,k] for i in model_change_rate_v3.i) + sum(model_change_rate_v3.S_depart[i,j,k] for i in model_change_rate_v3.i) <= 2*len(Stations) * model_change_rate_v3.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_change_rate_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3.ub_soc_depart_rule = pe.Constraint(model_change_rate_v3.i, model_change_rate_v3.j, model_change_rate_v3.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. Otherwise, make the container carry a battery. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3.
    def preprocessing_Y_rule(model_change_rate_v3, j, k):
        if k not in Trains[j]['containers']:
            return model_change_rate_v3.Y[j,k] == 0
        else:
            return model_change_rate_v3.Y[j,k] == 1
    model_change_rate_v3.preprocessing_Y_rule = pe.Constraint(model_change_rate_v3.j, model_change_rate_v3.k, rule = preprocessing_Y_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_change_rate_v3, j):
        return model_change_rate_v3.T_arrive['origin',j] + model_change_rate_v3.T_depart['origin',j] == 0
    model_change_rate_v3.preprocessing_T0_rule = pe.Constraint(model_change_rate_v3.j, rule = preprocessing_T0_rule)
      
    # print ('t...')
    # If a train is selected, then fix X=1. Otherwise fix X=0
    def preprocessing_X_rule(model_change_rate_v3, i):
        if i in stations_deploy_list:
            return model_change_rate_v3.X[i] == 1
        else:
            return model_change_rate_v3.X[i] == 0
    model_change_rate_v3.preprocessing_X_rule = pe.Constraint(model_change_rate_v3.i, rule = preprocessing_X_rule)

        
    # solve the model
    # print('Solving...')
    # opt= pyomo.opt.SolverFactory("cplex")
    # optimality_gap = 0.05
    # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 300
    # results=opt.solve(model_change_rate_v3, tee=True, keepfiles=True)
    # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_change_rate_v3, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_change_rate_v3)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # import os
    # from pyomo.environ import *
    couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    opt = SolverFactory("couenne") 
    # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # opt.options['logFile'] = 'couenne_log.txt'
    opt.options['output'] = 0
    results=opt.solve(model_change_rate_v3, timelimit=300, tee=False)
    # results.write()
    print (results.solver.termination_condition)
    

    time_end = time.time()
    processing_time = time_end - time_start
    
    
    '''Record variables'''
    print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    
    for i in Stations:
        X.update({i: model_change_rate_v3.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        for j in Trains:
            D[i].update({j: model_change_rate_v3.D[i,j].value})
            T_arrive[i].update({j: model_change_rate_v3.T_arrive[i,j].value})
            T_depart[i].update({j: model_change_rate_v3.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_change_rate_v3.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_change_rate_v3.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_change_rate_v3.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_change_rate_v3.Z_s[i,j,k].value})
                T_c[i][j].update({k: model_change_rate_v3.T_c[i,j,k].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_change_rate_v3.Y[j,k].value})
    
    
    return D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, results
    










''' This model is same as Model_change_rate_v2_ except:
    (1) In this model, constraints battery_sequential_rule (with product of 2 S variables) have been linearized by introducing a binary variable B
    (2) In this model, we use Piecewise Linear Approximation Rectangle (PLARec) to linearize constraints soc_time_charge_rule_a_Zs and soc_time_charge_rule_a_Zc.
'''
def Model_change_rate_v2_PLARec(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                                M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, gap_op, time_limit_op, mysolver):

    time_start = time.time()
    model_change_rate_v2_PLARec = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v2_PLARec.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2_PLARec.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2_PLARec.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v2_PLARec.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_change_rate_v2_PLARec.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v2_PLARec.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v2_PLARec.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v2_PLARec.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_change_rate_v2_PLARec.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    
    # Variables
    # print ('Defining variables...')
    model_change_rate_v2_PLARec.D = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v2_PLARec.S_arrive = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.NonNegativeReals)
    model_change_rate_v2_PLARec.S_depart = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.NonNegativeReals)
    model_change_rate_v2_PLARec.T_arrive = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v2_PLARec.T_depart = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v2_PLARec.X = pe.Var(model_change_rate_v2_PLARec.i, within = pe.Binary)
    model_change_rate_v2_PLARec.Y = pe.Var(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.Binary)
    model_change_rate_v2_PLARec.Z_c = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.Binary)
    model_change_rate_v2_PLARec.Z_s = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.Binary)
    model_change_rate_v2_PLARec.T_c = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    # variables for linearization
    model_change_rate_v2_PLARec.B = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
    model_change_rate_v2_PLARec.F = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_change_rate_v2_PLARec.beta = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
    model_change_rate_v2_PLARec.gamma = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
    model_change_rate_v2_PLARec.tau = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.v, within = pe.Binary)  # in PLA, binary variables for T_c
    model_change_rate_v2_PLARec.eta = pe.Var(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c


    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v2_PLARec.
    for j in model_change_rate_v2_PLARec.j:
        for k in model_change_rate_v2_PLARec.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v2_PLARec.Y[j,k].fix(0)
            else:
                # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
                model_change_rate_v2_PLARec.Y[j,k].fix(1)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v2_PLARec.j:
        model_change_rate_v2_PLARec.T_arrive['origin',j].fix(0)
        model_change_rate_v2_PLARec.T_depart['origin',j].fix(0)
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_change_rate_v2_PLARec):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v2_PLARec.X[i] for i in model_change_rate_v2_PLARec.i)
        cost_delay = sum(sum((model_change_rate_v2_PLARec.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v2_PLARec.j) for i in model_change_rate_v2_PLARec.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v2_PLARec.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_change_rate_v2_PLARec, i, j):
        return model_change_rate_v2_PLARec.D[i,j] >= model_change_rate_v2_PLARec.T_depart[i,j] - model_change_rate_v2_PLARec.T_arrive[i,j]
    model_change_rate_v2_PLARec.delay_define_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v2_PLARec, i):
        return sum(sum((model_change_rate_v2_PLARec.Z_c[i,j,k] + model_change_rate_v2_PLARec.Z_s[i,j,k]) \
                       for k in model_change_rate_v2_PLARec.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v2_PLARec.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v2_PLARec.X[i]
    model_change_rate_v2_PLARec.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v2_PLARec.i, rule = deploy_swap_charge_rule)        
    
    # # If station is deployed, then at least one battery must be swapped nor charged in station i
    # print ('opt2...')
    def deploy_swap_charge_must_rule_orig_feas(model_change_rate_v2_PLARec, i):
        return sum(sum((model_change_rate_v2_PLARec.Z_c[i,j,k] + model_change_rate_v2_PLARec.Z_s[i,j,k]) \
                        for k in model_change_rate_v2_PLARec.k if k in Trains[j]['containers']) \
                    for j in model_change_rate_v2_PLARec.j) \
                >= model_change_rate_v2_PLARec.X[i]
    model_change_rate_v2_PLARec.deploy_swap_charge_must_rule_orig_feas = pe.Constraint(model_change_rate_v2_PLARec.i, rule = deploy_swap_charge_must_rule_orig_feas)   

    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v2_PLARec, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.Z_s[i,j,k1] + model_change_rate_v2_PLARec.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k1, model_change_rate_v2_PLARec.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_change_rate_v2_PLARec, i, j):
        return model_change_rate_v2_PLARec.T_depart[i,j] - model_change_rate_v2_PLARec.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v2_PLARec.wait_passenger_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            if kid < len(Trains[j]['containers']):
                return sum(model_change_rate_v2_PLARec.Y[j,kp] \
                           for kp in model_change_rate_v2_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                       <= sum(1 for kp in model_change_rate_v2_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                          * model_change_rate_v2_PLARec.Y[j,k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.consecutive_battery_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v2_PLARec, i, j):
        return sum(model_change_rate_v2_PLARec.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v2_PLARec.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v2_PLARec, i, j):
        return sum(model_change_rate_v2_PLARec.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v2_PLARec.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_change_rate_v2_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v2_PLARec.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v2_PLARec.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.power_rule = pe.Constraint(model_change_rate_v2_PLARec.i1, model_change_rate_v2_PLARec.i2, model_change_rate_v2_PLARec.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule_linear_a(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return model_change_rate_v2_PLARec.S_arrive[i,j,k]  -  M * model_change_rate_v2_PLARec.B[i,j,k]  <=  0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.battery_sequential_rule_linear_a = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    def battery_sequential_rule_linear_b(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return (1 - model_change_rate_v2_PLARec.S_arrive[i,j,k_next])  +  M * model_change_rate_v2_PLARec.B[i,j,k] <=  M
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.battery_sequential_rule_linear_b = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] - model_change_rate_v2_PLARec.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_increase_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] >= model_change_rate_v2_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.swap_full_soc_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_change_rate_v2_PLARec, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return model_change_rate_v2_PLARec.S_arrive[i2,j,k] - model_change_rate_v2_PLARec.S_depart[i1,j,k] <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v2_PLARec.i1, model_change_rate_v2_PLARec.i2, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.T_depart[i,j] - model_change_rate_v2_PLARec.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v2_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.time_battery_swap_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_PLA_Zs(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v2_PLARec.F[i,j,k] \
                   - M * model_change_rate_v2_PLARec.Z_s[i,j,k] \
                   <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_PLA_Zc(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v2_PLARec.F[i,j,k] \
                   + epsilon * model_change_rate_v2_PLARec.Z_c[i,j,k] \
                   >= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_Zc)        

    # print ('k2...')
    def soc_time_charge_rule_PLA_beta1(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2_PLARec.beta[i,j,k,u] for u in model_change_rate_v2_PLARec.u if u != max(model_change_rate_v2_PLARec.u)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    def soc_time_charge_rule_PLA_gamma1(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2_PLARec.gamma[i,j,k,u] for u in model_change_rate_v2_PLARec.u) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_tau1(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2_PLARec.tau[i,j,k,v] for v in model_change_rate_v2_PLARec.v if v != max(model_change_rate_v2_PLARec.v)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_change_rate_v2_PLARec.gamma[i,j,k,u]) \
                       for u in model_change_rate_v2_PLARec.u) \
                   == model_change_rate_v2_PLARec.S_arrive[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_hour_charge[v]['hour'] * model_change_rate_v2_PLARec.tau[i,j,k,v] \
                        + (Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_change_rate_v2_PLARec.eta[i,j,k,v]) \
                       for v in model_change_rate_v2_PLARec.v if v != max(model_change_rate_v2_PLARec.v)) \
                   == model_change_rate_v2_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta(model_change_rate_v2_PLARec, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_change_rate_v2_PLARec.u) and u != max(model_change_rate_v2_PLARec.u):
            return model_change_rate_v2_PLARec.gamma[i,j,k,u] <= model_change_rate_v2_PLARec.beta[i,j,k,u-1] + model_change_rate_v2_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == min(model_change_rate_v2_PLARec.u):
            return model_change_rate_v2_PLARec.gamma[i,j,k,u] <= model_change_rate_v2_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == max(model_change_rate_v2_PLARec.u):
            return model_change_rate_v2_PLARec.gamma[i,j,k,u] <= model_change_rate_v2_PLARec.beta[i,j,k,u-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau(model_change_rate_v2_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_change_rate_v2_PLARec.v) and v != max(model_change_rate_v2_PLARec.v):
            return model_change_rate_v2_PLARec.eta[i,j,k,v] <= model_change_rate_v2_PLARec.tau[i,j,k,v-1] + model_change_rate_v2_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == min(model_change_rate_v2_PLARec.v):
            return model_change_rate_v2_PLARec.eta[i,j,k,v] <= model_change_rate_v2_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == max(model_change_rate_v2_PLARec.v):
            return model_change_rate_v2_PLARec.eta[i,j,k,v] <= model_change_rate_v2_PLARec.tau[i,j,k,v-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1(model_change_rate_v2_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_change_rate_v2_PLARec.v) and v != max(model_change_rate_v2_PLARec.v):
            return model_change_rate_v2_PLARec.eta[i,j,k,v] <= model_change_rate_v2_PLARec.tau[i,j,k,v]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    def soc_time_charge_rule_PLA_F_leq(model_change_rate_v2_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v2_PLARec.u) and v != max(model_change_rate_v2_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v2_PLARec.F[i,j,k] \
                   <= sum((model_change_rate_v2_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v2_PLARec.u) \
                      + w * model_change_rate_v2_PLARec.eta[i,j,k,v] \
                      + M * (2 - model_change_rate_v2_PLARec.tau[i,j,k,v] - model_change_rate_v2_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, model_change_rate_v2_PLARec.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    def soc_time_charge_rule_PLA_F_geq(model_change_rate_v2_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v2_PLARec.u) and v != max(model_change_rate_v2_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v2_PLARec.F[i,j,k] \
                   >= sum((model_change_rate_v2_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v2_PLARec.u) \
                      + w * model_change_rate_v2_PLARec.eta[i,j,k,v] \
                      - M * (2 - model_change_rate_v2_PLARec.tau[i,j,k,v] - model_change_rate_v2_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, model_change_rate_v2_PLARec.v, rule = soc_time_charge_rule_PLA_F_geq)        

    # print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v2_PLARec, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v2_PLARec.T_c[i,j,k] <= 1 - model_change_rate_v2_PLARec.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v2_PLARec.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.T_c[i,j,k] <= model_change_rate_v2_PLARec.T_depart[i,j] - model_change_rate_v2_PLARec.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.T_c[i,j,k] <= M * model_change_rate_v2_PLARec.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.Z_c[i,j,k] <= M * model_change_rate_v2_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] - model_change_rate_v2_PLARec.S_arrive[i,j,k] <= model_change_rate_v2_PLARec.Z_c[i,j,k] + model_change_rate_v2_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart['origin',j,k] >= model_change_rate_v2_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_depart_origin_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_arrive['origin',j,k] >= model_change_rate_v2_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.Z_c['origin',j,k] + model_change_rate_v2_PLARec.Z_s['origin',j,k] <= 2 * model_change_rate_v2_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_change_rate_v2_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v2_PLARec.T_arrive[i2,j] == model_change_rate_v2_PLARec.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.T_traveltime_rule = pe.Constraint(model_change_rate_v2_PLARec.i1, model_change_rate_v2_PLARec.i2, model_change_rate_v2_PLARec.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2_PLARec.Z_c[i,j,k] for i in model_change_rate_v2_PLARec.i) + sum(model_change_rate_v2_PLARec.Z_s[i,j,k] for i in model_change_rate_v2_PLARec.i) <= 2*len(Stations) * model_change_rate_v2_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v2_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v2_PLARec.S_arrive[i,j,k] for i in model_change_rate_v2_PLARec.i) + sum(model_change_rate_v2_PLARec.S_depart[i,j,k] for i in model_change_rate_v2_PLARec.i) <= 2*len(Stations) * model_change_rate_v2_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_change_rate_v2_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.ub_soc_depart_rule = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v2_PLARec.
    def preprocessing_Y0_rule(model_change_rate_v2_PLARec, j, k):
        if k not in Trains[j]['containers']:
            return model_change_rate_v2_PLARec.Y[j,k] == 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.preprocessing_Y0_rule = pe.Constraint(model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_change_rate_v2_PLARec, j):
        return model_change_rate_v2_PLARec.T_arrive['origin',j] + model_change_rate_v2_PLARec.T_depart['origin',j] == 0
    model_change_rate_v2_PLARec.preprocessing_T0_rule = pe.Constraint(model_change_rate_v2_PLARec.j, rule = preprocessing_T0_rule)
      
    # the value of gamma an eta variables are no greater than 1
    # *** print ('t...')
    def ub_gamma_rule_orig(model_change_rate_v2_PLARec, i, j, k, u):
        if k in Trains[j]['containers']:
            return - model_change_rate_v2_PLARec.gamma[i,j,k,u] >= -1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.ub_gamma_rule_orig = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.u, rule = ub_gamma_rule_orig)
    
    def ub_eta_rule_orig(model_change_rate_v2_PLARec, i, j, k, v):
        if k in Trains[j]['containers']:
            return - model_change_rate_v2_PLARec.eta[i,j,k,v] >= -1
        else:
            return pe.Constraint.Skip
    model_change_rate_v2_PLARec.ub_eta_rule_orig = pe.Constraint(model_change_rate_v2_PLARec.i, model_change_rate_v2_PLARec.j, model_change_rate_v2_PLARec.k, model_change_rate_v2_PLARec.v, rule = ub_eta_rule_orig)

        
    # solve the model
    # print('Solving...')
    # solve by cplex:
    if mysolver == 'cplex':
        opt= pyomo.opt.SolverFactory("cplex")
        opt.options["mip_tolerances_mipgap"] = gap_op
        opt.options["timelimit"] = 900
        results=opt.solve(model_change_rate_v2_PLARec, tee=True, keepfiles=True)
        # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_change_rate_v2_PLARec, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_change_rate_v2_PLARec)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_change_rate_v2_PLARec, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # solve by gurobi:
    elif mysolver == 'gurobi':
        solver = SolverFactory('gurobi')
        solver.options['DualReductions'] = 0  # Ensure extreme ray is available
        solver.options['PreSolve'] = 0  # Disable preprocessing
        solver.options["LogToConsole"] = 0  # Disable Gurobi console output
        solver.options['MIPGap'] = gap_op  # 1% optimality gap
        solver.options['TimeLimit'] = time_limit_op
        # solver.options['Cuts'] = 3  # Aggressive cutting planes
        # solver.options['VarBranch'] = 2  # Strong branching
        results = solver.solve(model_change_rate_v2_PLARec, tee=True, keepfiles=True)
        results.write()
    
    time_end = time.time()
    time_model = time_end - time_start
    upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
    gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
    
    # calculate obj
    cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v2_PLARec.X[i].value for i in model_change_rate_v2_PLARec.i)
    cost_delay = sum(sum((model_change_rate_v2_PLARec.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v2_PLARec.j) for i in model_change_rate_v2_PLARec.i)
    obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
    cost_fix_weight = penalty_fix_cost * cost_fix
    cost_delay_weight = penalty_delay * cost_delay
    
    '''Record variables'''
    # print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    F = {}
    
    for i in Stations:
        X.update({i: model_change_rate_v2_PLARec.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        F.update({i: {}})
        for j in Trains:
            D[i].update({j: model_change_rate_v2_PLARec.D[i,j].value})
            T_arrive[i].update({j: model_change_rate_v2_PLARec.T_arrive[i,j].value})
            T_depart[i].update({j: model_change_rate_v2_PLARec.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            F[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_change_rate_v2_PLARec.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_change_rate_v2_PLARec.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_change_rate_v2_PLARec.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_change_rate_v2_PLARec.Z_s[i,j,k].value})
                T_c[i][j].update({k: model_change_rate_v2_PLARec.T_c[i,j,k].value})
                F[i][j].update({k: model_change_rate_v2_PLARec.F[i,j,k].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_change_rate_v2_PLARec.Y[j,k].value})
    
    beta = {}
    gamma = {}
    tau = {}
    eta = {}
    B = {}
    for i in Stations:
        beta.update({i: {}})
        gamma.update({i: {}})
        tau.update({i: {}})
        eta.update({i: {}})
        B.update({i: {}})
        for j in Trains:
            beta[i].update({j: {}})
            gamma[i].update({j: {}})
            tau[i].update({j: {}})
            eta[i].update({j: {}})
            B[i].update({j: {}})
            for k in Trains[j]['containers']:
                beta[i][j].update({k: {}})
                gamma[i][j].update({k: {}})
                tau[i][j].update({k: {}})
                eta[i][j].update({k: {}})
                B[i][j].update({k: model_change_rate_v2_PLARec.B[i,j,k].value})
                for u in Segments_SOC:
                    if u != max(Segments_SOC):
                        beta[i][j][k].update({u: model_change_rate_v2_PLARec.beta[i,j,k,u].value})
                    gamma[i][j][k].update({u: model_change_rate_v2_PLARec.gamma[i,j,k,u].value})
                for v in Segments_hour_charge:
                    if v != max(Segments_hour_charge):
                        tau[i][j][k].update({v: model_change_rate_v2_PLARec.tau[i,j,k,v].value})
                    eta[i][j][k].update({v: model_change_rate_v2_PLARec.eta[i,j,k,v].value})
    
    
    return obj, cost_fix_weight, cost_delay_weight, D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, F, beta, gamma, tau, eta, B, results, time_model, gap, upper_bound, lower_bound
    










'''This model is same as Model_change_rate_v2_PLARec except the following:
    1. In this model, all X variables are fixed.
    2. In this model, if a battery k is in train j, then Y[j,k]=1'''
# def Model_change_rate_v3_PLARec(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
#                                 M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, stations_deploy_list, \
#                                 gap_op, time_limit_op, mysolver):

#     time_start = time.time()
#     model_change_rate_v3_PLARec = pe.ConcreteModel()
#     # print ('Defining indices and sets...')
#     # Sets and indices
#     model_change_rate_v3_PLARec.i = pe.Set(initialize = set(Stations.keys()))
#     model_change_rate_v3_PLARec.i1 = pe.Set(initialize = set(Stations.keys()))
#     model_change_rate_v3_PLARec.i2 = pe.Set(initialize = set(Stations.keys()))
#     model_change_rate_v3_PLARec.j = pe.Set(initialize = set(Trains.keys()))
#     model_change_rate_v3_PLARec.k = pe.Set(initialize = set(Containers.keys()))
#     model_change_rate_v3_PLARec.k1 = pe.Set(initialize = set(Containers.keys()))
#     model_change_rate_v3_PLARec.k2 = pe.Set(initialize = set(Containers.keys()))
#     model_change_rate_v3_PLARec.u = pe.Set(initialize = set(Segments_SOC.keys()))
#     model_change_rate_v3_PLARec.v = pe.Set(initialize = set(Segments_hour_charge.keys()))
    
#     # Variables
#     # print ('Defining variables...')
#     model_change_rate_v3_PLARec.D = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
#     model_change_rate_v3_PLARec.S_arrive = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)
#     model_change_rate_v3_PLARec.S_depart = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)
#     model_change_rate_v3_PLARec.T_arrive = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
#     model_change_rate_v3_PLARec.T_depart = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
#     model_change_rate_v3_PLARec.X = pe.Var(model_change_rate_v3_PLARec.i, within = pe.Binary)
#     model_change_rate_v3_PLARec.Y = pe.Var(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
#     model_change_rate_v3_PLARec.Z_c = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
#     model_change_rate_v3_PLARec.Z_s = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
#     model_change_rate_v3_PLARec.T_c = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
#     # variables for linearization
#     model_change_rate_v3_PLARec.B = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
#     model_change_rate_v3_PLARec.F = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
#     model_change_rate_v3_PLARec.beta = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
#     model_change_rate_v3_PLARec.gamma = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
#     model_change_rate_v3_PLARec.tau = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, within = pe.Binary)  # in PLA, binary variables for T_c
#     model_change_rate_v3_PLARec.eta = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c


#     # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
#     # print ('Preprocessing') 
#     # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3_PLARec.
#     for j in model_change_rate_v3_PLARec.j:
#         for k in model_change_rate_v3_PLARec.k:
#             if k not in Trains[j]['containers']:
#                 model_change_rate_v3_PLARec.Y[j,k].fix(0)
#             else:
#                 # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
#                 model_change_rate_v3_PLARec.Y[j,k].fix(1)
#     # For each train j, the depart and arrival time at the origin are both zero.
#     for j in model_change_rate_v3_PLARec.j:
#         model_change_rate_v3_PLARec.T_arrive['origin',j].fix(0)
#         model_change_rate_v3_PLARec.T_depart['origin',j].fix(0)
#     # If a train is selected, then fix X=1. Otherwise fix X=0
#     for i in model_change_rate_v3_PLARec.i:
#         if i in stations_deploy_list:
#             model_change_rate_v3_PLARec.X[i].fix(1)
#         else:
#             model_change_rate_v3_PLARec.X[i].fix(0)
    
    
#     # objective function
#     # print ('Reading objective...')
#     def obj_rule(model_change_rate_v3_PLARec):
#         cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v3_PLARec.X[i] for i in model_change_rate_v3_PLARec.i)
#         cost_delay = sum(sum((model_change_rate_v3_PLARec.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v3_PLARec.j) for i in model_change_rate_v3_PLARec.i)
#         obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
#         return obj  
#     model_change_rate_v3_PLARec.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
#     # constraints
#     # define delay time for train j in station i
#     # print ('a...')
#     def delay_define_rule(model_change_rate_v3_PLARec, i, j):
#         return model_change_rate_v3_PLARec.D[i,j] >= model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j]
#     model_change_rate_v3_PLARec.delay_define_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = delay_define_rule)
    
#     # If station is not deployed, then batteries can be neither swapped nor charged in station i
#     # print ('b...')
#     def deploy_swap_charge_rule(model_change_rate_v3_PLARec, i):
#         return sum(sum((model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]) \
#                         for k in model_change_rate_v3_PLARec.k if k in Trains[j]['containers']) \
#                     for j in model_change_rate_v3_PLARec.j) \
#                 <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v3_PLARec.X[i]
#     model_change_rate_v3_PLARec.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v3_PLARec.i, rule = deploy_swap_charge_rule)        
    
#     # If station is deployed, then at least one battery must be swapped nor charged in station i
#     # print ('opt2...')
#     def deploy_swap_charge_must_rule_orig_feas(model_change_rate_v3_PLARec, i):
#         return sum(sum((model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]) \
#                         for k in model_change_rate_v3_PLARec.k if k in Trains[j]['containers']) \
#                     for j in model_change_rate_v3_PLARec.j) \
#                 >= model_change_rate_v3_PLARec.X[i]
#     model_change_rate_v3_PLARec.deploy_swap_charge_must_rule_orig_feas = pe.Constraint(model_change_rate_v3_PLARec.i, rule = deploy_swap_charge_must_rule_orig_feas)   

#     # Train j cannot both swap and charge batteries in station i.
#     # print ('c...')
#     def noboth_swap_charge_rule(model_change_rate_v3_PLARec, i, j, k1, k2):
#         if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.Z_s[i,j,k1] + model_change_rate_v3_PLARec.Z_c[i,j,k2] <= 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k1, model_change_rate_v3_PLARec.k2, rule = noboth_swap_charge_rule)        
    
#     # Train j cannot depart station i until passenger trains have passed.
#     # print ('d...')
#     def wait_passenger_rule(model_change_rate_v3_PLARec, i, j):
#         return model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
#     model_change_rate_v3_PLARec.wait_passenger_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = wait_passenger_rule)        
    
#     # Upper bound on the number of batteries that train j can carry. 
#     # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
#     # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
#     # print ('e0...')
#     def consecutive_battery_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             kid = int(((k.split('container '))[1].split(' in')[0]))
#             if kid < len(Trains[j]['containers']):
#                 return sum(model_change_rate_v3_PLARec.Y[j,kp] \
#                             for kp in model_change_rate_v3_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
#                         <= sum(1 for kp in model_change_rate_v3_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
#                           * model_change_rate_v3_PLARec.Y[j,k]
#             else:
#                 return pe.Constraint.Skip
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.consecutive_battery_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = consecutive_battery_rule)  
    
#     # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
#     # print ('e...')
#     def max_number_batteries_station_rule(model_change_rate_v3_PLARec, i, j):
#         return sum(model_change_rate_v3_PLARec.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
#     model_change_rate_v3_PLARec.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = max_number_batteries_station_rule)  
    
#     # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
#     # print ('f...')
#     def max_number_chargers_station_rule(model_change_rate_v3_PLARec, i, j):
#         return sum(model_change_rate_v3_PLARec.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
#     model_change_rate_v3_PLARec.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = max_number_chargers_station_rule)  
    
#     # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
#     # print ('g...')
#     def power_rule(model_change_rate_v3_PLARec, i1, i2, j):
#         if i2 == Stations[i1]['station_after']:
#             return sum(model_change_rate_v3_PLARec.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v3_PLARec.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.power_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, rule = power_rule)          
    
#     # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
#     # print ('g1...')
#     def battery_sequential_rule_linear_a(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             k_index = Trains[j]['containers'].index(k)
#             if k_index < len(Trains[j]['containers'])-1:
#                 return model_change_rate_v3_PLARec.S_arrive[i,j,k]  -  M * model_change_rate_v3_PLARec.B[i,j,k]  <=  0
#             else:
#                 return pe.Constraint.Skip
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.battery_sequential_rule_linear_a = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = battery_sequential_rule_linear_a)          

#     # print ('g2...')
#     def battery_sequential_rule_linear_b(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             k_index = Trains[j]['containers'].index(k)
#             if k_index < len(Trains[j]['containers'])-1:
#                 k_next = Trains[j]['containers'][k_index+1]
#                 return (1 - model_change_rate_v3_PLARec.S_arrive[i,j,k_next])  +  M * model_change_rate_v3_PLARec.B[i,j,k]  <=  M
#             else:
#                 return pe.Constraint.Skip
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.battery_sequential_rule_linear_b = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = battery_sequential_rule_linear_b)          
    
#     # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
#     # print ('h...')
#     def soc_increase_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] - model_change_rate_v3_PLARec.S_arrive[i,j,k] >= 0
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_increase_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_increase_rule)          
    
#     # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
#     # print ('i...')
#     def swap_full_soc_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] >= model_change_rate_v3_PLARec.Z_s[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.swap_full_soc_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = swap_full_soc_rule)     
    
#     # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
#     # print ('i1...')
#     def soc_depart_arrive_between_stations_rule(model_change_rate_v3_PLARec, i1, i2, j, k):
#         if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
#             return model_change_rate_v3_PLARec.S_arrive[i2,j,k] - model_change_rate_v3_PLARec.S_depart[i1,j,k] <= 0
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_depart_arrive_between_stations_rule)     

#     # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
#     # print ('j...')
#     def time_battery_swap_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v3_PLARec.Z_s[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.time_battery_swap_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = time_battery_swap_rule)     
    
#     # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
#     # print ('k0...')
#     def soc_time_charge_rule_PLA_Zs(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] \
#                     + model_change_rate_v3_PLARec.F[i,j,k] \
#                     - 2 * model_change_rate_v3_PLARec.Z_s[i,j,k] \
#                     <= 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_Zs)        
    
#     # print ('k1...')
#     def soc_time_charge_rule_PLA_Zc(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] \
#                     + model_change_rate_v3_PLARec.F[i,j,k] \
#                     + epsilon * model_change_rate_v3_PLARec.Z_c[i,j,k] \
#                     >= 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_Zc)        

#     # print ('k2...')
#     def soc_time_charge_rule_PLA_beta1(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return sum(model_change_rate_v3_PLARec.beta[i,j,k,u] for u in model_change_rate_v3_PLARec.u if u != max(model_change_rate_v3_PLARec.u)) == 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_beta1)        

#     # print ('k3...')
#     def soc_time_charge_rule_PLA_gamma1(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return sum(model_change_rate_v3_PLARec.gamma[i,j,k,u] for u in model_change_rate_v3_PLARec.u) == 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_gamma1)        

#     # print ('k4...')
#     def soc_time_charge_rule_PLA_tau1(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return sum(model_change_rate_v3_PLARec.tau[i,j,k,v] for v in model_change_rate_v3_PLARec.v if v != max(model_change_rate_v3_PLARec.v)) == 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_tau1)        

#     # print ('k5...')
#     def soc_time_charge_rule_PLA_gamma_Sarrive(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return sum((Segments_SOC[u]['SOC'] * model_change_rate_v3_PLARec.gamma[i,j,k,u]) \
#                         for u in model_change_rate_v3_PLARec.u) \
#                     == model_change_rate_v3_PLARec.S_arrive[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

#     # print ('k6...')
#     def soc_time_charge_rule_PLA_tau_eta_Tc(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return sum((Segments_hour_charge[v]['hour'] * model_change_rate_v3_PLARec.tau[i,j,k,v] \
#                         + (Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_change_rate_v3_PLARec.eta[i,j,k,v]) \
#                         for v in model_change_rate_v3_PLARec.v if v != max(model_change_rate_v3_PLARec.v)) \
#                     == model_change_rate_v3_PLARec.T_c[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

#     # print ('k7...')
#     def soc_time_charge_rule_PLA_gamma_beta(model_change_rate_v3_PLARec, i, j, k, u):
#         if k in Trains[j]['containers'] and u != min(model_change_rate_v3_PLARec.u) and u != max(model_change_rate_v3_PLARec.u):
#             return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u-1] + model_change_rate_v3_PLARec.beta[i,j,k,u]
#         elif k in Trains[j]['containers'] and u == min(model_change_rate_v3_PLARec.u):
#             return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u]
#         elif k in Trains[j]['containers'] and u == max(model_change_rate_v3_PLARec.u):
#             return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u-1]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

#     # print ('k8...')
#     def soc_time_charge_rule_PLA_eta_tau(model_change_rate_v3_PLARec, i, j, k, v):
#         if k in Trains[j]['containers'] and v != min(model_change_rate_v3_PLARec.v) and v != max(model_change_rate_v3_PLARec.v):
#             return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v-1] + model_change_rate_v3_PLARec.tau[i,j,k,v]
#         elif k in Trains[j]['containers'] and v == min(model_change_rate_v3_PLARec.v):
#             return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v]
#         elif k in Trains[j]['containers'] and v == max(model_change_rate_v3_PLARec.v):
#             return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v-1]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau)        

#     # print ('k4...')
#     def soc_time_charge_rule_PLA_eta_tau1(model_change_rate_v3_PLARec, i, j, k, v):
#         if k in Trains[j]['containers'] and v != max(model_change_rate_v3_PLARec.v):
#             return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

#     # print ('k9...')
#     def soc_time_charge_rule_PLA_F_leq(model_change_rate_v3_PLARec, i, j, k, u, v):
#         if k in Trains[j]['containers'] and u != max(model_change_rate_v3_PLARec.u) and v != max(model_change_rate_v3_PLARec.v):
#             s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
#             s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
#             g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
#             g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
#             g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
#             g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
#             w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
#             return model_change_rate_v3_PLARec.F[i,j,k] \
#                     <= sum((model_change_rate_v3_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
#                           for w in model_change_rate_v3_PLARec.u) \
#                       + w * model_change_rate_v3_PLARec.eta[i,j,k,v] \
#                       + M * (2 - model_change_rate_v3_PLARec.tau[i,j,k,v] - model_change_rate_v3_PLARec.beta[i,j,k,u])
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_F_leq)        

#     # print ('k10...')
#     def soc_time_charge_rule_PLA_F_geq(model_change_rate_v3_PLARec, i, j, k, u, v):
#         if k in Trains[j]['containers'] and u != max(model_change_rate_v3_PLARec.u) and v != max(model_change_rate_v3_PLARec.v):
#             s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
#             s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
#             g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
#             g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
#             g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
#             g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
#             w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
#             return model_change_rate_v3_PLARec.F[i,j,k] \
#                     >= sum((model_change_rate_v3_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
#                           for w in model_change_rate_v3_PLARec.u) \
#                       + w * model_change_rate_v3_PLARec.eta[i,j,k,v] \
#                       - M * (2 - model_change_rate_v3_PLARec.tau[i,j,k,v] - model_change_rate_v3_PLARec.beta[i,j,k,u])
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_F_geq)        
    
#     # print ('l...')
#     # def soc_time_charge_rule_b(model_change_rate_v3_PLARec, i, j, k):
#     #     if k in Trains[j]['containers']:
#     #         return rate_charge * model_change_rate_v3_PLARec.T_c[i,j,k] <= 1 - model_change_rate_v3_PLARec.S_arrive[i,j,k]
#     #     else:
#     #         return pe.Constraint.Skip
#     # model_change_rate_v3_PLARec.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_b)
    
#     # print ('m...')
#     def soc_time_charge_rule_c(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.T_c[i,j,k] <= model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_c)
    
#     # print ('n...')
#     def soc_time_charge_rule_d(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.T_c[i,j,k] <= M * model_change_rate_v3_PLARec.Z_c[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_d)
    
#     # print ('n1...')
#     def soc_time_charge_rule_e(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.Z_c[i,j,k] <= M * model_change_rate_v3_PLARec.T_c[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_e)
    
#     # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
#     # print ('o...')
#     def no_swap_charge_soc_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] - model_change_rate_v3_PLARec.S_arrive[i,j,k] <= model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = no_swap_charge_soc_rule)
    
#     # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
#     # print ('o2...')
#     def soc_depart_origin_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart['origin',j,k] >= model_change_rate_v3_PLARec.Y[j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_depart_origin_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_depart_origin_rule)
    
#     # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
#     # print ('o3...')
#     def soc_arrive_origin_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_arrive['origin',j,k] >= model_change_rate_v3_PLARec.Y[j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_arrive_origin_rule)
    
#     # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
#     # print ('o4...')
#     def origin_no_chargeswap_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.Z_c['origin',j,k] + model_change_rate_v3_PLARec.Z_s['origin',j,k] <= 2 * model_change_rate_v3_PLARec.Y[j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = origin_no_chargeswap_rule)
    
#     # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
#     # print ('o5...')
#     def T_traveltime_rule(model_change_rate_v3_PLARec, i1, i2, j):
#         if i2 == Stations[i1]['station_after']:
#             return model_change_rate_v3_PLARec.T_arrive[i2,j] == model_change_rate_v3_PLARec.T_depart[i1,j] + TravelTime_train[j][i1][i2]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.T_traveltime_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, rule = T_traveltime_rule)
    
#     # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
#     # print ('o6...')
#     def nobattery_nochargeswap_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             return sum(model_change_rate_v3_PLARec.Z_c[i,j,k] for i in model_change_rate_v3_PLARec.i) + sum(model_change_rate_v3_PLARec.Z_s[i,j,k] for i in model_change_rate_v3_PLARec.i) <= 2*len(Stations) * model_change_rate_v3_PLARec.Y[j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = nobattery_nochargeswap_rule)
    
#     # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
#     # print ('o7...')
#     def nobattery_zerosoc_rule(model_change_rate_v3_PLARec, j, k):
#         if k in Trains[j]['containers']:
#             return sum(model_change_rate_v3_PLARec.S_arrive[i,j,k] for i in model_change_rate_v3_PLARec.i) + sum(model_change_rate_v3_PLARec.S_depart[i,j,k] for i in model_change_rate_v3_PLARec.i) <= 2*len(Stations) * model_change_rate_v3_PLARec.Y[j,k]
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = nobattery_zerosoc_rule)
    
#     # The SOC of each battery must not exceed 100%.
#     # print ('p...')
#     def ub_soc_arrive_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_arrive[i,j,k] <= 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = ub_soc_arrive_rule)
    
#     # print ('q...')
#     def ub_soc_depart_rule(model_change_rate_v3_PLARec, i, j, k):
#         if k in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.S_depart[i,j,k] <= 1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.ub_soc_depart_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = ub_soc_depart_rule)
    
#     # print ('r...')
#     # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3_PLARec.
#     def preprocessing_Y0_rule(model_change_rate_v3_PLARec, j, k):
#         if k not in Trains[j]['containers']:
#             return model_change_rate_v3_PLARec.Y[j,k] == 0
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.preprocessing_Y0_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = preprocessing_Y0_rule)
       
#     # print ('s...')
#     # For each train j, the depart and arrival time at the origin are both zero.
#     def preprocessing_T0_rule(model_change_rate_v3_PLARec, j):
#         return model_change_rate_v3_PLARec.T_arrive['origin',j] + model_change_rate_v3_PLARec.T_depart['origin',j] == 0
#     model_change_rate_v3_PLARec.preprocessing_T0_rule = pe.Constraint(model_change_rate_v3_PLARec.j, rule = preprocessing_T0_rule)
    
#     # the value of gamma an eta variables are no greater than 1
#     # *** print ('t...')
#     def ub_gamma_rule_orig(model_change_rate_v3_PLARec, i, j, k, u):
#         if k in Trains[j]['containers']:
#             return - model_change_rate_v3_PLARec.gamma[i,j,k,u] >= -1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.ub_gamma_rule_orig = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, rule = ub_gamma_rule_orig)
    
#     def ub_eta_rule_orig(model_change_rate_v3_PLARec, i, j, k, v):
#         if k in Trains[j]['containers']:
#             return - model_change_rate_v3_PLARec.eta[i,j,k,v] >= -1
#         else:
#             return pe.Constraint.Skip
#     model_change_rate_v3_PLARec.ub_eta_rule_orig = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = ub_eta_rule_orig)

    
        
#     # solve the model
#     print('Solving...')
#     # solve by cplex:
#     if mysolver == 'cplex':
#         opt= pyomo.opt.SolverFactory("cplex")
#         opt.options["mip_tolerances_mipgap"] = gap_op
#         opt.options["timelimit"] = 900
#         results=opt.solve(model_change_rate_v3_PLARec, tee=False, keepfiles=False)
#         # results.write()ange_rate_v3_PLARec, tee=False, keepfiles=False)
        
#     # solve by gurobi:
#     elif mysolver == 'gurobi':
#         solver = SolverFactory('gurobi')
#         # solver.options['DualReductions'] = 0  # Ensure extreme ray is available
#         solver.options['PreSolve'] = 2  # Aggressive preprocessing
#         solver.options['ScaleFlag'] = 2  # Automatic scaling
#         solver.options["LogToConsole"] = 0  # Disable Gurobi console output
#         solver.options['MIPGap'] = gap_op  # 1% optimality gap
#         solver.options['TimeLimit'] = time_limit_op # set time limit
#         solver.options['FeasibilityTol'] = 1e-9  # Stricter tolerance
#         solver.options['IntFeasTol'] = 1e-9  # Stricter tolerance for integer variables
#         results = solver.solve(model_change_rate_v3_PLARec, tee=False, keepfiles=False)
#         # results.write()
        
#     # # solve by ipopt: treats the variables as continuous
#     # solver = SolverFactory('ipopt')
#     # #solver.options['max_iter']= 10000
#     # results= solver.solve(model_change_rate_v3_PLARec, tee=True)    
#     # results.write()

#     # opt=SolverFactory('apopt.py')
#     # results=opt.solve(model_change_rate_v3_PLARec)
#     # results.write()
#     # instance.load(results)
    
#     # solve by Couenne: an open solver for mixed integer nonlinear problems
#     # # import os
#     # # from pyomo.environ import *
#     # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
#     # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
#     # opt = SolverFactory("couenne") 
#     # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
#     # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
#     # # opt.options['logFile'] = 'couenne_log.txt'
#     # results=opt.solve(model_change_rate_v3_PLARec, timelimit=300, logfile = 'mylog.txt', tee=True)
#     # results.write()
    
    
#     time_end = time.time()
#     time_model = time_end - time_start
#     # gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
    
    
    
#     '''Record variables'''
#     # print ('Recording variables...')
#     D = {}
#     S_arrive = {}
#     S_depart = {}
#     T_arrive = {}
#     T_depart = {}
#     T_c = {}
#     X = {}
#     Y = {}
#     Z_c = {}
#     Z_s = {}
#     F = {}
    
#     for i in Stations:
#         X.update({i: model_change_rate_v3_PLARec.X[i].value})
#         D.update({i: {}})
#         S_arrive.update({i: {}})
#         S_depart.update({i: {}})
#         T_arrive.update({i: {}})
#         T_depart.update({i: {}})
#         T_c.update({i: {}})
#         Z_c.update({i: {}})
#         Z_s.update({i: {}})
#         F.update({i: {}})
#         for j in Trains:
#             D[i].update({j: model_change_rate_v3_PLARec.D[i,j].value})
#             T_arrive[i].update({j: model_change_rate_v3_PLARec.T_arrive[i,j].value})
#             T_depart[i].update({j: model_change_rate_v3_PLARec.T_depart[i,j].value})
#             S_arrive[i].update({j: {}})
#             S_depart[i].update({j: {}})
#             Z_c[i].update({j: {}})
#             Z_s[i].update({j: {}})
#             T_c[i].update({j: {}})
#             F[i].update({j: {}})
#             for k in Trains[j]['containers']:
#                 S_arrive[i][j].update({k: model_change_rate_v3_PLARec.S_arrive[i,j,k].value})
#                 S_depart[i][j].update({k: model_change_rate_v3_PLARec.S_depart[i,j,k].value})
#                 Z_c[i][j].update({k: model_change_rate_v3_PLARec.Z_c[i,j,k].value})
#                 Z_s[i][j].update({k: model_change_rate_v3_PLARec.Z_s[i,j,k].value})
#                 T_c[i][j].update({k: model_change_rate_v3_PLARec.T_c[i,j,k].value})
#                 F[i][j].update({k: model_change_rate_v3_PLARec.F[i,j,k].value})
#     for j in Trains:
#         Y.update({j: {}})
#         for k in Trains[j]['containers']:
#             Y[j].update({k: model_change_rate_v3_PLARec.Y[j,k].value})
    
#     beta = {}
#     gamma = {}
#     tau = {}
#     eta = {}
#     B = {}
#     for i in Stations:
#         beta.update({i: {}})
#         gamma.update({i: {}})
#         tau.update({i: {}})
#         eta.update({i: {}})
#         B.update({i: {}})
#         for j in Trains:
#             beta[i].update({j: {}})
#             gamma[i].update({j: {}})
#             tau[i].update({j: {}})
#             eta[i].update({j: {}})
#             B[i].update({j: {}})
#             for k in Trains[j]['containers']:
#                 beta[i][j].update({k: {}})
#                 gamma[i][j].update({k: {}})
#                 tau[i][j].update({k: {}})
#                 eta[i][j].update({k: {}})
#                 B[i][j].update({k: model_change_rate_v3_PLARec.B[i,j,k].value})
#                 for u in Segments_SOC:
#                     if u != max(Segments_SOC):
#                         beta[i][j][k].update({u: model_change_rate_v3_PLARec.beta[i,j,k,u].value})
#                     gamma[i][j][k].update({u: model_change_rate_v3_PLARec.gamma[i,j,k,u].value})
#                 for v in Segments_hour_charge:
#                     if v != max(Segments_hour_charge):
#                         tau[i][j][k].update({v: model_change_rate_v3_PLARec.tau[i,j,k,v].value})
#                     eta[i][j][k].update({v: model_change_rate_v3_PLARec.eta[i,j,k,v].value})
    
    
#     return D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, F, beta, gamma, tau, eta, B, results, time_model



def Model_change_rate_v3_PLARec(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                                stations_deploy_list, M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, gap_op, time_limit_op, mysolver):

    time_start = time.time()
    model_change_rate_v3_PLARec = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v3_PLARec.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v3_PLARec.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v3_PLARec.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_change_rate_v3_PLARec.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_change_rate_v3_PLARec.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v3_PLARec.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v3_PLARec.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_change_rate_v3_PLARec.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_change_rate_v3_PLARec.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    
    # Variables
    # print ('Defining variables...')
    model_change_rate_v3_PLARec.D = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v3_PLARec.S_arrive = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)
    model_change_rate_v3_PLARec.S_depart = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)
    model_change_rate_v3_PLARec.T_arrive = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v3_PLARec.T_depart = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v3_PLARec.X = pe.Var(model_change_rate_v3_PLARec.i, within = pe.Binary)
    model_change_rate_v3_PLARec.Y = pe.Var(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
    model_change_rate_v3_PLARec.Z_c = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
    model_change_rate_v3_PLARec.Z_s = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)
    model_change_rate_v3_PLARec.T_c = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    # variables for linearization
    model_change_rate_v3_PLARec.B = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
    model_change_rate_v3_PLARec.F = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_change_rate_v3_PLARec.beta = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
    model_change_rate_v3_PLARec.gamma = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
    model_change_rate_v3_PLARec.tau = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, within = pe.Binary)  # in PLA, binary variables for T_c
    model_change_rate_v3_PLARec.eta = pe.Var(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c


    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3_PLARec.
    for j in model_change_rate_v3_PLARec.j:
        for k in model_change_rate_v3_PLARec.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v3_PLARec.Y[j,k].fix(0)
            else:
                # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
                model_change_rate_v3_PLARec.Y[j,k].fix(1)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v3_PLARec.j:
        model_change_rate_v3_PLARec.T_arrive['origin',j].fix(0)
        model_change_rate_v3_PLARec.T_depart['origin',j].fix(0)
    
    for i in model_change_rate_v3_PLARec.i:
        if i in stations_deploy_list:
            model_change_rate_v3_PLARec.X[i].fix(1)
        else:
            model_change_rate_v3_PLARec.X[i].fix(0)
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_change_rate_v3_PLARec):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v3_PLARec.X[i] for i in model_change_rate_v3_PLARec.i)
        cost_delay = sum(sum((model_change_rate_v3_PLARec.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v3_PLARec.j) for i in model_change_rate_v3_PLARec.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v3_PLARec.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_change_rate_v3_PLARec, i, j):
        return model_change_rate_v3_PLARec.D[i,j] >= model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j]
    model_change_rate_v3_PLARec.delay_define_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v3_PLARec, i):
        return sum(sum((model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]) \
                       for k in model_change_rate_v3_PLARec.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v3_PLARec.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v3_PLARec.X[i]
    model_change_rate_v3_PLARec.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v3_PLARec.i, rule = deploy_swap_charge_rule)        
    
    # # If station is deployed, then at least one battery must be swapped nor charged in station i
    # print ('opt2...')
    def deploy_swap_charge_must_rule_orig_feas(model_change_rate_v3_PLARec, i):
        return sum(sum((model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]) \
                        for k in model_change_rate_v3_PLARec.k if k in Trains[j]['containers']) \
                    for j in model_change_rate_v3_PLARec.j) \
                >= model_change_rate_v3_PLARec.X[i]
    model_change_rate_v3_PLARec.deploy_swap_charge_must_rule_orig_feas = pe.Constraint(model_change_rate_v3_PLARec.i, rule = deploy_swap_charge_must_rule_orig_feas)   

    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v3_PLARec, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.Z_s[i,j,k1] + model_change_rate_v3_PLARec.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k1, model_change_rate_v3_PLARec.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_change_rate_v3_PLARec, i, j):
        return model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v3_PLARec.wait_passenger_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            if kid < len(Trains[j]['containers']):
                return sum(model_change_rate_v3_PLARec.Y[j,kp] \
                           for kp in model_change_rate_v3_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                       <= sum(1 for kp in model_change_rate_v3_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                          * model_change_rate_v3_PLARec.Y[j,k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.consecutive_battery_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v3_PLARec, i, j):
        return sum(model_change_rate_v3_PLARec.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v3_PLARec.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v3_PLARec, i, j):
        return sum(model_change_rate_v3_PLARec.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v3_PLARec.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_change_rate_v3_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v3_PLARec.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v3_PLARec.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.power_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule_linear_a(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return model_change_rate_v3_PLARec.S_arrive[i,j,k]  -  M * model_change_rate_v3_PLARec.B[i,j,k]  <=  0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.battery_sequential_rule_linear_a = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    def battery_sequential_rule_linear_b(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return (1 - model_change_rate_v3_PLARec.S_arrive[i,j,k_next])  +  M * model_change_rate_v3_PLARec.B[i,j,k] <=  M
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.battery_sequential_rule_linear_b = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] - model_change_rate_v3_PLARec.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_increase_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] >= model_change_rate_v3_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.swap_full_soc_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_change_rate_v3_PLARec, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return model_change_rate_v3_PLARec.S_arrive[i2,j,k] - model_change_rate_v3_PLARec.S_depart[i1,j,k] <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v3_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.time_battery_swap_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_PLA_Zs(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v3_PLARec.F[i,j,k] \
                   - M * model_change_rate_v3_PLARec.Z_s[i,j,k] \
                   <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_PLA_Zc(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v3_PLARec.F[i,j,k] \
                   + epsilon * model_change_rate_v3_PLARec.Z_c[i,j,k] \
                   >= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_Zc)        

    # print ('k2...')
    def soc_time_charge_rule_PLA_beta1(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3_PLARec.beta[i,j,k,u] for u in model_change_rate_v3_PLARec.u if u != max(model_change_rate_v3_PLARec.u)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    def soc_time_charge_rule_PLA_gamma1(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3_PLARec.gamma[i,j,k,u] for u in model_change_rate_v3_PLARec.u) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_tau1(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3_PLARec.tau[i,j,k,v] for v in model_change_rate_v3_PLARec.v if v != max(model_change_rate_v3_PLARec.v)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_change_rate_v3_PLARec.gamma[i,j,k,u]) \
                       for u in model_change_rate_v3_PLARec.u) \
                   == model_change_rate_v3_PLARec.S_arrive[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_hour_charge[v]['hour'] * model_change_rate_v3_PLARec.tau[i,j,k,v] \
                        + (Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_change_rate_v3_PLARec.eta[i,j,k,v]) \
                       for v in model_change_rate_v3_PLARec.v if v != max(model_change_rate_v3_PLARec.v)) \
                   == model_change_rate_v3_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta(model_change_rate_v3_PLARec, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_change_rate_v3_PLARec.u) and u != max(model_change_rate_v3_PLARec.u):
            return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u-1] + model_change_rate_v3_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == min(model_change_rate_v3_PLARec.u):
            return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == max(model_change_rate_v3_PLARec.u):
            return model_change_rate_v3_PLARec.gamma[i,j,k,u] <= model_change_rate_v3_PLARec.beta[i,j,k,u-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau(model_change_rate_v3_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_change_rate_v3_PLARec.v) and v != max(model_change_rate_v3_PLARec.v):
            return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v-1] + model_change_rate_v3_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == min(model_change_rate_v3_PLARec.v):
            return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == max(model_change_rate_v3_PLARec.v):
            return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1(model_change_rate_v3_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_change_rate_v3_PLARec.v) and v != max(model_change_rate_v3_PLARec.v):
            return model_change_rate_v3_PLARec.eta[i,j,k,v] <= model_change_rate_v3_PLARec.tau[i,j,k,v]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    def soc_time_charge_rule_PLA_F_leq(model_change_rate_v3_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v3_PLARec.u) and v != max(model_change_rate_v3_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v3_PLARec.F[i,j,k] \
                   <= sum((model_change_rate_v3_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v3_PLARec.u) \
                      + w * model_change_rate_v3_PLARec.eta[i,j,k,v] \
                      + M * (2 - model_change_rate_v3_PLARec.tau[i,j,k,v] - model_change_rate_v3_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    def soc_time_charge_rule_PLA_F_geq(model_change_rate_v3_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v3_PLARec.u) and v != max(model_change_rate_v3_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v3_PLARec.F[i,j,k] \
                   >= sum((model_change_rate_v3_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v3_PLARec.u) \
                      + w * model_change_rate_v3_PLARec.eta[i,j,k,v] \
                      - M * (2 - model_change_rate_v3_PLARec.tau[i,j,k,v] - model_change_rate_v3_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, model_change_rate_v3_PLARec.v, rule = soc_time_charge_rule_PLA_F_geq)        

    # print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v3_PLARec, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v3_PLARec.T_c[i,j,k] <= 1 - model_change_rate_v3_PLARec.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v3_PLARec.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.T_c[i,j,k] <= model_change_rate_v3_PLARec.T_depart[i,j] - model_change_rate_v3_PLARec.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.T_c[i,j,k] <= M * model_change_rate_v3_PLARec.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.Z_c[i,j,k] <= M * model_change_rate_v3_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] - model_change_rate_v3_PLARec.S_arrive[i,j,k] <= model_change_rate_v3_PLARec.Z_c[i,j,k] + model_change_rate_v3_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart['origin',j,k] >= model_change_rate_v3_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_depart_origin_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_arrive['origin',j,k] >= model_change_rate_v3_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.Z_c['origin',j,k] + model_change_rate_v3_PLARec.Z_s['origin',j,k] <= 2 * model_change_rate_v3_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_change_rate_v3_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v3_PLARec.T_arrive[i2,j] == model_change_rate_v3_PLARec.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.T_traveltime_rule = pe.Constraint(model_change_rate_v3_PLARec.i1, model_change_rate_v3_PLARec.i2, model_change_rate_v3_PLARec.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3_PLARec.Z_c[i,j,k] for i in model_change_rate_v3_PLARec.i) + sum(model_change_rate_v3_PLARec.Z_s[i,j,k] for i in model_change_rate_v3_PLARec.i) <= 2*len(Stations) * model_change_rate_v3_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v3_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v3_PLARec.S_arrive[i,j,k] for i in model_change_rate_v3_PLARec.i) + sum(model_change_rate_v3_PLARec.S_depart[i,j,k] for i in model_change_rate_v3_PLARec.i) <= 2*len(Stations) * model_change_rate_v3_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_change_rate_v3_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.ub_soc_depart_rule = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v3_PLARec.
    def preprocessing_Y0_rule(model_change_rate_v3_PLARec, j, k):
        if k not in Trains[j]['containers']:
            return model_change_rate_v3_PLARec.Y[j,k] == 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.preprocessing_Y0_rule = pe.Constraint(model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_change_rate_v3_PLARec, j):
        return model_change_rate_v3_PLARec.T_arrive['origin',j] + model_change_rate_v3_PLARec.T_depart['origin',j] == 0
    model_change_rate_v3_PLARec.preprocessing_T0_rule = pe.Constraint(model_change_rate_v3_PLARec.j, rule = preprocessing_T0_rule)
      
    # the value of gamma an eta variables are no greater than 1
    # *** print ('t...')
    def ub_gamma_rule_orig(model_change_rate_v3_PLARec, i, j, k, u):
        if k in Trains[j]['containers']:
            return - model_change_rate_v3_PLARec.gamma[i,j,k,u] >= -1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.ub_gamma_rule_orig = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.u, rule = ub_gamma_rule_orig)
    
    def ub_eta_rule_orig(model_change_rate_v3_PLARec, i, j, k, v):
        if k in Trains[j]['containers']:
            return - model_change_rate_v3_PLARec.eta[i,j,k,v] >= -1
        else:
            return pe.Constraint.Skip
    model_change_rate_v3_PLARec.ub_eta_rule_orig = pe.Constraint(model_change_rate_v3_PLARec.i, model_change_rate_v3_PLARec.j, model_change_rate_v3_PLARec.k, model_change_rate_v3_PLARec.v, rule = ub_eta_rule_orig)

        
    # solve the model
    # print('Solving...')
    # solve by cplex:
    if mysolver == 'cplex':
        opt= pyomo.opt.SolverFactory("cplex")
        opt.options["mip_tolerances_mipgap"] = gap_op
        opt.options["timelimit"] = 900
        results=opt.solve(model_change_rate_v3_PLARec, tee=True, keepfiles=True)
        # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_change_rate_v3_PLARec, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_change_rate_v3_PLARec)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_change_rate_v3_PLARec, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # solve by gurobi:
    elif mysolver == 'gurobi':
        solver = SolverFactory('gurobi')
        solver.options['DualReductions'] = 0  # Ensure extreme ray is available
        solver.options['PreSolve'] = 0  # Disable preprocessing
        solver.options["LogToConsole"] = 0  # Disable Gurobi console output
        solver.options['MIPGap'] = gap_op  # 1% optimality gap
        solver.options['TimeLimit'] = time_limit_op
        # solver.options['Cuts'] = 3  # Aggressive cutting planes
        # solver.options['VarBranch'] = 2  # Strong branching
        results = solver.solve(model_change_rate_v3_PLARec, tee=True, keepfiles=True)
        results.write()
    
    time_end = time.time()
    time_model = time_end - time_start
    
    
    '''Record variables'''
    # print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    F = {}
    beta = {}
    gamma = {}
    tau = {}
    eta = {}
    B = {}
    obj, cost_fix_weight, cost_delay_weight = 'None', 'None', 'None'
    
    
    if results.solver.termination_condition != TerminationCondition.infeasible:
    
        # calculate obj
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v3_PLARec.X[i].value for i in model_change_rate_v3_PLARec.i)
        cost_delay = sum(sum((model_change_rate_v3_PLARec.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v3_PLARec.j) for i in model_change_rate_v3_PLARec.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        cost_fix_weight = penalty_fix_cost * cost_fix
        cost_delay_weight = penalty_delay * cost_delay
        
        for i in Stations:
            X.update({i: model_change_rate_v3_PLARec.X[i].value})
            D.update({i: {}})
            S_arrive.update({i: {}})
            S_depart.update({i: {}})
            T_arrive.update({i: {}})
            T_depart.update({i: {}})
            T_c.update({i: {}})
            Z_c.update({i: {}})
            Z_s.update({i: {}})
            F.update({i: {}})
            for j in Trains:
                D[i].update({j: model_change_rate_v3_PLARec.D[i,j].value})
                T_arrive[i].update({j: model_change_rate_v3_PLARec.T_arrive[i,j].value})
                T_depart[i].update({j: model_change_rate_v3_PLARec.T_depart[i,j].value})
                S_arrive[i].update({j: {}})
                S_depart[i].update({j: {}})
                Z_c[i].update({j: {}})
                Z_s[i].update({j: {}})
                T_c[i].update({j: {}})
                F[i].update({j: {}})
                for k in Trains[j]['containers']:
                    S_arrive[i][j].update({k: model_change_rate_v3_PLARec.S_arrive[i,j,k].value})
                    S_depart[i][j].update({k: model_change_rate_v3_PLARec.S_depart[i,j,k].value})
                    Z_c[i][j].update({k: model_change_rate_v3_PLARec.Z_c[i,j,k].value})
                    Z_s[i][j].update({k: model_change_rate_v3_PLARec.Z_s[i,j,k].value})
                    T_c[i][j].update({k: model_change_rate_v3_PLARec.T_c[i,j,k].value})
                    F[i][j].update({k: model_change_rate_v3_PLARec.F[i,j,k].value})
        for j in Trains:
            Y.update({j: {}})
            for k in Trains[j]['containers']:
                Y[j].update({k: model_change_rate_v3_PLARec.Y[j,k].value})
        
        for i in Stations:
            beta.update({i: {}})
            gamma.update({i: {}})
            tau.update({i: {}})
            eta.update({i: {}})
            B.update({i: {}})
            for j in Trains:
                beta[i].update({j: {}})
                gamma[i].update({j: {}})
                tau[i].update({j: {}})
                eta[i].update({j: {}})
                B[i].update({j: {}})
                for k in Trains[j]['containers']:
                    beta[i][j].update({k: {}})
                    gamma[i][j].update({k: {}})
                    tau[i][j].update({k: {}})
                    eta[i][j].update({k: {}})
                    B[i][j].update({k: model_change_rate_v3_PLARec.B[i,j,k].value})
                    for u in Segments_SOC:
                        if u != max(Segments_SOC):
                            beta[i][j][k].update({u: model_change_rate_v3_PLARec.beta[i,j,k,u].value})
                        gamma[i][j][k].update({u: model_change_rate_v3_PLARec.gamma[i,j,k,u].value})
                    for v in Segments_hour_charge:
                        if v != max(Segments_hour_charge):
                            tau[i][j][k].update({v: model_change_rate_v3_PLARec.tau[i,j,k,v].value})
                        eta[i][j][k].update({v: model_change_rate_v3_PLARec.eta[i,j,k,v].value})
    
    
    return obj, cost_fix_weight, cost_delay_weight, D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, F, beta, gamma, tau, eta, B, results, time_model
    









'''This model is same as Model_change_rate_v2_PLARec except the following:
    1. In this model, all X variables are fixed at 1 (not same as v3), i.e., all stations are deployed.
    2. In this model, if a battery k is in train j, then Y[j,k]=1'''
def Model_change_rate_v4_PLARec(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                                M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, pi, Q):

    time_start = time.time()
    model_change_rate_v4_PLARec = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_change_rate_v4_PLARec.i = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v4_PLARec.i1 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v4_PLARec.i2 = pe.Set(initialize = set(Stations.keys()))
    model_change_rate_v4_PLARec.j = pe.Set(initialize = set(Trains.keys()))
    model_change_rate_v4_PLARec.k = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v4_PLARec.k1 = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v4_PLARec.k2 = pe.Set(initialize = set(Containers.keys()))
    model_change_rate_v4_PLARec.u = pe.Set(initialize = set(Segments_SOC.keys()))
    model_change_rate_v4_PLARec.v = pe.Set(initialize = set(Segments_hour_charge.keys()))
    
    # Variables
    # print ('Defining variables...')
    model_change_rate_v4_PLARec.D = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v4_PLARec.S_arrive = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v4_PLARec.S_depart = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.NonNegativeReals, bounds = (0,1))
    model_change_rate_v4_PLARec.T_arrive = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v4_PLARec.T_depart = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, within = pe.NonNegativeReals)
    model_change_rate_v4_PLARec.X = pe.Var(model_change_rate_v4_PLARec.i, within = pe.Binary)
    model_change_rate_v4_PLARec.Y = pe.Var(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.Binary)
    model_change_rate_v4_PLARec.Z_c = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.Binary)
    model_change_rate_v4_PLARec.Z_s = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.Binary)
    model_change_rate_v4_PLARec.T_c = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    # variables for linearization
    model_change_rate_v4_PLARec.B = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
    model_change_rate_v4_PLARec.F = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_change_rate_v4_PLARec.beta = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
    model_change_rate_v4_PLARec.gamma = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.u, within = pe.NonNegativeReals, bounds = (0,1))  # in PLA, continuous variables for S_arrive
    model_change_rate_v4_PLARec.tau = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.v, within = pe.Binary)  # in PLA, binary variables for T_c
    model_change_rate_v4_PLARec.eta = pe.Var(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.v, within = pe.NonNegativeReals, bounds = (0,1))  # in PLA, continuous variables for T_c


    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v4_PLARec.
    for j in model_change_rate_v4_PLARec.j:
        for k in model_change_rate_v4_PLARec.k:
            if k not in Trains[j]['containers']:
                model_change_rate_v4_PLARec.Y[j,k].fix(0)
            else:
                # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
                model_change_rate_v4_PLARec.Y[j,k].fix(1)
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_change_rate_v4_PLARec.j:
        model_change_rate_v4_PLARec.T_arrive['origin',j].fix(0)
        model_change_rate_v4_PLARec.T_depart['origin',j].fix(0)
    # All stations are deployed
    for i in model_change_rate_v4_PLARec.i:
        model_change_rate_v4_PLARec.X[i].fix(1)
       
    
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_change_rate_v4_PLARec):
        cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v4_PLARec.X[i] for i in model_change_rate_v4_PLARec.i)
        cost_delay = sum(sum((model_change_rate_v4_PLARec.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v4_PLARec.j) for i in model_change_rate_v4_PLARec.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_change_rate_v4_PLARec.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_change_rate_v4_PLARec, i, j):
        return model_change_rate_v4_PLARec.D[i,j] >= model_change_rate_v4_PLARec.T_depart[i,j] - model_change_rate_v4_PLARec.T_arrive[i,j]
    model_change_rate_v4_PLARec.delay_define_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule(model_change_rate_v4_PLARec, i):
        return sum(sum((model_change_rate_v4_PLARec.Z_c[i,j,k] + model_change_rate_v4_PLARec.Z_s[i,j,k]) \
                       for k in model_change_rate_v4_PLARec.k if k in Trains[j]['containers']) \
                   for j in model_change_rate_v4_PLARec.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_change_rate_v4_PLARec.X[i]
    model_change_rate_v4_PLARec.deploy_swap_charge_rule = pe.Constraint(model_change_rate_v4_PLARec.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule(model_change_rate_v4_PLARec, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.Z_s[i,j,k1] + model_change_rate_v4_PLARec.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.noboth_swap_charge_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k1, model_change_rate_v4_PLARec.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_change_rate_v4_PLARec, i, j):
        return model_change_rate_v4_PLARec.T_depart[i,j] - model_change_rate_v4_PLARec.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_change_rate_v4_PLARec.wait_passenger_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            if kid < len(Trains[j]['containers']):
                return sum(model_change_rate_v4_PLARec.Y[j,kp] \
                           for kp in model_change_rate_v4_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                       <= sum(1 for kp in model_change_rate_v4_PLARec.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                          * model_change_rate_v4_PLARec.Y[j,k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.consecutive_battery_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule(model_change_rate_v4_PLARec, i, j):
        return sum(model_change_rate_v4_PLARec.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_change_rate_v4_PLARec.max_number_batteries_station_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule(model_change_rate_v4_PLARec, i, j):
        return sum(model_change_rate_v4_PLARec.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_change_rate_v4_PLARec.max_number_chargers_station_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_change_rate_v4_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_change_rate_v4_PLARec.S_depart[i1,j,k] for k in Trains[j]['containers']) - sum(model_change_rate_v4_PLARec.S_arrive[i2,j,k] for k in Trains[j]['containers']) == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.power_rule = pe.Constraint(model_change_rate_v4_PLARec.i1, model_change_rate_v4_PLARec.i2, model_change_rate_v4_PLARec.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule_linear_a(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return model_change_rate_v4_PLARec.S_arrive[i,j,k]  -  M * model_change_rate_v4_PLARec.B[i,j,k]  <=  0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.battery_sequential_rule_linear_a = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    def battery_sequential_rule_linear_b(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return (1 - model_change_rate_v4_PLARec.S_arrive[i,j,k_next])  +  M * model_change_rate_v4_PLARec.B[i,j,k]  <=  M
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.battery_sequential_rule_linear_b = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] - model_change_rate_v4_PLARec.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_increase_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] >= model_change_rate_v4_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.swap_full_soc_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_change_rate_v4_PLARec, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return model_change_rate_v4_PLARec.S_arrive[i2,j,k] - model_change_rate_v4_PLARec.S_depart[i1,j,k] <= 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_depart_arrive_between_stations_rule = pe.Constraint(model_change_rate_v4_PLARec.i1, model_change_rate_v4_PLARec.i2, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.T_depart[i,j] - model_change_rate_v4_PLARec.T_arrive[i,j] >= hour_battery_swap * model_change_rate_v4_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.time_battery_swap_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_PLA_Zs(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v4_PLARec.F[i,j,k] \
                   - 2 * model_change_rate_v4_PLARec.Z_s[i,j,k] \
                   <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_PLA_Zc(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] \
                   + model_change_rate_v4_PLARec.F[i,j,k] \
                   + epsilon * model_change_rate_v4_PLARec.Z_c[i,j,k] \
                   >= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_Zc)        

    # print ('k2...')
    def soc_time_charge_rule_PLA_beta1(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v4_PLARec.beta[i,j,k,u] for u in model_change_rate_v4_PLARec.u if u != max(model_change_rate_v4_PLARec.u)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    def soc_time_charge_rule_PLA_gamma1(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v4_PLARec.gamma[i,j,k,u] for u in model_change_rate_v4_PLARec.u) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_tau1(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v4_PLARec.tau[i,j,k,v] for v in model_change_rate_v4_PLARec.v if v != max(model_change_rate_v4_PLARec.v)) == 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_change_rate_v4_PLARec.gamma[i,j,k,u]) \
                       for u in model_change_rate_v4_PLARec.u) \
                   == model_change_rate_v4_PLARec.S_arrive[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_hour_charge[v]['hour'] * model_change_rate_v4_PLARec.tau[i,j,k,v] \
                        + (Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_change_rate_v4_PLARec.eta[i,j,k,v]) \
                       for v in model_change_rate_v4_PLARec.v if v != max(model_change_rate_v4_PLARec.v)) \
                   == model_change_rate_v4_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta(model_change_rate_v4_PLARec, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_change_rate_v4_PLARec.u) and u != max(model_change_rate_v4_PLARec.u):
            return model_change_rate_v4_PLARec.gamma[i,j,k,u] <= model_change_rate_v4_PLARec.beta[i,j,k,u-1] + model_change_rate_v4_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == min(model_change_rate_v4_PLARec.u):
            return model_change_rate_v4_PLARec.gamma[i,j,k,u] <= model_change_rate_v4_PLARec.beta[i,j,k,u]
        elif k in Trains[j]['containers'] and u == max(model_change_rate_v4_PLARec.u):
            return model_change_rate_v4_PLARec.gamma[i,j,k,u] <= model_change_rate_v4_PLARec.beta[i,j,k,u-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau(model_change_rate_v4_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_change_rate_v4_PLARec.v) and v != max(model_change_rate_v4_PLARec.v):
            return model_change_rate_v4_PLARec.eta[i,j,k,v] <= model_change_rate_v4_PLARec.tau[i,j,k,v-1] + model_change_rate_v4_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == min(model_change_rate_v4_PLARec.v):
            return model_change_rate_v4_PLARec.eta[i,j,k,v] <= model_change_rate_v4_PLARec.tau[i,j,k,v]
        elif k in Trains[j]['containers'] and v == max(model_change_rate_v4_PLARec.v):
            return model_change_rate_v4_PLARec.eta[i,j,k,v] <= model_change_rate_v4_PLARec.tau[i,j,k,v-1]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1(model_change_rate_v4_PLARec, i, j, k, v):
        if k in Trains[j]['containers'] and v != max(model_change_rate_v4_PLARec.v):
            return model_change_rate_v4_PLARec.eta[i,j,k,v] <= model_change_rate_v4_PLARec.tau[i,j,k,v]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    def soc_time_charge_rule_PLA_F_leq(model_change_rate_v4_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v4_PLARec.u) and v != max(model_change_rate_v4_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v4_PLARec.F[i,j,k] \
                   <= sum((model_change_rate_v4_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v4_PLARec.u) \
                      + w * model_change_rate_v4_PLARec.eta[i,j,k,v] \
                      + M * (2 - model_change_rate_v4_PLARec.tau[i,j,k,v] - model_change_rate_v4_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.u, model_change_rate_v4_PLARec.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    def soc_time_charge_rule_PLA_F_geq(model_change_rate_v4_PLARec, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_change_rate_v4_PLARec.u) and v != max(model_change_rate_v4_PLARec.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_change_rate_v4_PLARec.F[i,j,k] \
                   >= sum((model_change_rate_v4_PLARec.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                          for w in model_change_rate_v4_PLARec.u) \
                      + w * model_change_rate_v4_PLARec.eta[i,j,k,v] \
                      - M * (2 - model_change_rate_v4_PLARec.tau[i,j,k,v] - model_change_rate_v4_PLARec.beta[i,j,k,u])
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, model_change_rate_v4_PLARec.u, model_change_rate_v4_PLARec.v, rule = soc_time_charge_rule_PLA_F_geq)        
    
    # print ('l...')
    # def soc_time_charge_rule_b(model_change_rate_v4_PLARec, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_change_rate_v4_PLARec.T_c[i,j,k] <= 1 - model_change_rate_v4_PLARec.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_change_rate_v4_PLARec.soc_time_charge_rule_b = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.T_c[i,j,k] <= model_change_rate_v4_PLARec.T_depart[i,j] - model_change_rate_v4_PLARec.T_arrive[i,j]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_c = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.T_c[i,j,k] <= M * model_change_rate_v4_PLARec.Z_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_d = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.Z_c[i,j,k] <= M * model_change_rate_v4_PLARec.T_c[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_time_charge_rule_e = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] - model_change_rate_v4_PLARec.S_arrive[i,j,k] <= model_change_rate_v4_PLARec.Z_c[i,j,k] + model_change_rate_v4_PLARec.Z_s[i,j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.no_swap_charge_soc_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart['origin',j,k] >= model_change_rate_v4_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_depart_origin_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_arrive['origin',j,k] >= model_change_rate_v4_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.soc_arrive_origin_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.Z_c['origin',j,k] + model_change_rate_v4_PLARec.Z_s['origin',j,k] <= 2 * model_change_rate_v4_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.origin_no_chargeswap_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_change_rate_v4_PLARec, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_change_rate_v4_PLARec.T_arrive[i2,j] == model_change_rate_v4_PLARec.T_depart[i1,j] + TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.T_traveltime_rule = pe.Constraint(model_change_rate_v4_PLARec.i1, model_change_rate_v4_PLARec.i2, model_change_rate_v4_PLARec.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v4_PLARec.Z_c[i,j,k] for i in model_change_rate_v4_PLARec.i) + sum(model_change_rate_v4_PLARec.Z_s[i,j,k] for i in model_change_rate_v4_PLARec.i) <= 2*len(Stations) * model_change_rate_v4_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.nobattery_nochargeswap_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_change_rate_v4_PLARec, j, k):
        if k in Trains[j]['containers']:
            return sum(model_change_rate_v4_PLARec.S_arrive[i,j,k] for i in model_change_rate_v4_PLARec.i) + sum(model_change_rate_v4_PLARec.S_depart[i,j,k] for i in model_change_rate_v4_PLARec.i) <= 2*len(Stations) * model_change_rate_v4_PLARec.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.nobattery_zerosoc_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_arrive[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.ub_soc_arrive_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_change_rate_v4_PLARec, i, j, k):
        if k in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.S_depart[i,j,k] <= 1
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.ub_soc_depart_rule = pe.Constraint(model_change_rate_v4_PLARec.i, model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_change_rate_v4_PLARec.
    def preprocessing_Y0_rule(model_change_rate_v4_PLARec, j, k):
        if k not in Trains[j]['containers']:
            return model_change_rate_v4_PLARec.Y[j,k] == 0
        else:
            return pe.Constraint.Skip
    model_change_rate_v4_PLARec.preprocessing_Y0_rule = pe.Constraint(model_change_rate_v4_PLARec.j, model_change_rate_v4_PLARec.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_change_rate_v4_PLARec, j):
        return model_change_rate_v4_PLARec.T_arrive['origin',j] + model_change_rate_v4_PLARec.T_depart['origin',j] == 0
    model_change_rate_v4_PLARec.preprocessing_T0_rule = pe.Constraint(model_change_rate_v4_PLARec.j, rule = preprocessing_T0_rule)
      
    
    # # define dual variables
    # model_change_rate_v4_PLARec.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    # model_change_rate_v4_PLARec.c = Constraint()
    # model_change_rate_v4_PLARec.dual[model_change_rate_v4_PLARec.c] = 1.0
    
        
    # solve the model
    print('Solving...')
    opt= pyomo.opt.SolverFactory("cplex")
    # optimality_gap = 0.05
    # opt.options["mip_tolerances_mipgap"] = optimality_gap
    opt.options["timelimit"] = 900
    results=opt.solve(model_change_rate_v4_PLARec, tee=False, keepfiles=False)
    results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_change_rate_v4_PLARec, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_change_rate_v4_PLARec)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_change_rate_v4_PLARec, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # find value of dual variable for all constraints:
    for constraint_name, constraint_object in model_change_rate_v4_PLARec.component_map(Constraint).items():
        if constraint_name != 'c':
            pi[Q].update({constraint_name: dict()})
            for index in constraint_object.keys():
                # dual_value = model_change_rate_v4_PLARec.dual[constraint_object[index]]  # Get dual value
                pi[Q][constraint_name].update({index: 0})   # in initialization, we make pi variables equal zero
        
        
    time_end = time.time()
    time_model = time_end - time_start
    upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
    gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
    
    cost_fix = sum(Stations[i]['cost_fix'] * model_change_rate_v4_PLARec.X[i].value for i in model_change_rate_v4_PLARec.i)
    cost_delay = sum(sum((model_change_rate_v4_PLARec.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_change_rate_v4_PLARec.j) for i in model_change_rate_v4_PLARec.i)
    obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
    
    '''Record variables'''
    # print ('Recording variables...')
    D = {}
    S_arrive = {}
    S_depart = {}
    T_arrive = {}
    T_depart = {}
    T_c = {}
    X = {}
    Y = {}
    Z_c = {}
    Z_s = {}
    F = {}
    
    for i in Stations:
        X.update({i: model_change_rate_v4_PLARec.X[i].value})
        D.update({i: {}})
        S_arrive.update({i: {}})
        S_depart.update({i: {}})
        T_arrive.update({i: {}})
        T_depart.update({i: {}})
        T_c.update({i: {}})
        Z_c.update({i: {}})
        Z_s.update({i: {}})
        F.update({i: {}})
        for j in Trains:
            D[i].update({j: model_change_rate_v4_PLARec.D[i,j].value})
            T_arrive[i].update({j: model_change_rate_v4_PLARec.T_arrive[i,j].value})
            T_depart[i].update({j: model_change_rate_v4_PLARec.T_depart[i,j].value})
            S_arrive[i].update({j: {}})
            S_depart[i].update({j: {}})
            Z_c[i].update({j: {}})
            Z_s[i].update({j: {}})
            T_c[i].update({j: {}})
            F[i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[i][j].update({k: model_change_rate_v4_PLARec.S_arrive[i,j,k].value})
                S_depart[i][j].update({k: model_change_rate_v4_PLARec.S_depart[i,j,k].value})
                Z_c[i][j].update({k: model_change_rate_v4_PLARec.Z_c[i,j,k].value})
                Z_s[i][j].update({k: model_change_rate_v4_PLARec.Z_s[i,j,k].value})
                T_c[i][j].update({k: model_change_rate_v4_PLARec.T_c[i,j,k].value})
                F[i][j].update({k: model_change_rate_v4_PLARec.F[i,j,k].value})
    for j in Trains:
        Y.update({j: {}})
        for k in Trains[j]['containers']:
            Y[j].update({k: model_change_rate_v4_PLARec.Y[j,k].value})
    
    beta = {}
    gamma = {}
    tau = {}
    eta = {}
    for i in Stations:
        beta.update({i: {}})
        gamma.update({i: {}})
        tau.update({i: {}})
        eta.update({i: {}})
        for j in Trains:
            beta[i].update({j: {}})
            gamma[i].update({j: {}})
            tau[i].update({j: {}})
            eta[i].update({j: {}})
            for k in Trains[j]['containers']:
                beta[i][j].update({k: {}})
                gamma[i][j].update({k: {}})
                tau[i][j].update({k: {}})
                eta[i][j].update({k: {}})
                for u in Segments_SOC:
                    if u != max(Segments_SOC):
                        beta[i][j][k].update({u: model_change_rate_v4_PLARec.beta[i,j,k,u].value})
                    gamma[i][j][k].update({u: model_change_rate_v4_PLARec.gamma[i,j,k,u].value})
                for v in Segments_hour_charge:
                    if v != max(Segments_hour_charge):
                        tau[i][j][k].update({v: model_change_rate_v4_PLARec.tau[i,j,k,v].value})
                    eta[i][j][k].update({v: model_change_rate_v4_PLARec.eta[i,j,k,v].value})
    
    
    
    
    return obj, D, S_arrive, S_depart, T_arrive, T_depart, T_c, X, Y, Z_c, Z_s, F, beta, gamma, tau, eta, results, time_model, gap, upper_bound, lower_bound, pi
    


''' This model is based on Model_change_rate_v2_PLARec. 
Model_change_rate_v2_PLARec corresponds to OP in latex, while this model corresponds to RMP in latex, specifically:
    (1) For variables, this model removes all continuous variables from Model_change_rate_v2_PLARec, and add a new variable W
    (2) For the objective function, this model replaces all terms involving continuous variables with a new variable W
    (3) For constraints:
        a. This model has added optimality cuts w >= (b-Dv)*pi[p] (for all p in EP), where b is RHS vector of Model_change_rate_v2_PLARec, D is the constraint matrix for integer variables, and p is an extreme point of DSP
        b. This model has added feasibility cuts 0 >= (b-Dv)*pi[r] (for all r in ER), where b is RHS vector of Model_change_rate_v2_PLARec, D is the constraint matrix for integer variables, and r is an extreme ray of DSP
        c. This model has kept all constraints NOT involving any continuous variable
'''
def Model_Bender_RMP_v2(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                        M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, \
                        X, Y, Z_c, Z_s, beta, tau, B, W, pi, Q, results_SP, UB, itr_while, ratio_UB, obj_RMP, results_RMP, LB, gap_RMP):
    time_start = time.time()
    model_Bender_RMP_v2 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # extract keys from pi and classify keys for DSP extreme points and DSP extreme rays respectively
    er_set = set({})
    ep_set = set({})
    for q in pi:
        if pi[q]['source_type'] == 'extreme_point':
            ep_set.add(q)
        elif pi[q]['source_type'] == 'extreme_ray':
            er_set.add(q)
    # Sets and indices
    model_Bender_RMP_v2.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v2.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v2.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v2.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_Bender_RMP_v2.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v2.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v2.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v2.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_Bender_RMP_v2.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    # model_Bender_RMP_v2.p = pe.Set(initialize = sorted(set(pi.keys())))
    model_Bender_RMP_v2.p = pe.Set(initialize = sorted(ep_set))
    # model_Bender_RMP_v2.r = pe.Set(initialize = sorted(set(pi.keys())))
    model_Bender_RMP_v2.r = pe.Set(initialize = sorted(er_set))
    
    # Variables
    # print ('Defining variables...')
    model_Bender_RMP_v2.X = pe.Var(model_Bender_RMP_v2.i, within = pe.Binary)
    model_Bender_RMP_v2.Y = pe.Var(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, within = pe.Binary)
    model_Bender_RMP_v2.Z_c = pe.Var(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, within = pe.Binary)
    model_Bender_RMP_v2.Z_s = pe.Var(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, within = pe.Binary)
    # variables for linearization
    model_Bender_RMP_v2.B = pe.Var(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
    model_Bender_RMP_v2.beta = pe.Var(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, model_Bender_RMP_v2.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
    model_Bender_RMP_v2.tau = pe.Var(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, model_Bender_RMP_v2.v, within = pe.Binary)  # in PLA, binary variables for T_c
    # introduce new RMP variable: W
    model_Bender_RMP_v2.W = pe.Var([0], within = pe.Reals)
    
    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_RMP_v2.
    for j in model_Bender_RMP_v2.j:
        for k in model_Bender_RMP_v2.k:
            model_Bender_RMP_v2.Z_c['origin',j,k].fix(0) # batteries are not charged in the origin
            model_Bender_RMP_v2.Z_s['origin',j,k].fix(0) # batteries are not swapped in the origin
            if k not in Trains[j]['containers']:
                model_Bender_RMP_v2.Y[j,k].fix(0)
            else:
                # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
                model_Bender_RMP_v2.Y[j,k].fix(1)
    # If consist k is not in train j, then the corresponding Z, B, beta and tau variables are all zero.
    for i in model_Bender_RMP_v2.i:
        for j in model_Bender_RMP_v2.j:
            for k in model_Bender_RMP_v2.k:
                if k not in Trains[j]['containers']:
                    model_Bender_RMP_v2.Z_c[i,j,k].fix(0)
                    model_Bender_RMP_v2.Z_s[i,j,k].fix(0)
                    model_Bender_RMP_v2.B[i,j,k].fix(0)
                    for u in model_Bender_RMP_v2.u:
                        model_Bender_RMP_v2.beta[i,j,k,u].fix(0)
                    for v in model_Bender_RMP_v2.v:
                        model_Bender_RMP_v2.tau[i,j,k,v].fix(0)
    
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_Bender_RMP_v2):
        cost_fix = sum(Stations[i]['cost_fix'] * model_Bender_RMP_v2.X[i] for i in model_Bender_RMP_v2.i)
        constant = - penalty_delay * sum(sum(Trains[j]['stations'][i]['time_wait'] for j in model_Bender_RMP_v2.j) for i in model_Bender_RMP_v2.i)
        obj = penalty_fix_cost * cost_fix + model_Bender_RMP_v2.W[0] + constant
        return obj  
    model_Bender_RMP_v2.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    # constraints    
    # add feasibility cut from extreme ray
    def extreme_ray_rule(model_Bender_RMP_v2, r):
        return 0 >= \
                + sum(sum((Trains[j]['stations'][i]['time_wait']*pi[r]['wait_passenger_rule'][i, j]) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((Power_train[j][i1][i2] * pi[r]['power_rule'][i1,i2,j]) \
                              for i2 in model_Bender_RMP_v2.i2 if i2 == Stations[i1]['station_after']) \
                          for i1 in model_Bender_RMP_v2.i1) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((- pi[r]['battery_sequential_rule_linear_a'][i,j,k] *  M * model_Bender_RMP_v2.B[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(((1-M) * pi[r]['battery_sequential_rule_linear_b'][i,j,k] + M*pi[r]['battery_sequential_rule_linear_b'][i,j,k] * model_Bender_RMP_v2.B[i,j,k])\
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[r]['swap_full_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((hour_battery_swap * pi[r]['time_battery_swap_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[r]['soc_time_charge_rule_PLA_Zs'][i,j,k] \
                                - M*pi[r]['soc_time_charge_rule_PLA_Zs'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[r]['soc_time_charge_rule_PLA_Zc'][i,j,k] \
                                - pi[r]['soc_time_charge_rule_PLA_Zc'][i,j,k]*epsilon * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[r]['soc_time_charge_rule_PLA_gamma1'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[r]['soc_time_charge_rule_PLA_tau_eta_Tc'][i,j,k] * sum((Segments_hour_charge[v]['hour'] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                                                                          for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v))) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u-1] \
                                  - pi[r]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u != min(model_Bender_RMP_v2.u) and u != max(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u == min(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u-1]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u == max(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v-1] \
                                  - pi[r]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v != min(model_Bender_RMP_v2.v) and v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v == min(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v-1]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v == max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[r]['soc_time_charge_rule_PLA_eta_tau1'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v] ) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v != min(model_Bender_RMP_v2.v) and v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum(sum((- 2*pi[r]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M \
                                      + pi[r]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.tau[i,j,k,v] \
                                      + pi[r]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                      for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_RMP_v2.j) \
                              for i in model_Bender_RMP_v2.i) \
                          for u in model_Bender_RMP_v2.u if u != max(model_Bender_RMP_v2.u)) \
                      for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum(sum((- 2*pi[r]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M \
                                      + pi[r]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.tau[i,j,k,v] \
                                      + pi[r]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                      for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_RMP_v2.j) \
                              for i in model_Bender_RMP_v2.i) \
                          for u in model_Bender_RMP_v2.u if u != max(model_Bender_RMP_v2.u)) \
                      for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum((- M*pi[r]['soc_time_charge_rule_d'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[r]['soc_time_charge_rule_e'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((- pi[r]['no_swap_charge_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k] \
                              - pi[r]['no_swap_charge_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum((pi[r]['soc_depart_origin_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum((pi[r]['soc_arrive_origin_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((pi[r]['T_traveltime_rule'][i1,i2,j] * TravelTime_train[j][i1][i2]) \
                              for i2 in model_Bender_RMP_v2.i2 if i2 == Stations[i1]['station_after']) \
                          for i1 in model_Bender_RMP_v2.i1) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum((-2*len(Stations)*pi[r]['nobattery_zerosoc_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((-pi[r]['ub_soc_arrive_rule'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[r]['ub_soc_depart_rule'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(sum((-pi[r]['ub_gamma_rule'][i,j,k,str(u)]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u) \
                + sum(sum(sum(sum((-pi[r]['ub_eta_rule'][i,j,k,str(v)]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v) 
    model_Bender_RMP_v2.extreme_ray_rule = pe.Constraint(model_Bender_RMP_v2.r, rule = extreme_ray_rule)
    
    # add optimality cut from extreme ray
    def extreme_point_rule(model_Bender_RMP_v2, p):
        return model_Bender_RMP_v2.W[0] >= \
                + sum(sum((Trains[j]['stations'][i]['time_wait']*pi[p]['wait_passenger_rule'][i, j]) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((Power_train[j][i1][i2] * pi[p]['power_rule'][i1,i2,j]) \
                              for i2 in model_Bender_RMP_v2.i2 if i2 == Stations[i1]['station_after']) \
                          for i1 in model_Bender_RMP_v2.i1) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((- pi[p]['battery_sequential_rule_linear_a'][i,j,k] *  M * model_Bender_RMP_v2.B[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(((1-M) * pi[p]['battery_sequential_rule_linear_b'][i,j,k] + M*pi[p]['battery_sequential_rule_linear_b'][i,j,k] * model_Bender_RMP_v2.B[i,j,k])\
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[p]['swap_full_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((hour_battery_swap * pi[p]['time_battery_swap_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[p]['soc_time_charge_rule_PLA_Zs'][i,j,k] \
                                - M*pi[p]['soc_time_charge_rule_PLA_Zs'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[p]['soc_time_charge_rule_PLA_Zc'][i,j,k] \
                                - pi[p]['soc_time_charge_rule_PLA_Zc'][i,j,k]*epsilon * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[p]['soc_time_charge_rule_PLA_gamma1'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[p]['soc_time_charge_rule_PLA_tau_eta_Tc'][i,j,k] * sum((Segments_hour_charge[v]['hour'] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                                                                          for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v))) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u-1] \
                                  - pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u != min(model_Bender_RMP_v2.u) and u != max(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u == min(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * model_Bender_RMP_v2.beta[i,j,k,u-1]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u if u == max(model_Bender_RMP_v2.u)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v-1] \
                                  - pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v != min(model_Bender_RMP_v2.v) and v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v == min(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v-1]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v == max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau1'][i,j,k,str(v)] * model_Bender_RMP_v2.tau[i,j,k,v] ) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v if v != min(model_Bender_RMP_v2.v) and v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum(sum((- 2*pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M \
                                      + pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.tau[i,j,k,v] \
                                      + pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                      for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_RMP_v2.j) \
                              for i in model_Bender_RMP_v2.i) \
                          for u in model_Bender_RMP_v2.u if u != max(model_Bender_RMP_v2.u)) \
                      for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum(sum(sum((- 2*pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M \
                                      + pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.tau[i,j,k,v] \
                                      + pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * model_Bender_RMP_v2.beta[i,j,k,u]) \
                                      for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_RMP_v2.j) \
                              for i in model_Bender_RMP_v2.i) \
                          for u in model_Bender_RMP_v2.u if u != max(model_Bender_RMP_v2.u)) \
                      for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v)) \
                + sum(sum(sum((- M*pi[p]['soc_time_charge_rule_d'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((pi[p]['soc_time_charge_rule_e'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((- pi[p]['no_swap_charge_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_c[i,j,k] \
                              - pi[p]['no_swap_charge_soc_rule'][i,j,k] * model_Bender_RMP_v2.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum((pi[p]['soc_depart_origin_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum((pi[p]['soc_arrive_origin_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((pi[p]['T_traveltime_rule'][i1,i2,j] * TravelTime_train[j][i1][i2]) \
                              for i2 in model_Bender_RMP_v2.i2 if i2 == Stations[i1]['station_after']) \
                          for i1 in model_Bender_RMP_v2.i1) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum((-2*len(Stations)*pi[p]['nobattery_zerosoc_rule'][j,k] * model_Bender_RMP_v2.Y[j,k]) \
                          for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                      for j in model_Bender_RMP_v2.j) \
                + sum(sum(sum((-pi[p]['ub_soc_arrive_rule'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum((-pi[p]['ub_soc_depart_rule'][i,j,k]) \
                              for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v2.j) \
                      for i in model_Bender_RMP_v2.i) \
                + sum(sum(sum(sum((-pi[p]['ub_gamma_rule'][i,j,k,str(u)]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for u in model_Bender_RMP_v2.u) \
                + sum(sum(sum(sum((-pi[p]['ub_eta_rule'][i,j,k,str(v)]) \
                                  for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                              for j in model_Bender_RMP_v2.j) \
                          for i in model_Bender_RMP_v2.i) \
                      for v in model_Bender_RMP_v2.v) 
    model_Bender_RMP_v2.extreme_point_rule = pe.Constraint(model_Bender_RMP_v2.p, rule = extreme_point_rule)

    '''add extra feasibility cuts: '''
    # This optimality is from Fix Algorithm. The idea is: the power from deployed stations must >= the power required on the route. 
    # The calculation of power_demand, power_supply_origin, and power_min_station is same as that from FRE_FixAlg_v1.py.
    # print ('opt1...')
    def power_support_rule_feas(model_Bender_RMP_v2, j):
        power_demand = sum(Power_train[j][i][Stations[i]['station_after']] \
                                for i in Stations if i != 'destination') 
        power_supply_origin = len(Trains[j]['containers'])
        power_min_station = power_demand - power_supply_origin  # the minimum amount of power train j requires from stations
        return sum((Stations[i]['max_power_provide'] * model_Bender_RMP_v2.X[i]) \
                    for i in model_Bender_RMP_v2.i) \
                >= power_min_station
    model_Bender_RMP_v2.power_support_rule_feas = pe.Constraint(model_Bender_RMP_v2.j, rule = power_support_rule_feas)        
   
    # # If station is deployed, then at least one battery must be swapped nor charged in station i
    # print ('opt2...')
    def deploy_swap_charge_must_rule_orig_feas(model_Bender_RMP_v2, i):
        return sum(sum((model_Bender_RMP_v2.Z_c[i,j,k] + model_Bender_RMP_v2.Z_s[i,j,k]) \
                        for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                    for j in model_Bender_RMP_v2.j) \
                >= model_Bender_RMP_v2.X[i]
    model_Bender_RMP_v2.deploy_swap_charge_must_rule_orig_feas = pe.Constraint(model_Bender_RMP_v2.i, rule = deploy_swap_charge_must_rule_orig_feas)        
    
    # The physical explanation of B[i,j,k] is: B[i,j,k]=0 indicates that the power in battery k of train j is used up (SOC=0) at station i. B[i,j,k]=1 indicates SOC of battery k is > 0.
    # Feasibility cuts for B variables:
    # When B[i,j,k]=0, then the power in batteries consists 1,...,k-1 is used up, so B[i,j,1]=...=B[i,j,k-1]=0
    # print ('opt3...')
    def battery_sequential_rule_linear_a_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            return sum(model_Bender_RMP_v2.B[i,j,kp] \
                        for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) < k_index) \
                    <= (k_index+1-1) * model_Bender_RMP_v2.B[i,j,k]
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.battery_sequential_rule_linear_a_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = battery_sequential_rule_linear_a_feas)        

    # When B[i,j,k]=1, then the power in batteries consists k+1,...,kj is full (SOC=100%) (where kj is the total number of consists in train j), so B[i,j,1]=...=B[i,j,k-1]=1.
    # print ('opt4...')
    def battery_sequential_rule_linear_b_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            return sum(model_Bender_RMP_v2.B[i,j,kp] \
                        for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) > k_index) \
                    >= (len(Trains[j]['containers']) - k_index-1) * model_Bender_RMP_v2.B[i,j,k]
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.battery_sequential_rule_linear_b_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = battery_sequential_rule_linear_b_feas)        
    
    # If the power in consist k is used not used up, then power in consist k+1 is not used up either
    # print ('opt5...')
    def battery_sequential_rule_linear_c_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index <= len(Trains[j]['containers'])-2:
                k_next = Trains[j]['containers'][k_index+1]
                return model_Bender_RMP_v2.B[i,j,k_next] >= model_Bender_RMP_v2.B[i,j,k] 
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.battery_sequential_rule_linear_c_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = battery_sequential_rule_linear_c_feas)        

    # Feasibility cuts for the connection between B and X variables:
    # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then a charging/swapping station must be deployed in location i.
    # print ('opt6...')
    def model_B_X_a_feas(model_Bender_RMP_v2, i, j):
        i_next = Stations[i]['station_after']
        if i != 'origin' and i != 'destination' and i_next != 'None':
            return M * model_Bender_RMP_v2.X[i] \
                    >= Power_train[j][i][i_next] - sum(model_Bender_RMP_v2.B[i,j,k] for k in Trains[j]['containers'])
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_X_a_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = model_B_X_a_feas)
    
    # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then a charging/swapping station must be deployed in location i.
    def model_B_X_b_feas(model_Bender_RMP_v2, i, j):
        i_next = Stations[i]['station_after']
        if i != 'origin' and i != 'destination' and i_next != 'None':
            return M * model_Bender_RMP_v2.X[i] \
                    >= Power_train[j][i][i_next] * (1 - sum(model_Bender_RMP_v2.B[i,j,k] \
                                                            for k in Trains[j]['containers']) )
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_X_b_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = model_B_X_b_feas)

    # Feasibility cuts for the connection between B and Y variables:
    # if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    def model_B_Y_feas(model_Bender_RMP_v2, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v2.Y[j,k] == model_Bender_RMP_v2.B['origin',j,k]
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_Y_feas = pe.Constraint(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = model_B_Y_feas)

    # Feasibility cuts for the connection between B and Z variables:
    # If the maximum potential amount of power in train j (\sum_{k \in K_j} B_{ijk}) is smaller than the power required from location i to i', then at least one battery of train j must be charged or swapped in location i.
    def model_B_Z_a_feas(model_Bender_RMP_v2, i, j):
        i_next = Stations[i]['station_after']
        if i != 'origin' and i != 'destination' and i_next != 'None':
            return M * sum((model_Bender_RMP_v2.Z_c[i,j,k] + model_Bender_RMP_v2.Z_s[i,j,k]) \
                            for k in Trains[j]['containers']) \
                    >= Power_train[j][i][i_next] - sum(model_Bender_RMP_v2.B[i,j,k] \
                                                      for k in Trains[j]['containers'])
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_Z_a_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = model_B_Z_a_feas)

    # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then at least one battery in train j must be charged/swapped in location i.
    def model_B_Z_b_feas(model_Bender_RMP_v2, i, j):
        i_next = Stations[i]['station_after']
        if i != 'origin' and i != 'destination' and i_next != 'None':
            return M * sum((model_Bender_RMP_v2.Z_c[i,j,k] + model_Bender_RMP_v2.Z_s[i,j,k]) \
                            for k in Trains[j]['containers']) \
                    >= Power_train[j][i][i_next] * (1 - sum(model_Bender_RMP_v2.B[i,j,k] \
                                                            for k in Trains[j]['containers']) )
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_Z_b_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = model_B_Z_b_feas)
    
    # Feasibility cuts for the connection between B and beta variables:
    # If B[ijk]=0, then S_arrive[ijk]=0, and we have beta[ijk0]=1
    # print ('opt7...')
    def model_B_beta_a_rule_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            return sum(model_Bender_RMP_v2.beta[i,j,kp,min(model_Bender_RMP_v2.u)] \
                       for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) <= k_index) \
                   + (k_index+1) * model_Bender_RMP_v2.B[i,j,k] \
                   >= k_index+1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_beta_a_rule_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = model_B_beta_a_rule_feas)
    
    # If B[ijk]=1, then S_arrive[i,j,k+1]=1, and we have beta[i,j,k+1,n]=0
    # print ('opt8...')
    def model_B_beta_b_rule_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index <= len(Trains[j]['containers'])-2:
                return sum(model_Bender_RMP_v2.beta[i,j,kp,max(model_Bender_RMP_v2.u)] \
                            for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) >= k_index+1) \
                        - (len(Trains[j]['containers']) - (k_index+1)) * model_Bender_RMP_v2.B[i,j,k] \
                        >= 0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_beta_b_rule_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = model_B_beta_b_rule_feas)        

    def model_B_beta_c_rule_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v2.beta[i,j,k,min(model_Bender_RMP_v2.u)] \
                   + model_Bender_RMP_v2.B[i,j,k] \
                   >= 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_beta_c_rule_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = model_B_beta_c_rule_feas)
    
    # If B[ijk]=1, then S_arrive[i,j,k+1]=1, and we have beta[i,j,k+1,n]=0
    # print ('opt8...')
    def model_B_beta_d_rule_feas(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index <= len(Trains[j]['containers'])-2:
                k_next = Trains[j]['containers'][k_index+1]
                return model_Bender_RMP_v2.beta[i,j,k_next,max(model_Bender_RMP_v2.u)] \
                       - model_Bender_RMP_v2.B[i,j,k] \
                       >= 0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.model_B_beta_d_rule_feas = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = model_B_beta_d_rule_feas)        


    '''original constraints'''
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule_orig(model_Bender_RMP_v2, i):
        return sum(sum((model_Bender_RMP_v2.Z_c[i,j,k] + model_Bender_RMP_v2.Z_s[i,j,k]) \
                       for k in model_Bender_RMP_v2.k if k in Trains[j]['containers']) \
                   for j in model_Bender_RMP_v2.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_Bender_RMP_v2.X[i]
    model_Bender_RMP_v2.deploy_swap_charge_rule_orig = pe.Constraint(model_Bender_RMP_v2.i, rule = deploy_swap_charge_rule_orig)        
    
    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule_orig(model_Bender_RMP_v2, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_Bender_RMP_v2.Z_s[i,j,k1] + model_Bender_RMP_v2.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.noboth_swap_charge_rule_orig = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k1, model_Bender_RMP_v2.k2, rule = noboth_swap_charge_rule_orig)            
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule_orig(model_Bender_RMP_v2, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            if kid < len(Trains[j]['containers']):
                return sum(model_Bender_RMP_v2.Y[j,kp] \
                           for kp in model_Bender_RMP_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                       <= sum(1 for kp in model_Bender_RMP_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                          * model_Bender_RMP_v2.Y[j,k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.consecutive_battery_rule_orig = pe.Constraint(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = consecutive_battery_rule_orig)  
        
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule_orig(model_Bender_RMP_v2, i, j):
        return sum(model_Bender_RMP_v2.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_Bender_RMP_v2.max_number_batteries_station_rule_orig = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = max_number_batteries_station_rule_orig)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule_orig(model_Bender_RMP_v2, i, j):
        return sum(model_Bender_RMP_v2.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_Bender_RMP_v2.max_number_chargers_station_rule_orig = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, rule = max_number_chargers_station_rule_orig)  
    
    # print ('k2...')
    def soc_time_charge_rule_PLA_beta1_orig(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v2.beta[i,j,k,u] for u in model_Bender_RMP_v2.u if u != max(model_Bender_RMP_v2.u)) == 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.soc_time_charge_rule_PLA_beta1_orig = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = soc_time_charge_rule_PLA_beta1_orig)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_tau1_orig(model_Bender_RMP_v2, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v2.tau[i,j,k,v] for v in model_Bender_RMP_v2.v if v != max(model_Bender_RMP_v2.v)) == 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.soc_time_charge_rule_PLA_tau1_orig = pe.Constraint(model_Bender_RMP_v2.i, model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = soc_time_charge_rule_PLA_tau1_orig)        
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule_orig(model_Bender_RMP_v2, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v2.Z_c['origin',j,k] + model_Bender_RMP_v2.Z_s['origin',j,k] <= 2 * model_Bender_RMP_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.origin_no_chargeswap_rule_orig = pe.Constraint(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = origin_no_chargeswap_rule_orig)

    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule_orig(model_Bender_RMP_v2, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v2.Z_c[i,j,k] for i in model_Bender_RMP_v2.i) \
                   + sum(model_Bender_RMP_v2.Z_s[i,j,k] for i in model_Bender_RMP_v2.i) \
                   <= 2*len(Stations) * model_Bender_RMP_v2.Y[j,k]
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v2.nobattery_nochargeswap_rule_orig = pe.Constraint(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = nobattery_nochargeswap_rule_orig)
        
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_RMP_v2.
    # print ('r...')
    # def preprocessing_Y0_rule_orig(model_Bender_RMP_v2, j, k):
    #     if k not in Trains[j]['containers']:
    #         return model_Bender_RMP_v2.Y[j,k] == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v2.preprocessing_Y0_rule_orig = pe.Constraint(model_Bender_RMP_v2.j, model_Bender_RMP_v2.k, rule = preprocessing_Y0_rule_orig)
      
    # solve the model
    # print('Solving...')
    # solve by gurobi
    opt = pyomo.opt.SolverFactory('gurobi')
    opt.options['MIPGap'] = gap_RMP  # 1% optimality gap
    results = opt.solve(model_Bender_RMP_v2, tee=False, keepfiles=False)
    
    # solve by cplex:
    # opt= pyomo.opt.SolverFactory("cplex")
    # # optimality_gap = 0.05
    # # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 900
    # results=opt.solve(model_Bender_RMP_v2, tee=False, keepfiles=False)
    # # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_Bender_RMP_v2, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_Bender_RMP_v2)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_Bender_RMP_v2, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    
    time_end = time.time()
    time_model = time_end - time_start
    # upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
    # gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
    
    '''Record variables'''
    # print ('Recording variables...')
    W.update({itr_while: model_Bender_RMP_v2.W[0].value})
    
    X.update({itr_while:{}})
    Y.update({itr_while:{}})
    Z_c.update({itr_while:{}})
    Z_s.update({itr_while:{}})
    for i in Stations:
        X[itr_while].update({i: model_Bender_RMP_v2.X[i].value})
        Z_c[itr_while].update({i: {}})
        Z_s[itr_while].update({i: {}})
        for j in Trains:
            Z_c[itr_while][i].update({j: {}})
            Z_s[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                Z_c[itr_while][i][j].update({k: model_Bender_RMP_v2.Z_c[i,j,k].value})
                Z_s[itr_while][i][j].update({k: model_Bender_RMP_v2.Z_s[i,j,k].value})
    for j in Trains:
        Y[itr_while].update({j: {}})
        for k in Trains[j]['containers']:
            Y[itr_while][j].update({k: model_Bender_RMP_v2.Y[j,k].value})
            
    beta.update({itr_while:{}})
    tau.update({itr_while:{}})
    B.update({itr_while:{}})
    for i in Stations:
        beta[itr_while].update({i: {}})
        tau[itr_while].update({i: {}})
        B[itr_while].update({i: {}})
        for j in Trains:
            beta[itr_while][i].update({j: {}})
            tau[itr_while][i].update({j: {}})
            B[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                beta[itr_while][i][j].update({k: {}})
                tau[itr_while][i][j].update({k: {}})
                if model_Bender_RMP_v2.B[i,j,k].value == None:
                    B[itr_while][i][j].update({k: 1})
                else:
                    B[itr_while][i][j].update({k: model_Bender_RMP_v2.B[i,j,k].value})
                for u in Segments_SOC:
                    if u != max(Segments_SOC):
                        beta[itr_while][i][j][k].update({u: model_Bender_RMP_v2.beta[i,j,k,u].value})
                for v in Segments_hour_charge:
                    if v != max(Segments_hour_charge):
                        tau[itr_while][i][j][k].update({v: model_Bender_RMP_v2.tau[i,j,k,v].value})
    
    # calculate the objective function value
    cost_fix = sum(Stations[i]['cost_fix'] * model_Bender_RMP_v2.X[i].value for i in model_Bender_RMP_v2.i)
    constant = - penalty_delay * sum(sum(Trains[j]['stations'][i]['time_wait'] for j in model_Bender_RMP_v2.j) for i in model_Bender_RMP_v2.i)
    obj = penalty_fix_cost * cost_fix + model_Bender_RMP_v2.W[0].value + constant
    obj_RMP.update({itr_while: obj})
    if itr_while > 0:
        if obj > LB[itr_while-1]:
            LB[itr_while] = obj
        else:
            LB[itr_while] = LB[itr_while-1]
    results_RMP.update({itr_while: results})
    
    
    return obj_RMP, LB, W, X, Y, Z_c, Z_s, beta, tau, B, results_RMP, time_model














'''This model is SP of the original problem (model_change_rate_v2_PLARec). The difference between this model and model_change_rate_v2_PLARec is as follows:
    (1) For variables, this model only has continuous variables. The values of integer variables are all fixed, which are obtained from Model_Bender_RMP_v2
    (2) For the objective function, this model obly considers terms of continuous variables. 
    (3) For constraints, this model is same as model_change_rate_v2_PLARec (except that the terms with fixed value of integer variables are no longer variables. They serve as constants instead.)
'''
def Model_Bender_SP_v1(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                       M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, \
                       X, Y, Z_c, Z_s, beta, tau, B, D, S_arrive, S_depart, T_arrive, T_depart, T_c, F, gamma, eta, \
                       pi, Q, ER, EP, itr_while, ratio_ER, obj_SP, results_SP, UB):

    time_start = time.time()
    model_Bender_SP_v1 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_Bender_SP_v1.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v1.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v1.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v1.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_Bender_SP_v1.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v1.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v1.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v1.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_Bender_SP_v1.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    
    # Variables
    # print ('Defining variables...')
    model_Bender_SP_v1.D = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SP_v1.S_arrive = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, within = pe.NonNegativeReals)
    model_Bender_SP_v1.S_depart = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, within = pe.NonNegativeReals)
    model_Bender_SP_v1.T_arrive = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SP_v1.T_depart = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SP_v1.T_c = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    # variables for linearization
    model_Bender_SP_v1.F = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_Bender_SP_v1.gamma = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
    model_Bender_SP_v1.eta = pe.Var(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c


    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_Bender_SP_v1.j:
        model_Bender_SP_v1.T_arrive['origin',j].fix(0)
        model_Bender_SP_v1.T_depart['origin',j].fix(0)
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_Bender_SP_v1):
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SP_v1.i)
        cost_delay = sum(sum((model_Bender_SP_v1.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SP_v1.j) for i in model_Bender_SP_v1.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_Bender_SP_v1.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_Bender_SP_v1, i, j):
        return model_Bender_SP_v1.D[i,j] - model_Bender_SP_v1.T_depart[i,j] + model_Bender_SP_v1.T_arrive[i,j] >= 0
    model_Bender_SP_v1.delay_define_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('b...')
    # def deploy_swap_charge_rule(model_Bender_SP_v1, i):
    #     return sum(sum((Z_c[itr_while][i][j][k] + Z_s[itr_while][i][j][k]) \
    #                    for k in model_Bender_SP_v1.k if k in Trains[j]['containers']) \
    #                for j in model_Bender_SP_v1.j) \
    #             <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * X[i][itr_while]
    # model_Bender_SP_v1.deploy_swap_charge_rule = pe.Constraint(model_Bender_SP_v1.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('c...')
    # def noboth_swap_charge_rule(model_Bender_SP_v1, i, j, k1, k2):
    #     if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
    #         return Z_s[itr_while][i][j][k1] + Z_c[itr_while][i][j][k2] <= 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.noboth_swap_charge_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k1, model_Bender_SP_v1.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_Bender_SP_v1, i, j):
        return model_Bender_SP_v1.T_depart[i,j] - model_Bender_SP_v1.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_Bender_SP_v1.wait_passenger_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # this constraint should be removed from SP, because it does not contain any variables (Y are constants).
    # print ('e0...')
    # def consecutive_battery_rule(model_Bender_SP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         kid = int(((k.split('container '))[1].split(' in')[0]))
    #         if kid < len(Trains[j]['containers']):
    #             return sum(Y[itr_while][j][kp] \
    #                        for kp in model_Bender_SP_v1.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                    <= sum(1 for kp in model_Bender_SP_v1.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                       * Y[itr_while][j][k]
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.consecutive_battery_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('e...')
    # def max_number_batteries_station_rule(model_Bender_SP_v1, i, j):
    #     return sum(Z_s[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    # model_Bender_SP_v1.max_number_batteries_station_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('f...')
    # def max_number_chargers_station_rule(model_Bender_SP_v1, i, j):
    #     return sum(Z_c[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    # model_Bender_SP_v1.max_number_chargers_station_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_Bender_SP_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_Bender_SP_v1.S_depart[i1,j,k] for k in Trains[j]['containers']) \
                   - sum(model_Bender_SP_v1.S_arrive[i2,j,k] for k in Trains[j]['containers']) \
                   == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.power_rule = pe.Constraint(model_Bender_SP_v1.i1, model_Bender_SP_v1.i2, model_Bender_SP_v1.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule_linear_a(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return - model_Bender_SP_v1.S_arrive[i,j,k] >=  -M * B[itr_while][i][j][k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.battery_sequential_rule_linear_a = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    def battery_sequential_rule_linear_b(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return model_Bender_SP_v1.S_arrive[i,j,k_next] >=  -M+1 + M * B[itr_while][i][j][k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.battery_sequential_rule_linear_b = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.S_depart[i,j,k] - model_Bender_SP_v1.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_increase_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.S_depart[i,j,k] >= Z_s[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.swap_full_soc_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_Bender_SP_v1, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return - model_Bender_SP_v1.S_arrive[i2,j,k] + model_Bender_SP_v1.S_depart[i1,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_depart_arrive_between_stations_rule = pe.Constraint(model_Bender_SP_v1.i1, model_Bender_SP_v1.i2, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.T_depart[i,j] - model_Bender_SP_v1.T_arrive[i,j] >= hour_battery_swap * Z_s[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.time_battery_swap_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_PLA_Zs(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.S_depart[i,j,k] \
                   - model_Bender_SP_v1.F[i,j,k] \
                   >= -1 - M * Z_s[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_PLA_Zc(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.S_depart[i,j,k] \
                   + model_Bender_SP_v1.F[i,j,k] \
                   >= 1 - epsilon * Z_c[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_Zc)        

    # This constraint should be removed from SP,  because it does not have variables.
    # print ('k2...')
    # def soc_time_charge_rule_PLA_beta1(model_Bender_SP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(beta[itr_while][i][j][k][u] for u in model_Bender_SP_v1.u if u != max(model_Bender_SP_v1.u)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    def soc_time_charge_rule_PLA_gamma1(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_SP_v1.gamma[i,j,k,u] for u in model_Bender_SP_v1.u) == 1
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # This constraint should be removed from SP,  because it does not have variables.
    # print ('k4...')
    # def soc_time_charge_rule_PLA_tau1(model_Bender_SP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(tau[itr_while][i][j][k][v] for v in model_Bender_SP_v1.v if v != max(model_Bender_SP_v1.v)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_Bender_SP_v1.gamma[i,j,k,u]) \
                       for u in model_Bender_SP_v1.u) \
                   - model_Bender_SP_v1.S_arrive[i,j,k] \
                   == 0
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(((Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_Bender_SP_v1.eta[i,j,k,v]) \
                       for v in model_Bender_SP_v1.v if v != max(model_Bender_SP_v1.v)) \
                   - model_Bender_SP_v1.T_c[i,j,k] \
                   == -sum((Segments_hour_charge[v]['hour'] * tau[itr_while][i][j][k][v]) \
                           for v in model_Bender_SP_v1.v if v != max(model_Bender_SP_v1.v))
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta(model_Bender_SP_v1, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_Bender_SP_v1.u) and u != max(model_Bender_SP_v1.u):
            return - model_Bender_SP_v1.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u-1] - beta[itr_while][i][j][k][u]
        elif k in Trains[j]['containers'] and u == min(model_Bender_SP_v1.u):
            return - model_Bender_SP_v1.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u]
        elif k in Trains[j]['containers'] and u == max(model_Bender_SP_v1.u):
            return - model_Bender_SP_v1.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u-1]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau(model_Bender_SP_v1, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_SP_v1.v) and v != max(model_Bender_SP_v1.v):
            return - model_Bender_SP_v1.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v-1] - tau[itr_while][i][j][k][v]
        elif k in Trains[j]['containers'] and v == min(model_Bender_SP_v1.v):
            return - model_Bender_SP_v1.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v]
        elif k in Trains[j]['containers'] and v == max(model_Bender_SP_v1.v):
            return - model_Bender_SP_v1.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v-1]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1(model_Bender_SP_v1, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_SP_v1.v) and v != max(model_Bender_SP_v1.v):
            return - model_Bender_SP_v1.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    def soc_time_charge_rule_PLA_F_leq(model_Bender_SP_v1, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_SP_v1.u) and v != max(model_Bender_SP_v1.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return sum((model_Bender_SP_v1.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                       for w in model_Bender_SP_v1.u) \
                   + w * model_Bender_SP_v1.eta[i,j,k,v] \
                   - model_Bender_SP_v1.F[i,j,k] \
                   >= M * (-2 + tau[itr_while][i][j][k][v] + beta[itr_while][i][j][k][u])
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.u, model_Bender_SP_v1.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    def soc_time_charge_rule_PLA_F_geq(model_Bender_SP_v1, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_SP_v1.u) and v != max(model_Bender_SP_v1.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_Bender_SP_v1.F[i,j,k] \
                   - sum((model_Bender_SP_v1.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                         for w in model_Bender_SP_v1.u) \
                   - w * model_Bender_SP_v1.eta[i,j,k,v] \
                   >= M * (-2 + tau[itr_while][i][j][k][v] + beta[itr_while][i][j][k][u])
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.u, model_Bender_SP_v1.v, rule = soc_time_charge_rule_PLA_F_geq)        
    
    # This constraint shoule be removed because it doesn't suite changing charge rate.
    # print ('l...')
    # def soc_time_charge_rule_b(model_Bender_SP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_Bender_SP_v1.T_c[i,j,k] <= 1 - model_Bender_SP_v1.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.soc_time_charge_rule_b = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.T_depart[i,j] - model_Bender_SP_v1.T_arrive[i,j] - model_Bender_SP_v1.T_c[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_c = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.T_c[i,j,k] >= -M * Z_c[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_d = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return M * model_Bender_SP_v1.T_c[i,j,k] >= Z_c[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_time_charge_rule_e = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.S_depart[i,j,k] + model_Bender_SP_v1.S_arrive[i,j,k] >= -Z_c[itr_while][i][j][k] - Z_s[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.no_swap_charge_soc_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_Bender_SP_v1, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.S_depart['origin',j,k] >= Y[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_depart_origin_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_Bender_SP_v1, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v1.S_arrive['origin',j,k] >= Y[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.soc_arrive_origin_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    # This constraint should be removed from SP,  because it does not have variables.
    # def origin_no_chargeswap_rule(model_Bender_SP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         return Z_c[itr_while]['origin'][j][k] + Z_s[itr_while]['origin'][j][k] <= 2 * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.origin_no_chargeswap_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_Bender_SP_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_Bender_SP_v1.T_arrive[i2,j] - model_Bender_SP_v1.T_depart[i1,j] == TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.T_traveltime_rule = pe.Constraint(model_Bender_SP_v1.i1, model_Bender_SP_v1.i2, model_Bender_SP_v1.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('o6...')
    # def nobattery_nochargeswap_rule(model_Bender_SP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(Z_c[itr_while][i][j][k] for i in model_Bender_SP_v1.i) + sum(Z_s[itr_while][i][j][k] for i in model_Bender_SP_v1.i) <= 2*len(Stations) * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.nobattery_nochargeswap_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_Bender_SP_v1, j, k):
        if k in Trains[j]['containers']:
            return - sum(model_Bender_SP_v1.S_arrive[i,j,k] for i in model_Bender_SP_v1.i) \
                   - sum(model_Bender_SP_v1.S_depart[i,j,k] for i in model_Bender_SP_v1.i) \
                   >= -2*len(Stations) * Y[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.nobattery_zerosoc_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.S_arrive[i,j,k] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.ub_soc_arrive_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_Bender_SP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.S_depart[i,j,k] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.ub_soc_depart_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_SP_v1.
    # This constraint should be removed from SP,  because it does not have variables.
    # def preprocessing_Y0_rule(model_Bender_SP_v1, j, k):
    #     if k not in Trains[j]['containers']:
    #         return Y[itr_while][j][k] == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v1.preprocessing_Y0_rule = pe.Constraint(model_Bender_SP_v1.j, model_Bender_SP_v1.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_Bender_SP_v1, j):
        return model_Bender_SP_v1.T_arrive['origin',j] + model_Bender_SP_v1.T_depart['origin',j] == 0
    model_Bender_SP_v1.preprocessing_T0_rule = pe.Constraint(model_Bender_SP_v1.j, rule = preprocessing_T0_rule)
    
    # the value of gamma an eta variables are no greater than 1
    # print ('t...')
    def ub_gamma_rule(model_Bender_SP_v1, i, j, k, u):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.gamma[i,j,k,u] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.ub_gamma_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.u, rule = ub_gamma_rule)
    
    def ub_eta_rule(model_Bender_SP_v1, i, j, k, v):
        if k in Trains[j]['containers']:
            return - model_Bender_SP_v1.eta[i,j,k,v] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v1.ub_eta_rule = pe.Constraint(model_Bender_SP_v1.i, model_Bender_SP_v1.j, model_Bender_SP_v1.k, model_Bender_SP_v1.v, rule = ub_eta_rule)
    
        
    # solve the model
    # print('Solving...')    
    # Solve with Gurobi
    solver = SolverFactory('gurobi')
    solver.options['DualReductions'] = 0  # Ensure extreme ray is available
    solver.options['PreSolve'] = 0  # Disable preprocessing
    solver.options["LogToConsole"] = 0  # Disable Gurobi console output
    results = solver.solve(model_Bender_SP_v1, tee=False)
    
    # Check for infeasibility
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("\nPrimal LP is infeasible. Extracting dual information...")
        upper_bound, lower_bound = 'NA', 'NA'
        gap = 'NA'
        
        # Step 1: Save Pyomo model in LP format
        lp_filename = "pyomo_gurobi_model_Bender_SP_v1.lp"
        model_Bender_SP_v1.write(lp_filename, format='lp')

        # Step 2: Load into Gurobi for advanced analysis
        gurobi_model_Bender_SP_v1 = gp.read(lp_filename)
        gurobi_model_Bender_SP_v1.setParam("Method", 1)  # Ensure Dual Simplex is used
        gurobi_model_Bender_SP_v1.setParam("InfUnbdInfo", 1)  # Enable infeasibility certificate
        gurobi_model_Bender_SP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
        gurobi_model_Bender_SP_v1.optimize()

        # constraint name mapping
        constraint_mapping = {}
        index = 0
        for pyomo_constr in model_Bender_SP_v1.component_objects(Constraint, active=True):
            pyomo_name = pyomo_constr.name
            for index_pyomo in pyomo_constr:
                # print (pyomo_name, index_pyomo)
                constraint = pyomo_constr[index_pyomo]
                gurobi_name = gurobi_model_Bender_SP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                index +=1
                
        # Step 4: Extract the Farkas dual extreme ray (dual infeasibility certificate)
        if gurobi_model_Bender_SP_v1.status == gp.GRB.INFEASIBLE:
            print("Gurobi confirms infeasibility.")

            '''-------------extract a dual extreme point-----------'''
            Q += 1
            pi.update({Q: {}})
            pi[Q].update({'source_iteration': itr_while})
            pi[Q].update({'source_type': 'extreme_point'})
            pi[Q].update({'source_model': 'SP'})
            EP.update({'%d'%itr_while + 'SP': {}})
            EP['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
            EP['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
            # Step 1: Save Pyomo model to a file
            lp_filename = "pyomo_gurobi_model_Bender_SP_v1.lp"
            model_Bender_SP_v1.write(lp_filename, format='lp')
        
            # Step 2: Load the model_Bender_SP_v1 into Gurobi
            gurobi_model_Bender_SP_v1 = gp.read(lp_filename)
            gurobi_model_Bender_SP_v1.setParam("DualReductions", 0)  # Ensure extreme ray is available
            gurobi_model_Bender_SP_v1.setParam("PreSolve", 0)  # Disable preprocessing
            gurobi_model_Bender_SP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
        
            # Step 3: Optimize the model_Bender_SP_v1 with Gurobi (to compute infeasibility certificate)
            gurobi_model_Bender_SP_v1.optimize()
        
            # Step 4: Extract dual values (for constraints) if available    
            try:
                # Fetch duals from the constraints
                dual_values = gurobi_model_Bender_SP_v1.getAttr("Pi", gurobi_model_Bender_SP_v1.getConstrs())
                # Step 5: Map Pyomo constraint names to Gurobi constraint names
                constraint_mapping = {}
                index = 0
                for pyomo_constr in model_Bender_SP_v1.component_objects(Constraint, active=True):
                    pyomo_name = pyomo_constr.name
                    for index_pyomo in pyomo_constr:
                        # print (pyomo_name, index_pyomo)
                        constraint = pyomo_constr[index_pyomo]
                        gurobi_name = gurobi_model_Bender_SP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                        constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                        index +=1
                # Step 6: Print dual values with mapped names
                # print("\nDual values (Dual Infeasibility Certificate):")
                for constr, dual_value in zip(gurobi_model_Bender_SP_v1.getConstrs(), dual_values):
                    original_name = constraint_mapping.get(constr.ConstrName, constr.ConstrName)
                    # print(f"Solver constraint: {constr.ConstrName} <==> Pyomo constraint: {original_name}, Dual: {dual_value}")
                    pyomo_name = original_name.partition(' -- ')[0]
                    index_pyomo = original_name.partition(' -- ')[2]
                    if pyomo_name not in pi[Q]:                    
                        pi[Q].update({pyomo_name: {}})
                    if pyomo_name not in EP['%d'%itr_while + 'SP']:                    
                        EP['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                    # split index_pyomo by ", " and "(" and ")"
                    index_list_quote = index_pyomo.split(', ')
                    if len(index_list_quote) >= 2:
                        index_list_quote[0] = index_list_quote[0][1:]
                        index_list_quote[-1] = index_list_quote[-1][:-1]
                    index_list = [item.strip("'") for item in index_list_quote]
                    pi[Q][pyomo_name].update({tuple(index_list): dual_value})
                    EP['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): dual_value})
            except gp.GurobiError as e:
                print(f"GurobiError while extracting duals: {e}")

            '''-------------extract a dual extreme ray-------------'''
            ER.update({'%d'%itr_while + 'SP': {}})
            ER['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
            ER['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
            # Get Farkas dual values
            print("\nDual Extreme Ray (Farkas Certificate):")
            for constr in gurobi_model_Bender_SP_v1.getConstrs():
                try:
                    farkas_dual = -constr.getAttr("FarkasDual")  # Extract Farkas dual values
                    original_name = constraint_mapping[constr.constrName]
                    pyomo_name = original_name.partition(' -- ')[0]
                    index_pyomo = original_name.partition(' -- ')[2]
                    if pyomo_name not in ER['%d'%itr_while + 'SP']:                    
                        ER['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                    # split index_pyomo by ", " and "(" and ")"
                    index_list_quote = index_pyomo.split(', ')
                    if len(index_list_quote) >= 2:
                        index_list_quote[0] = index_list_quote[0][1:]
                        index_list_quote[-1] = index_list_quote[-1][:-1]
                    index_list = [item.strip("'") for item in index_list_quote]
                    ER['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): farkas_dual})
                    # print(f"Constraint {original_name}: Dual Extreme Ray = {farkas_dual}")
                except gp.GurobiError as e:
                    print(f"Could not retrieve FarkasDual for constraint {constr.constrName}: {e}")

            '''-------------add vector ER['%d'%itr_while + 'SP'] to pi[Q]-------------'''
            Q += 1
            pi.update({Q: {}})
            pi[Q].update({'source_iteration': itr_while})
            pi[Q].update({'source_type': 'extreme_ray'})
            pi[Q].update({'source_model': 'SP'})
            for name in ER['%d'%itr_while + 'SP']:
                if name != 'source_model' and name != 'source_iteration':
                    if name not in pi[Q]:
                        pi[Q].update({name: {}})
                    for ind in ER['%d'%itr_while + 'SP'][name]:
                        er_value = ER['%d'%itr_while + 'SP'][name][ind]
                        pi[Q][name].update({ind: (ratio_ER**itr_while)*er_value})
            
    else:
        print("Primal LP is feasible or has another issue.")
        # calculate optimality gap
        upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
        gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
        
        '''extract a dual extreme point'''
        Q += 1
        pi.update({Q: {}})
        pi[Q].update({'source_iteration': itr_while})
        pi[Q].update({'source_type': 'extreme_point'})
        pi[Q].update({'source_model': 'SP'})
        EP.update({'%d'%itr_while + 'SP': {}})
        EP['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
        EP['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
        # Step 1: Save Pyomo model to a file
        lp_filename = "pyomo_gurobi_model_Bender_SP_v1.lp"
        model_Bender_SP_v1.write(lp_filename, format='lp')
    
        # Step 2: Load the model_Bender_SP_v1 into Gurobi
        gurobi_model_Bender_SP_v1 = gp.read(lp_filename)
        gurobi_model_Bender_SP_v1.setParam("DualReductions", 0)  # Ensure extreme ray is available
        gurobi_model_Bender_SP_v1.setParam("PreSolve", 0)  # Disable preprocessing
        gurobi_model_Bender_SP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
    
        # Step 3: Optimize the model_Bender_SP_v1 with Gurobi (to compute infeasibility certificate)
        gurobi_model_Bender_SP_v1.optimize()
        # Step 4: Extract dual values (for constraints) if available    
        try:
            # Fetch duals from the constraints
            dual_values = gurobi_model_Bender_SP_v1.getAttr("Pi", gurobi_model_Bender_SP_v1.getConstrs())
            # Step 5: Map Pyomo constraint names to Gurobi constraint names
            constraint_mapping = {}
            index = 0
            for pyomo_constr in model_Bender_SP_v1.component_objects(Constraint, active=True):
                pyomo_name = pyomo_constr.name
                for index_pyomo in pyomo_constr:
                    # print (pyomo_name, index_pyomo)
                    constraint = pyomo_constr[index_pyomo]
                    gurobi_name = gurobi_model_Bender_SP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                    constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                    index +=1
            # Step 6: Print dual values with mapped names
            # print("\nDual values (Dual Infeasibility Certificate):")
            for constr, dual_value in zip(gurobi_model_Bender_SP_v1.getConstrs(), dual_values):
                original_name = constraint_mapping.get(constr.ConstrName, constr.ConstrName)
                # print(f"Solver constraint: {constr.ConstrName} <==> Pyomo constraint: {original_name}, Dual: {dual_value}")
                pyomo_name = original_name.partition(' -- ')[0]
                index_pyomo = original_name.partition(' -- ')[2]
                if pyomo_name not in pi[Q]:                    
                    pi[Q].update({pyomo_name: {}})
                if pyomo_name not in EP['%d'%itr_while + 'SP']:                    
                    EP['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                # split index_pyomo by ", " and "(" and ")"
                index_list_quote = index_pyomo.split(', ')
                if len(index_list_quote) >= 2:
                    index_list_quote[0] = index_list_quote[0][1:]
                    index_list_quote[-1] = index_list_quote[-1][:-1]
                index_list = [item.strip("'") for item in index_list_quote]
                pi[Q][pyomo_name].update({tuple(index_list): dual_value})
                EP['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): dual_value})
        except gp.GurobiError as e:
            print(f"GurobiError while extracting duals: {e}")
       
        
    # opt= pyomo.opt.SolverFactory("cplex")
    # # optimality_gap = 0.05
    # # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 900
    # results=opt.solve(model_Bender_SP_v1, tee=False, keepfiles=False)
    # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_Bender_SP_v1, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_Bender_SP_v1)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_Bender_SP_v1, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # calculate obj
    if results.solver.termination_condition == TerminationCondition.optimal:
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SP_v1.i)
        cost_delay = sum(sum((model_Bender_SP_v1.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SP_v1.j) for i in model_Bender_SP_v1.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        obj_SP.update({itr_while:obj})
        if itr_while > 0:
            if obj < UB[itr_while-1]:
                UB[itr_while] = obj
            else:
                UB[itr_while] = UB[itr_while-1]
    else:
        UB[itr_while] = UB[itr_while-1]
        obj_SP.update({itr_while: 'None'})
    
    results_SP.update({itr_while: results})
    
    time_end = time.time()
    time_model = time_end - time_start

    
    
    '''Record variables'''
    # print ('Recording variables...')
    D.update({itr_while:{}})
    S_arrive.update({itr_while:{}})
    S_depart.update({itr_while:{}})
    T_arrive.update({itr_while:{}})
    T_depart.update({itr_while:{}})
    T_c.update({itr_while:{}})
    F.update({itr_while:{}})
    
    for i in Stations:
        D[itr_while].update({i: {}})
        S_arrive[itr_while].update({i: {}})
        S_depart[itr_while].update({i: {}})
        T_arrive[itr_while].update({i: {}})
        T_depart[itr_while].update({i: {}})
        T_c[itr_while].update({i: {}})
        F[itr_while].update({i: {}})
        for j in Trains:
            D[itr_while][i].update({j: model_Bender_SP_v1.D[i,j].value})
            T_arrive[itr_while][i].update({j: model_Bender_SP_v1.T_arrive[i,j].value})
            T_depart[itr_while][i].update({j: model_Bender_SP_v1.T_depart[i,j].value})
            S_arrive[itr_while][i].update({j: {}})
            S_depart[itr_while][i].update({j: {}})
            T_c[itr_while][i].update({j: {}})
            F[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[itr_while][i][j].update({k: model_Bender_SP_v1.S_arrive[i,j,k].value})
                S_depart[itr_while][i][j].update({k: model_Bender_SP_v1.S_depart[i,j,k].value})
                T_c[itr_while][i][j].update({k: model_Bender_SP_v1.T_c[i,j,k].value})
                F[itr_while][i][j].update({k: model_Bender_SP_v1.F[i,j,k].value})
    
    gamma.update({itr_while:{}})
    eta.update({itr_while:{}})
    for i in Stations:
        gamma[itr_while].update({i: {}})
        eta[itr_while].update({i: {}})
        for j in Trains:
            gamma[itr_while][i].update({j: {}})
            eta[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                gamma[itr_while][i][j].update({k: {}})
                eta[itr_while][i][j].update({k: {}})
                for u in Segments_SOC:
                    gamma[itr_while][i][j][k].update({u: model_Bender_SP_v1.gamma[i,j,k,u].value})
                for v in Segments_hour_charge:
                    eta[itr_while][i][j][k].update({v: model_Bender_SP_v1.eta[i,j,k,v].value})
    
    
    return obj_SP, UB, D, S_arrive, S_depart, T_arrive, T_depart, T_c, F, gamma, eta, results_SP, time_model, gap, upper_bound, lower_bound, pi, Q, ER, EP
    












'''This model is the SP' version of Model_Bender_SP_v1. Model_Bender_SP_v1 corresponds to SP in latex, while this model corresponds to SP'.'''
def Model_Bender_SPP_v1(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                       M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, \
                       X, Y, Z_c, Z_s, beta, tau, B, D, S_arrive, S_depart, T_arrive, T_depart, T_c, F, gamma, eta, \
                       X_relint, Y_relint, Z_c_relint, Z_s_relint, beta_relint, tau_relint, B_relint, \
                       pi, Q, ER, EP, itr_while, ratio_ER, obj_SP, results_SP, UB):

    time_start = time.time()
    model_Bender_SPP_v1 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_Bender_SPP_v1.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SPP_v1.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SPP_v1.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SPP_v1.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_Bender_SPP_v1.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SPP_v1.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SPP_v1.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SPP_v1.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_Bender_SPP_v1.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    
    # Variables
    # print ('Defining variables...')
    model_Bender_SPP_v1.D = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SPP_v1.S_arrive = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, within = pe.NonNegativeReals)
    model_Bender_SPP_v1.S_depart = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, within = pe.NonNegativeReals)
    model_Bender_SPP_v1.T_arrive = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SPP_v1.T_depart = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, within = pe.NonNegativeReals)
    model_Bender_SPP_v1.T_c = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, within = pe.NonNegativeReals)  # the amount of charging time of the battery in container k train j at station i
    # variables for linearization
    model_Bender_SPP_v1.F = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_Bender_SPP_v1.gamma = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
    model_Bender_SPP_v1.eta = pe.Var(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c
    # new variable for SPP
    model_Bender_SPP_v1.R = pe.Var([0], within = pe.NonNegativeReals)

    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_Bender_SPP_v1.j:
        model_Bender_SPP_v1.T_arrive['origin',j].fix(0)
        model_Bender_SPP_v1.T_depart['origin',j].fix(0)
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_Bender_SPP_v1):
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SPP_v1.i)
        cost_delay = sum(sum((model_Bender_SPP_v1.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SPP_v1.j) for i in model_Bender_SPP_v1.i)
        for q in pi:
            if pi[q]['source_iteration'] == itr_while and pi[q]['source_type'] == 'extreme_ray' and pi[q]['source_model'] == 'SP':
                p = q
                break
        cost_extra = - (sum(sum((Trains[j]['stations'][i]['time_wait']*pi[p]['wait_passenger_rule'][i, j]) \
                                for j in model_Bender_SPP_v1.j) \
                            for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((Power_train[j][i1][i2] * pi[p]['power_rule'][i1,i2,j]) \
                                      for i2 in model_Bender_SPP_v1.i2 if i2 == Stations[i1]['station_after']) \
                                  for i1 in model_Bender_SPP_v1.i1) \
                              for j in model_Bender_SPP_v1.j) \
                        + sum(sum(sum((- pi[p]['battery_sequential_rule_linear_a'][i,j,k] *  M * B[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum(((1-M) * pi[p]['battery_sequential_rule_linear_b'][i,j,k] + M*pi[p]['battery_sequential_rule_linear_b'][i,j,k] * B[itr_while][i][j][k])\
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers'] and Trains[j]['containers'].index(k) < len(Trains[j]['containers'])-1) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((pi[p]['swap_full_soc_rule'][i,j,k] * Z_s[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((hour_battery_swap * pi[p]['time_battery_swap_rule'][i,j,k] * Z_s[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((-pi[p]['soc_time_charge_rule_PLA_Zs'][i,j,k] \
                                        - M*pi[p]['soc_time_charge_rule_PLA_Zs'][i,j,k] * Z_s[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((pi[p]['soc_time_charge_rule_PLA_Zc'][i,j,k] \
                                        - pi[p]['soc_time_charge_rule_PLA_Zc'][i,j,k]*epsilon * Z_c[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((pi[p]['soc_time_charge_rule_PLA_gamma1'][i,j,k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((-pi[p]['soc_time_charge_rule_PLA_tau_eta_Tc'][i,j,k] * sum((Segments_hour_charge[v]['hour'] * tau[itr_while][i][j][k][v]) \
                                                                                                  for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v))) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * beta[itr_while][i][j][k][u-1] \
                                          - pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * beta[itr_while][i][j][k][u]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for u in model_Bender_SPP_v1.u if u != min(model_Bender_SPP_v1.u) and u != max(model_Bender_SPP_v1.u)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * beta[itr_while][i][j][k][u]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for u in model_Bender_SPP_v1.u if u == min(model_Bender_SPP_v1.u)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_gamma_beta'][i,j,k,str(u)] * beta[itr_while][i][j][k][u-1]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for u in model_Bender_SPP_v1.u if u == max(model_Bender_SPP_v1.u)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * tau[itr_while][i][j][k][v-1] \
                                          - pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * tau[itr_while][i][j][k][v]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for v in model_Bender_SPP_v1.v if v != min(model_Bender_SPP_v1.v) and v != max(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * tau[itr_while][i][j][k][v]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for v in model_Bender_SPP_v1.v if v == min(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau'][i,j,k,str(v)] * tau[itr_while][i][j][k][v-1]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for v in model_Bender_SPP_v1.v if v == max(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum(sum((- pi[p]['soc_time_charge_rule_PLA_eta_tau1'][i,j,k,str(v)] * tau[itr_while][i][j][k][v] ) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for v in model_Bender_SPP_v1.v if v != min(model_Bender_SPP_v1.v) and v != max(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum(sum(sum((- 2*pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M \
                                              + pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * tau[itr_while][i][j][k][v] \
                                              + pi[p]['soc_time_charge_rule_PLA_F_leq'][i,j,k,str(u),str(v)]*M * beta[itr_while][i][j][k][u]) \
                                              for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                          for j in model_Bender_SPP_v1.j) \
                                      for i in model_Bender_SPP_v1.i) \
                                  for u in model_Bender_SPP_v1.u if u != max(model_Bender_SPP_v1.u)) \
                              for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum(sum(sum((- 2*pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M \
                                              + pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * tau[itr_while][i][j][k][v] \
                                              + pi[p]['soc_time_charge_rule_PLA_F_geq'][i,j,k,str(u),str(v)]*M * beta[itr_while][i][j][k][u]) \
                                              for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                          for j in model_Bender_SPP_v1.j) \
                                      for i in model_Bender_SPP_v1.i) \
                                  for u in model_Bender_SPP_v1.u if u != max(model_Bender_SPP_v1.u)) \
                              for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v)) \
                        + sum(sum(sum((- M*pi[p]['soc_time_charge_rule_d'][i,j,k] * Z_c[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((pi[p]['soc_time_charge_rule_e'][i,j,k] * Z_c[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((- pi[p]['no_swap_charge_soc_rule'][i,j,k] * Z_c[itr_while][i][j][k] \
                                      - pi[p]['no_swap_charge_soc_rule'][i,j,k] * Z_s[itr_while][i][j][k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum((pi[p]['soc_depart_origin_rule'][j,k] * Y[itr_while][j][k]) \
                                  for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                              for j in model_Bender_SPP_v1.j) \
                        + sum(sum((pi[p]['soc_arrive_origin_rule'][j,k] * Y[itr_while][j][k]) \
                                  for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                              for j in model_Bender_SPP_v1.j) \
                        + sum(sum(sum((pi[p]['T_traveltime_rule'][i1,i2,j] * TravelTime_train[j][i1][i2]) \
                                      for i2 in model_Bender_SPP_v1.i2 if i2 == Stations[i1]['station_after']) \
                                  for i1 in model_Bender_SPP_v1.i1) \
                              for j in model_Bender_SPP_v1.j) \
                        + sum(sum((-2*len(Stations)*pi[p]['nobattery_zerosoc_rule'][j,k] * Y[itr_while][j][k]) \
                                  for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                              for j in model_Bender_SPP_v1.j) \
                        + sum(sum(sum((-pi[p]['ub_soc_arrive_rule'][i,j,k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum((-pi[p]['ub_soc_depart_rule'][i,j,k]) \
                                      for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                  for j in model_Bender_SPP_v1.j) \
                              for i in model_Bender_SPP_v1.i) \
                        + sum(sum(sum(sum((-pi[p]['ub_gamma_rule'][i,j,k,str(u)]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for u in model_Bender_SPP_v1.u) \
                        + sum(sum(sum(sum((-pi[p]['ub_eta_rule'][i,j,k,str(v)]) \
                                          for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
                                      for j in model_Bender_SPP_v1.j) \
                                  for i in model_Bender_SPP_v1.i) \
                              for v in model_Bender_SPP_v1.v) ) \
                    * model_Bender_SPP_v1.R[0]
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay + cost_extra
        return obj  
    model_Bender_SPP_v1.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_Bender_SPP_v1, i, j):
        return model_Bender_SPP_v1.D[i,j] - model_Bender_SPP_v1.T_depart[i,j] + model_Bender_SPP_v1.T_arrive[i,j] >= 0
    model_Bender_SPP_v1.delay_define_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('b...')
    # def deploy_swap_charge_rule(model_Bender_SPP_v1, i):
    #     return sum(sum((Z_c[itr_while][i][j][k] + Z_s[itr_while][i][j][k]) \
    #                    for k in model_Bender_SPP_v1.k if k in Trains[j]['containers']) \
    #                for j in model_Bender_SPP_v1.j) \
    #             <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * X[i][itr_while]
    # model_Bender_SPP_v1.deploy_swap_charge_rule = pe.Constraint(model_Bender_SPP_v1.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('c...')
    # def noboth_swap_charge_rule(model_Bender_SPP_v1, i, j, k1, k2):
    #     if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
    #         return Z_s[itr_while][i][j][k1] + Z_c[itr_while][i][j][k2] <= 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.noboth_swap_charge_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k1, model_Bender_SPP_v1.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_Bender_SPP_v1, i, j):
        return model_Bender_SPP_v1.T_depart[i,j] - model_Bender_SPP_v1.T_arrive[i,j] - Trains[j]['stations'][i]['time_wait'] * model_Bender_SPP_v1.R[0] \
                >= Trains[j]['stations'][i]['time_wait']
    model_Bender_SPP_v1.wait_passenger_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # this constraint should be removed from SP, because it does not contain any variables (Y are constants).
    # print ('e0...')
    # def consecutive_battery_rule(model_Bender_SPP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         kid = int(((k.split('container '))[1].split(' in')[0]))
    #         if kid < len(Trains[j]['containers']):
    #             return sum(Y[itr_while][j][kp] \
    #                        for kp in model_Bender_SPP_v1.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                    <= sum(1 for kp in model_Bender_SPP_v1.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                       * Y[itr_while][j][k]
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.consecutive_battery_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('e...')
    # def max_number_batteries_station_rule(model_Bender_SPP_v1, i, j):
    #     return sum(Z_s[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    # model_Bender_SPP_v1.max_number_batteries_station_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('f...')
    # def max_number_chargers_station_rule(model_Bender_SPP_v1, i, j):
    #     return sum(Z_c[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    # model_Bender_SPP_v1.max_number_chargers_station_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # print ('g...')
    def power_rule(model_Bender_SPP_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_Bender_SPP_v1.S_depart[i1,j,k] for k in Trains[j]['containers']) \
                   - sum(model_Bender_SPP_v1.S_arrive[i2,j,k] for k in Trains[j]['containers']) \
                   - Power_train[j][i1][i2] * model_Bender_SPP_v1.R[0] \
                   == Power_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.power_rule = pe.Constraint(model_Bender_SPP_v1.i1, model_Bender_SPP_v1.i2, model_Bender_SPP_v1.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # print ('g1...')
    def battery_sequential_rule_linear_a(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return - model_Bender_SPP_v1.S_arrive[i,j,k] + M*B[itr_while][i][j][k] * model_Bender_SPP_v1.R[0] \
                       >= -M * B_relint[itr_while][i][j][k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.battery_sequential_rule_linear_a = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    def battery_sequential_rule_linear_b(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return model_Bender_SPP_v1.S_arrive[i,j,k_next] \
                       + (M-1 - M * B[itr_while][i][j][k]) * model_Bender_SPP_v1.R[0] \
                       >= -M+1 + M * B_relint[itr_while][i][j][k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.battery_sequential_rule_linear_b = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # print ('h...')
    def soc_increase_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.S_depart[i,j,k] - model_Bender_SPP_v1.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_increase_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # print ('i...')
    def swap_full_soc_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.S_depart[i,j,k] - Z_s[itr_while][i][j][k] * model_Bender_SPP_v1.R[0] >= Z_s_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.swap_full_soc_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # print ('i1...')
    def soc_depart_arrive_between_stations_rule(model_Bender_SPP_v1, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return - model_Bender_SPP_v1.S_arrive[i2,j,k] + model_Bender_SPP_v1.S_depart[i1,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_depart_arrive_between_stations_rule = pe.Constraint(model_Bender_SPP_v1.i1, model_Bender_SPP_v1.i2, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.T_depart[i,j] - model_Bender_SPP_v1.T_arrive[i,j] \
                   - hour_battery_swap*Z_s[itr_while][i][j][k] * model_Bender_SPP_v1.R[0] \
                   >= hour_battery_swap * Z_s_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.time_battery_swap_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # print ('k0...')
    def soc_time_charge_rule_PLA_Zs(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.S_depart[i,j,k] \
                   - model_Bender_SPP_v1.F[i,j,k] \
                   + (1 + M*Z_s[itr_while][i][j][k]) * model_Bender_SPP_v1.R[0] \
                   >= -1 - M * Z_s_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    def soc_time_charge_rule_PLA_Zc(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.S_depart[i,j,k] \
                   + model_Bender_SPP_v1.F[i,j,k] \
                   - (1 - epsilon * Z_c[itr_while][i][j][k]) * model_Bender_SPP_v1.R[0] \
                   >= 1 - epsilon * Z_c_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_Zc)        

    # This constraint should be removed from SP,  because it does not have variables.
    # print ('k2...')
    # def soc_time_charge_rule_PLA_beta1(model_Bender_SPP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(beta[itr_while][i][j][k][u] for u in model_Bender_SPP_v1.u if u != max(model_Bender_SPP_v1.u)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    def soc_time_charge_rule_PLA_gamma1(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_SPP_v1.gamma[i,j,k,u] for u in model_Bender_SPP_v1.u) - model_Bender_SPP_v1.R[0] == 1
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # This constraint should be removed from SP,  because it does not have variables.
    # print ('k4...')
    # def soc_time_charge_rule_PLA_tau1(model_Bender_SPP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(tau[itr_while][i][j][k][v] for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_Bender_SPP_v1.gamma[i,j,k,u]) \
                       for u in model_Bender_SPP_v1.u) \
                   - model_Bender_SPP_v1.S_arrive[i,j,k] \
                   == 0
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return sum(((Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_Bender_SPP_v1.eta[i,j,k,v]) \
                       for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v)) \
                   - model_Bender_SPP_v1.T_c[i,j,k] \
                   + sum((Segments_hour_charge[v]['hour'] * tau[itr_while][i][j][k][v]) \
                           for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v)) \
                      * model_Bender_SPP_v1.R[0] \
                   == -sum((Segments_hour_charge[v]['hour'] * tau_relint[itr_while][i][j][k][v]) \
                           for v in model_Bender_SPP_v1.v if v != max(model_Bender_SPP_v1.v))
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta(model_Bender_SPP_v1, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_Bender_SPP_v1.u) and u != max(model_Bender_SPP_v1.u):
            return - model_Bender_SPP_v1.gamma[i,j,k,u] + (beta[itr_while][i][j][k][u-1] + beta[itr_while][i][j][k][u]) * model_Bender_SPP_v1.R[0] \
                   >= -beta_relint[itr_while][i][j][k][u-1] - beta_relint[itr_while][i][j][k][u]
        elif k in Trains[j]['containers'] and u == min(model_Bender_SPP_v1.u):
            return - model_Bender_SPP_v1.gamma[i,j,k,u] + beta[itr_while][i][j][k][u] * model_Bender_SPP_v1.R[0] \
                   >= -beta_relint[itr_while][i][j][k][u]
        elif k in Trains[j]['containers'] and u == max(model_Bender_SPP_v1.u):
            return - model_Bender_SPP_v1.gamma[i,j,k,u] + beta[itr_while][i][j][k][u-1] * model_Bender_SPP_v1.R[0] \
                   >= -beta_relint[itr_while][i][j][k][u-1]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau(model_Bender_SPP_v1, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_SPP_v1.v) and v != max(model_Bender_SPP_v1.v):
            return - model_Bender_SPP_v1.eta[i,j,k,v] + (tau[itr_while][i][j][k][v-1] + tau[itr_while][i][j][k][v]) * model_Bender_SPP_v1.R[0] \
                   >= -tau_relint[itr_while][i][j][k][v-1] - tau_relint[itr_while][i][j][k][v]
        elif k in Trains[j]['containers'] and v == min(model_Bender_SPP_v1.v):
            return - model_Bender_SPP_v1.eta[i,j,k,v] + tau[itr_while][i][j][k][v] * model_Bender_SPP_v1.R[0] \
                   >= -tau_relint[itr_while][i][j][k][v]
        elif k in Trains[j]['containers'] and v == max(model_Bender_SPP_v1.v):
            return - model_Bender_SPP_v1.eta[i,j,k,v] + tau[itr_while][i][j][k][v-1] * model_Bender_SPP_v1.R[0] \
                   >= -tau_relint[itr_while][i][j][k][v-1]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1(model_Bender_SPP_v1, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_SPP_v1.v) and v != max(model_Bender_SPP_v1.v):
            return - model_Bender_SPP_v1.eta[i,j,k,v] + tau[itr_while][i][j][k][v] * model_Bender_SPP_v1.R[0] \
                   >= -tau_relint[itr_while][i][j][k][v]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    def soc_time_charge_rule_PLA_F_leq(model_Bender_SPP_v1, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_SPP_v1.u) and v != max(model_Bender_SPP_v1.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return sum((model_Bender_SPP_v1.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                       for w in model_Bender_SPP_v1.u) \
                   + w * model_Bender_SPP_v1.eta[i,j,k,v] \
                   - model_Bender_SPP_v1.F[i,j,k] \
                   + M * (2 - tau[itr_while][i][j][k][v] - beta[itr_while][i][j][k][u]) * model_Bender_SPP_v1.R[0] \
                   >= M * (-2 + tau_relint[itr_while][i][j][k][v] + beta_relint[itr_while][i][j][k][u])
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.u, model_Bender_SPP_v1.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    def soc_time_charge_rule_PLA_F_geq(model_Bender_SPP_v1, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_SPP_v1.u) and v != max(model_Bender_SPP_v1.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_Bender_SPP_v1.F[i,j,k] \
                   - sum((model_Bender_SPP_v1.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                         for w in model_Bender_SPP_v1.u) \
                   - w * model_Bender_SPP_v1.eta[i,j,k,v] \
                   + M * (2 - tau[itr_while][i][j][k][v] - beta[itr_while][i][j][k][u]) * model_Bender_SPP_v1.R[0] \
                   >= M * (-2 + tau_relint[itr_while][i][j][k][v] + beta_relint[itr_while][i][j][k][u])
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.u, model_Bender_SPP_v1.v, rule = soc_time_charge_rule_PLA_F_geq)        
    
    # This constraint shoule be removed because it doesn't suite changing charge rate.
    # print ('l...')
    # def soc_time_charge_rule_b(model_Bender_SPP_v1, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_Bender_SPP_v1.T_c[i,j,k] <= 1 - model_Bender_SPP_v1.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.soc_time_charge_rule_b = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.T_depart[i,j] - model_Bender_SPP_v1.T_arrive[i,j] - model_Bender_SPP_v1.T_c[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_c = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    def soc_time_charge_rule_d(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.T_c[i,j,k] + M*Z_c[itr_while][i][j][k] * model_Bender_SPP_v1.R[0] \
                   >= -M * Z_c_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_d = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    def soc_time_charge_rule_e(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return M * model_Bender_SPP_v1.T_c[i,j,k] - Z_c[itr_while][i][j][k] * model_Bender_SPP_v1.R[0] \
                   >= Z_c_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_time_charge_rule_e = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # print ('o...')
    def no_swap_charge_soc_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.S_depart[i,j,k] + model_Bender_SPP_v1.S_arrive[i,j,k] \
                   + (Z_c[itr_while][i][j][k] + Z_s[itr_while][i][j][k]) * model_Bender_SPP_v1.R[0] \
                   >= -Z_c_relint[itr_while][i][j][k] - Z_s_relint[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.no_swap_charge_soc_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # print ('o2...')
    def soc_depart_origin_rule(model_Bender_SPP_v1, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.S_depart['origin',j,k] - Y[itr_while][j][k] * model_Bender_SPP_v1.R[0] >= Y_relint[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_depart_origin_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # print ('o3...')
    def soc_arrive_origin_rule(model_Bender_SPP_v1, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SPP_v1.S_arrive['origin',j,k] - Y[itr_while][j][k] * model_Bender_SPP_v1.R[0] >= Y_relint[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.soc_arrive_origin_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    # This constraint should be removed from SP,  because it does not have variables.
    # def origin_no_chargeswap_rule(model_Bender_SPP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         return Z_c[itr_while]['origin'][j][k] + Z_s[itr_while]['origin'][j][k] <= 2 * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.origin_no_chargeswap_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_Bender_SPP_v1, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_Bender_SPP_v1.T_arrive[i2,j] - model_Bender_SPP_v1.T_depart[i1,j] \
                   - TravelTime_train[j][i1][i2] * model_Bender_SPP_v1.R[0] \
                   == TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.T_traveltime_rule = pe.Constraint(model_Bender_SPP_v1.i1, model_Bender_SPP_v1.i2, model_Bender_SPP_v1.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('o6...')
    # def nobattery_nochargeswap_rule(model_Bender_SPP_v1, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(Z_c[itr_while][i][j][k] for i in model_Bender_SPP_v1.i) + sum(Z_s[itr_while][i][j][k] for i in model_Bender_SPP_v1.i) <= 2*len(Stations) * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.nobattery_nochargeswap_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # print ('o7...')
    def nobattery_zerosoc_rule(model_Bender_SPP_v1, j, k):
        if k in Trains[j]['containers']:
            return - sum(model_Bender_SPP_v1.S_arrive[i,j,k] for i in model_Bender_SPP_v1.i) \
                   - sum(model_Bender_SPP_v1.S_depart[i,j,k] for i in model_Bender_SPP_v1.i) \
                   + 2*len(Stations) * Y[itr_while][j][k] * model_Bender_SPP_v1.R[0] \
                   >= -2*len(Stations) * Y_relint[itr_while][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.nobattery_zerosoc_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # print ('p...')
    def ub_soc_arrive_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.S_arrive[i,j,k] + model_Bender_SPP_v1.R[0] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.ub_soc_arrive_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = ub_soc_arrive_rule)
    
    # print ('q...')
    def ub_soc_depart_rule(model_Bender_SPP_v1, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.S_depart[i,j,k] + model_Bender_SPP_v1.R[0] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.ub_soc_depart_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_SPP_v1.
    # This constraint should be removed from SP,  because it does not have variables.
    # def preprocessing_Y0_rule(model_Bender_SPP_v1, j, k):
    #     if k not in Trains[j]['containers']:
    #         return Y[itr_while][j][k] == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SPP_v1.preprocessing_Y0_rule = pe.Constraint(model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_Bender_SPP_v1, j):
        return model_Bender_SPP_v1.T_arrive['origin',j] + model_Bender_SPP_v1.T_depart['origin',j] == 0
    model_Bender_SPP_v1.preprocessing_T0_rule = pe.Constraint(model_Bender_SPP_v1.j, rule = preprocessing_T0_rule)
    
    # print ('t...')
    # the value of gamma an eta variables are no greater than 1
    def ub_gamma_rule(model_Bender_SPP_v1, i, j, k, u):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.gamma[i,j,k,u] + model_Bender_SPP_v1.R[0] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.ub_gamma_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.u, rule = ub_gamma_rule)
    
    def ub_eta_rule(model_Bender_SPP_v1, i, j, k, v):
        if k in Trains[j]['containers']:
            return - model_Bender_SPP_v1.eta[i,j,k,v] + model_Bender_SPP_v1.R[0] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_SPP_v1.ub_eta_rule = pe.Constraint(model_Bender_SPP_v1.i, model_Bender_SPP_v1.j, model_Bender_SPP_v1.k, model_Bender_SPP_v1.v, rule = ub_eta_rule)
    
        
    # solve the model
    # print('Solving...')    
    # Solve with Gurobi
    solver = SolverFactory('gurobi')
    solver.options['DualReductions'] = 0  # Ensure extreme ray is available
    solver.options['PreSolve'] = 0  # Disable preprocessing
    solver.options["LogToConsole"] = 0  # Disable Gurobi console output
    results = solver.solve(model_Bender_SPP_v1, tee=False)
    
    # Check for infeasibility
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("\nPrimal LP is infeasible. Extracting dual information...")
        upper_bound, lower_bound = 'NA', 'NA'
        gap = 'NA'
        
        # Step 1: Save Pyomo model in LP format
        lp_filename = "pyomo_gurobi_model_Bender_SPP_v1.lp"
        model_Bender_SPP_v1.write(lp_filename, format='lp')

        # Step 2: Load into Gurobi for advanced analysis
        gurobi_model_Bender_SPP_v1 = gp.read(lp_filename)
        gurobi_model_Bender_SPP_v1.setParam("Method", 1)  # Ensure Dual Simplex is used
        gurobi_model_Bender_SPP_v1.setParam("InfUnbdInfo", 1)  # Enable infeasibility certificate
        gurobi_model_Bender_SPP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
        gurobi_model_Bender_SPP_v1.optimize()

        # constraint name mapping
        constraint_mapping = {}
        index = 0
        for pyomo_constr in model_Bender_SPP_v1.component_objects(Constraint, active=True):
            pyomo_name = pyomo_constr.name
            for index_pyomo in pyomo_constr:
                # print (pyomo_name, index_pyomo)
                constraint = pyomo_constr[index_pyomo]
                gurobi_name = gurobi_model_Bender_SPP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                index +=1
                
        # Step 4: Extract the Farkas dual extreme ray (dual infeasibility certificate)
        if gurobi_model_Bender_SPP_v1.status == gp.GRB.INFEASIBLE:
            print("Gurobi confirms infeasibility.")

            '''-------------extract a dual extreme point-----------'''
            # Step 1: Save Pyomo model to a file
            lp_filename = "pyomo_gurobi_model_Bender_SPP_v1.lp"
            model_Bender_SPP_v1.write(lp_filename, format='lp')
        
            # Step 2: Load the model_Bender_SPP_v1 into Gurobi
            gurobi_model_Bender_SPP_v1 = gp.read(lp_filename)
            gurobi_model_Bender_SPP_v1.setParam("DualReductions", 0)  # Ensure extreme ray is available
            gurobi_model_Bender_SPP_v1.setParam("PreSolve", 0)  # Disable preprocessing
            gurobi_model_Bender_SPP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
        
            # Step 3: Optimize the model_Bender_SPP_v1 with Gurobi (to compute infeasibility certificate)
            gurobi_model_Bender_SPP_v1.optimize()
        
            # Step 4: Extract dual values (for constraints) if available    
            try:
                # Fetch duals from the constraints
                dual_values = gurobi_model_Bender_SPP_v1.getAttr("Pi", gurobi_model_Bender_SPP_v1.getConstrs())
                # Step 5: Map Pyomo constraint names to Gurobi constraint names
                constraint_mapping = {}
                index = 0
                for pyomo_constr in model_Bender_SPP_v1.component_objects(Constraint, active=True):
                    pyomo_name = pyomo_constr.name
                    for index_pyomo in pyomo_constr:
                        # print (pyomo_name, index_pyomo)
                        constraint = pyomo_constr[index_pyomo]
                        gurobi_name = gurobi_model_Bender_SPP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                        constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                        index +=1
            except gp.GurobiError as e:
                print(f"GurobiError while extracting duals: {e}")
                
                
            '''-------------extract a dual extreme ray-------------'''
            ER.update({'%d'%itr_while + 'SPP': {}})
            ER['%d'%itr_while + 'SPP'].update({'source_model': 'SPP'})
            ER['%d'%itr_while + 'SPP'].update({'source_iteration': itr_while})
            # Get Farkas dual values
            print("\nDual Extreme Ray (Farkas Certificate):")
            for constr in gurobi_model_Bender_SPP_v1.getConstrs():
                try:
                    farkas_dual = -constr.getAttr("FarkasDual")  # Extract Farkas dual values
                    original_name = constraint_mapping[constr.constrName]
                    pyomo_name = original_name.partition(' -- ')[0]
                    index_pyomo = original_name.partition(' -- ')[2]
                    if pyomo_name not in ER['%d'%itr_while + 'SPP']:                    
                        ER['%d'%itr_while + 'SPP'].update({pyomo_name: {}})
                    # split index_pyomo by ", " and "(" and ")"
                    index_list_quote = index_pyomo.split(', ')
                    if len(index_list_quote) >= 2:
                        index_list_quote[0] = index_list_quote[0][1:]
                        index_list_quote[-1] = index_list_quote[-1][:-1]
                    index_list = [item.strip("'") for item in index_list_quote]
                    ER['%d'%itr_while + 'SPP'][pyomo_name].update({tuple(index_list): farkas_dual})
                    # print(f"Constraint {original_name}: Dual Extreme Ray = {farkas_dual}")
                except gp.GurobiError as e:
                    print(f"Could not retrieve FarkasDual for constraint {constr.constrName}: {e}")

            '''-------------add vector ER['%d'%itr_while + 'SPP'] to pi[Q]-------------'''
            Q += 1
            pi.update({Q: {}})
            pi[Q].update({'source_iteration': itr_while})
            pi[Q].update({'source_type': 'extreme_ray'})
            pi[Q].update({'source_model': 'SPP'})
            for name in ER['%d'%itr_while + 'SPP']:
                if name != 'source_model' and name != 'source_iteration':
                    if name not in pi[Q]:
                        pi[Q].update({name: {}})
                    for ind in ER['%d'%itr_while + 'SPP'][name]:
                        er_value = ER['%d'%itr_while + 'SPP'][name][ind]
                        pi[Q][name].update({ind: (ratio_ER**itr_while)*er_value})
            
    else:
        print("Primal LP is feasible or has another issue.")
        # calculate optimality gap
        upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
        gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
        
        '''extract a dual extreme point'''
        Q += 1
        pi.update({Q: {}})
        pi[Q].update({'source_iteration': itr_while})
        pi[Q].update({'source_type': 'extreme_point'})
        pi[Q].update({'source_model': 'SPP'})
        EP.update({'%d'%itr_while + 'SPP': {}})
        EP['%d'%itr_while + 'SPP'].update({'source_model': 'SPP'})
        EP['%d'%itr_while + 'SPP'].update({'source_iteration': itr_while})
        # Step 1: Save Pyomo model to a file
        lp_filename = "pyomo_gurobi_model_Bender_SPP_v1.lp"
        model_Bender_SPP_v1.write(lp_filename, format='lp')
    
        # Step 2: Load the model_Bender_SPP_v1 into Gurobi
        gurobi_model_Bender_SPP_v1 = gp.read(lp_filename)
        gurobi_model_Bender_SPP_v1.setParam("DualReductions", 0)  # Ensure extreme ray is available
        gurobi_model_Bender_SPP_v1.setParam("PreSolve", 0)  # Disable preprocessing
        gurobi_model_Bender_SPP_v1.setParam("OutputFlag", 0)  # Disable Gurobi console output
    
        # Step 3: Optimize the model_Bender_SPP_v1 with Gurobi (to compute infeasibility certificate)
        gurobi_model_Bender_SPP_v1.optimize()
        # Step 4: Extract dual values (for constraints) if available    
        try:
            # Fetch duals from the constraints
            dual_values = gurobi_model_Bender_SPP_v1.getAttr("Pi", gurobi_model_Bender_SPP_v1.getConstrs())
            # Step 5: Map Pyomo constraint names to Gurobi constraint names
            constraint_mapping = {}
            index = 0
            for pyomo_constr in model_Bender_SPP_v1.component_objects(Constraint, active=True):
                pyomo_name = pyomo_constr.name
                for index_pyomo in pyomo_constr:
                    # print (pyomo_name, index_pyomo)
                    constraint = pyomo_constr[index_pyomo]
                    gurobi_name = gurobi_model_Bender_SPP_v1.getConstrs()[index].ConstrName  # Gurobi-generated name
                    constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                    index +=1
            # Step 6: Print dual values with mapped names
            # print("\nDual values (Dual Infeasibility Certificate):")
            for constr, dual_value in zip(gurobi_model_Bender_SPP_v1.getConstrs(), dual_values):
                original_name = constraint_mapping.get(constr.ConstrName, constr.ConstrName)
                # print(f"Solver constraint: {constr.ConstrName} <==> Pyomo constraint: {original_name}, Dual: {dual_value}")
                pyomo_name = original_name.partition(' -- ')[0]
                index_pyomo = original_name.partition(' -- ')[2]
                if pyomo_name not in pi[Q]:                    
                    pi[Q].update({pyomo_name: {}})
                if pyomo_name not in EP['%d'%itr_while + 'SPP']:                    
                    EP['%d'%itr_while + 'SPP'].update({pyomo_name: {}})
                # split index_pyomo by ", " and "(" and ")"
                index_list_quote = index_pyomo.split(', ')
                if len(index_list_quote) >= 2:
                    index_list_quote[0] = index_list_quote[0][1:]
                    index_list_quote[-1] = index_list_quote[-1][:-1]
                index_list = [item.strip("'") for item in index_list_quote]
                pi[Q][pyomo_name].update({tuple(index_list): dual_value})
                EP['%d'%itr_while + 'SPP'][pyomo_name].update({tuple(index_list): dual_value})
        except gp.GurobiError as e:
            print(f"GurobiError while extracting duals: {e}")
       
        
    # opt= pyomo.opt.SolverFactory("cplex")
    # # optimality_gap = 0.05
    # # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 900
    # results=opt.solve(model_Bender_SPP_v1, tee=False, keepfiles=False)
    # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_Bender_SPP_v1, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_Bender_SPP_v1)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_Bender_SPP_v1, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # calculate obj
    if results.solver.termination_condition == TerminationCondition.optimal:
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SPP_v1.i)
        cost_delay = sum(sum((model_Bender_SPP_v1.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SPP_v1.j) for i in model_Bender_SPP_v1.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        obj_SP.update({itr_while:obj})
        if itr_while > 0:
            if obj < UB[itr_while-1]:
                UB[itr_while] = obj
            else:
                UB[itr_while] = UB[itr_while-1]
    else:
        UB[itr_while] = UB[itr_while-1]
        obj_SP.update({itr_while: 'None'})
    
    results_SP.update({itr_while: results})
    
    time_end = time.time()
    time_model = time_end - time_start

    
    
    '''Record variables'''
    # print ('Recording variables...')
    D.update({itr_while:{}})
    S_arrive.update({itr_while:{}})
    S_depart.update({itr_while:{}})
    T_arrive.update({itr_while:{}})
    T_depart.update({itr_while:{}})
    T_c.update({itr_while:{}})
    F.update({itr_while:{}})
    
    for i in Stations:
        D[itr_while].update({i: {}})
        S_arrive[itr_while].update({i: {}})
        S_depart[itr_while].update({i: {}})
        T_arrive[itr_while].update({i: {}})
        T_depart[itr_while].update({i: {}})
        T_c[itr_while].update({i: {}})
        F[itr_while].update({i: {}})
        for j in Trains:
            D[itr_while][i].update({j: model_Bender_SPP_v1.D[i,j].value})
            T_arrive[itr_while][i].update({j: model_Bender_SPP_v1.T_arrive[i,j].value})
            T_depart[itr_while][i].update({j: model_Bender_SPP_v1.T_depart[i,j].value})
            S_arrive[itr_while][i].update({j: {}})
            S_depart[itr_while][i].update({j: {}})
            T_c[itr_while][i].update({j: {}})
            F[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                S_arrive[itr_while][i][j].update({k: model_Bender_SPP_v1.S_arrive[i,j,k].value})
                S_depart[itr_while][i][j].update({k: model_Bender_SPP_v1.S_depart[i,j,k].value})
                T_c[itr_while][i][j].update({k: model_Bender_SPP_v1.T_c[i,j,k].value})
                F[itr_while][i][j].update({k: model_Bender_SPP_v1.F[i,j,k].value})
    
    gamma.update({itr_while:{}})
    eta.update({itr_while:{}})
    for i in Stations:
        gamma[itr_while].update({i: {}})
        eta[itr_while].update({i: {}})
        for j in Trains:
            gamma[itr_while][i].update({j: {}})
            eta[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                gamma[itr_while][i][j].update({k: {}})
                eta[itr_while][i][j].update({k: {}})
                for u in Segments_SOC:
                    gamma[itr_while][i][j][k].update({u: model_Bender_SPP_v1.gamma[i,j,k,u].value})
                for v in Segments_hour_charge:
                    eta[itr_while][i][j][k].update({v: model_Bender_SPP_v1.eta[i,j,k,v].value})
    
    
    return obj_SP, UB, D, S_arrive, S_depart, T_arrive, T_depart, T_c, F, gamma, eta, results_SP, time_model, gap, upper_bound, lower_bound, pi, Q, ER, EP
    
















''' This model is based on Model_Bender_RMP_v2. 
The difference between this model (v3) and Model_Bender_RMP_v2 is: in addition to variables X, Y, Z, B, beta and tau, this model also includes variables S, gamma, eta and F.
'''
def Model_Bender_RMP_v3(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                        M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, \
                        X, Y, Z_c, Z_s, beta, tau, B, S_arrive, S_depart, T_c, gamma, eta, F, W, pi, Q, results_SP, UB, itr_while, ratio_UB, obj_RMP, results_RMP, LB, gap_RMP, time_RMP):
    time_start = time.time()
    model_Bender_RMP_v3 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # extract keys from pi and classify keys for DSP extreme points and DSP extreme rays respectively
    er_set = set({})
    ep_set = set({})
    for q in pi:
        if pi[q]['source_type'] == 'extreme_point':
            ep_set.add(q)
        elif pi[q]['source_type'] == 'extreme_ray':
            er_set.add(q)
    # Sets and indices
    model_Bender_RMP_v3.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v3.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v3.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_RMP_v3.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_Bender_RMP_v3.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v3.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v3.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_RMP_v3.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_Bender_RMP_v3.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    # model_Bender_RMP_v3.p = pe.Set(initialize = sorted(set(pi.keys())))
    model_Bender_RMP_v3.p = pe.Set(initialize = sorted(ep_set))
    # model_Bender_RMP_v3.r = pe.Set(initialize = sorted(set(pi.keys())))
    model_Bender_RMP_v3.r = pe.Set(initialize = sorted(er_set))
    
    # Variables
    # print ('Defining variables...')
    model_Bender_RMP_v3.X = pe.Var(model_Bender_RMP_v3.i, within = pe.Binary)
    model_Bender_RMP_v3.Y = pe.Var(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.Binary)
    model_Bender_RMP_v3.Z_c = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.Binary)
    model_Bender_RMP_v3.Z_s = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.Binary)
    model_Bender_RMP_v3.S_arrive = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.NonNegativeReals)
    model_Bender_RMP_v3.S_depart = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.NonNegativeReals)
    model_Bender_RMP_v3.T_c = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.NonNegativeReals)
    # variables for linearization
    model_Bender_RMP_v3.B = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.Binary)  # binary variables to linearize constraints battery_sequential_rule
    model_Bender_RMP_v3.beta = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, within = pe.Binary)  # in PLA, binary variables for S_arrive
    model_Bender_RMP_v3.tau = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.v, within = pe.Binary)  # in PLA, binary variables for T_c
    model_Bender_RMP_v3.F = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, within = pe.NonNegativeReals)  # in PLA, nonnegative continuous variables to replace (1 - S_arrive) * (1-r)**T_c
    model_Bender_RMP_v3.gamma = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, within = pe.NonNegativeReals)  # in PLA, continuous variables for S_arrive
    model_Bender_RMP_v3.eta = pe.Var(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.v, within = pe.NonNegativeReals)  # in PLA, continuous variables for T_c

    # introduce new RMP variable: W
    model_Bender_RMP_v3.W = pe.Var([0], within = pe.Reals)
    
    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_RMP_v3.
    for j in model_Bender_RMP_v3.j:
        for k in model_Bender_RMP_v3.k:
            # model_Bender_RMP_v3.Z_c['origin',j,k].fix(0) # batteries are not charged in the origin
            # model_Bender_RMP_v3.Z_s['origin',j,k].fix(0) # batteries are not swapped in the origin
            if k not in Trains[j]['containers']:
                model_Bender_RMP_v3.Y[j,k].fix(0)
            else:
                # to minimize the number of stations, we assign as many batteries to each train as possible, so each consist carries a battery
                model_Bender_RMP_v3.Y[j,k].fix(1)
    # If consist k is not in train j, then the corresponding Z, B, beta and tau variables are all zero.
    # for i in model_Bender_RMP_v3.i:
    #     for j in model_Bender_RMP_v3.j:
    #         for k in model_Bender_RMP_v3.k:
    #             if k not in Trains[j]['containers']:
    #                 model_Bender_RMP_v3.Z_c[i,j,k].fix(0)
    #                 model_Bender_RMP_v3.Z_s[i,j,k].fix(0)
    #                 model_Bender_RMP_v3.B[i,j,k].fix(0)
    #                 for u in model_Bender_RMP_v3.u:
    #                     model_Bender_RMP_v3.beta[i,j,k,u].fix(0)
    #                 for v in model_Bender_RMP_v3.v:
    #                     model_Bender_RMP_v3.tau[i,j,k,v].fix(0)
    
    
    # objective function
    print ('Reading objective...')
    def obj_rule(model_Bender_RMP_v3):
        cost_fix = sum(Stations[i]['cost_fix'] * model_Bender_RMP_v3.X[i] for i in model_Bender_RMP_v3.i)
        constant = - penalty_delay * sum(sum(Trains[j]['stations'][i]['time_wait'] for j in model_Bender_RMP_v3.j) for i in model_Bender_RMP_v3.i)
        obj = penalty_fix_cost * cost_fix + model_Bender_RMP_v3.W[0] + constant
        return obj  
    model_Bender_RMP_v3.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints    
    # add feasibility cut from extreme ray
    def extreme_ray_rule(model_Bender_RMP_v3, r):
        return 0 >= \
                + sum(sum((Trains[j]['stations'][i]['time_wait']*pi[r]['wait_passenger_rule'][i, j]) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((hour_battery_swap * pi[r]['time_battery_swap_rule'][i,j,k] * model_Bender_RMP_v3.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((pi[r]['soc_time_charge_rule_c'][i,j,k] * model_Bender_RMP_v3.T_c[i,j,k]) \
                              for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((TravelTime_train[j][i1][i2] * pi[r]['T_traveltime_rule'][i1,i2,j]) \
                              for j in model_Bender_RMP_v3.j) \
                          for i2 in model_Bender_RMP_v3.i2 if i2 == Stations[i1]['station_after']) \
                      for i1 in model_Bender_RMP_v3.i1)
    model_Bender_RMP_v3.extreme_ray_rule = pe.Constraint(model_Bender_RMP_v3.r, rule = extreme_ray_rule)
    
    # add optimality cut from extreme ray
    def extreme_point_rule(model_Bender_RMP_v3, p):
        return model_Bender_RMP_v3.W[0] >= \
                + sum(sum((Trains[j]['stations'][i]['time_wait']*pi[p]['wait_passenger_rule'][i, j]) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((hour_battery_swap * pi[p]['time_battery_swap_rule'][i,j,k] * model_Bender_RMP_v3.Z_s[i,j,k]) \
                              for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((pi[p]['soc_time_charge_rule_c'][i,j,k] * model_Bender_RMP_v3.T_c[i,j,k]) \
                              for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                          for j in model_Bender_RMP_v3.j) \
                      for i in model_Bender_RMP_v3.i) \
                + sum(sum(sum((TravelTime_train[j][i1][i2] * pi[p]['T_traveltime_rule'][i1,i2,j]) \
                              for j in model_Bender_RMP_v3.j) \
                          for i2 in model_Bender_RMP_v3.i2 if i2 == Stations[i1]['station_after']) \
                      for i1 in model_Bender_RMP_v3.i1)
    model_Bender_RMP_v3.extreme_point_rule = pe.Constraint(model_Bender_RMP_v3.p, rule = extreme_point_rule)

    '''add extra feasibility cuts: '''
    # # This optimality is from Fix Algorithm. The idea is: the power from deployed stations must >= the power required on the route. 
    # # The calculation of power_demand, power_supply_origin, and power_min_station is same as that from FRE_FixAlg_v1.py.
    # # print ('opt1...')
    # def power_support_rule_feas(model_Bender_RMP_v3, j):
    #     power_demand = sum(Power_train[j][i][Stations[i]['station_after']] \
    #                             for i in Stations if i != 'destination') 
    #     power_supply_origin = len(Trains[j]['containers'])
    #     power_min_station = power_demand - power_supply_origin  # the minimum amount of power train j requires from stations
    #     return sum((Stations[i]['max_power_provide'] * model_Bender_RMP_v3.X[i]) \
    #                 for i in model_Bender_RMP_v3.i) \
    #             >= power_min_station
    # model_Bender_RMP_v3.power_support_rule_feas = pe.Constraint(model_Bender_RMP_v3.j, rule = power_support_rule_feas)             
    # 
    # # The physical explanation of B[i,j,k] is: B[i,j,k]=0 indicates that the power in battery k of train j is used up (SOC=0) at station i. B[i,j,k]=1 indicates SOC of battery k is > 0.
    # # Feasibility cuts for B variables:
    # # When B[i,j,k]=0, then the power in batteries consists 1,...,k-1 is used up, so B[i,j,1]=...=B[i,j,k-1]=0
    # # print ('opt3...')
    # def battery_sequential_rule_linear_a_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         return sum(model_Bender_RMP_v3.B[i,j,kp] \
    #                     for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) < k_index) \
    #                 <= (k_index+1-1) * model_Bender_RMP_v3.B[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.battery_sequential_rule_linear_a_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = battery_sequential_rule_linear_a_feas)        

    # # When B[i,j,k]=1, then the power in batteries consists k+1,...,kj is full (SOC=100%) (where kj is the total number of consists in train j), so B[i,j,1]=...=B[i,j,k-1]=1.
    # # print ('opt4...')
    # def battery_sequential_rule_linear_b_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         return sum(model_Bender_RMP_v3.B[i,j,kp] \
    #                     for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) > k_index) \
    #                 >= (len(Trains[j]['containers']) - k_index-1) * model_Bender_RMP_v3.B[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.battery_sequential_rule_linear_b_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = battery_sequential_rule_linear_b_feas)        
    
    # # If the power in consist k is used not used up, then power in consist k+1 is not used up either
    # # print ('opt5...')
    # def battery_sequential_rule_linear_c_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         if k_index <= len(Trains[j]['containers'])-2:
    #             k_next = Trains[j]['containers'][k_index+1]
    #             return model_Bender_RMP_v3.B[i,j,k_next] >= model_Bender_RMP_v3.B[i,j,k] 
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.battery_sequential_rule_linear_c_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = battery_sequential_rule_linear_c_feas)        

    # # Feasibility cuts for the connection between B and X variables:
    # # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then a charging/swapping station must be deployed in location i.
    # # print ('opt6...')
    # def model_B_X_a_feas(model_Bender_RMP_v3, i, j):
    #     i_next = Stations[i]['station_after']
    #     if i != 'origin' and i != 'destination' and i_next != 'None':
    #         return M * model_Bender_RMP_v3.X[i] \
    #                 >= Power_train[j][i][i_next] - sum(model_Bender_RMP_v3.B[i,j,k] for k in Trains[j]['containers'])
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_X_a_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = model_B_X_a_feas)
    
    # # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then a charging/swapping station must be deployed in location i.
    # def model_B_X_b_feas(model_Bender_RMP_v3, i, j):
    #     i_next = Stations[i]['station_after']
    #     if i != 'origin' and i != 'destination' and i_next != 'None':
    #         return M * model_Bender_RMP_v3.X[i] \
    #                 >= Power_train[j][i][i_next] * (1 - sum(model_Bender_RMP_v3.B[i,j,k] \
    #                                                         for k in Trains[j]['containers']) )
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_X_b_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = model_B_X_b_feas)

    # # Feasibility cuts for the connection between B and Y variables:
    # # if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # def model_B_Y_feas(model_Bender_RMP_v3, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_RMP_v3.Y[j,k] == model_Bender_RMP_v3.B['origin',j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_Y_feas = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = model_B_Y_feas)

    # # Feasibility cuts for the connection between B and Z variables:
    # # If the maximum potential amount of power in train j (\sum_{k \in K_j} B_{ijk}) is smaller than the power required from location i to i', then at least one battery of train j must be charged or swapped in location i.
    # def model_B_Z_a_feas(model_Bender_RMP_v3, i, j):
    #     i_next = Stations[i]['station_after']
    #     if i != 'origin' and i != 'destination' and i_next != 'None':
    #         return M * sum((model_Bender_RMP_v3.Z_c[i,j,k] + model_Bender_RMP_v3.Z_s[i,j,k]) \
    #                         for k in Trains[j]['containers']) \
    #                 >= Power_train[j][i][i_next] - sum(model_Bender_RMP_v3.B[i,j,k] \
    #                                                   for k in Trains[j]['containers'])
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_Z_a_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = model_B_Z_a_feas)

    # # When arriving at location i, if all batteries in train j have a SOC of zero, and the power required from location i to i' (i' is immediately after i) is greater than zero ($e_{i,i',j}>0$), then at least one battery in train j must be charged/swapped in location i.
    # def model_B_Z_b_feas(model_Bender_RMP_v3, i, j):
    #     i_next = Stations[i]['station_after']
    #     if i != 'origin' and i != 'destination' and i_next != 'None':
    #         return M * sum((model_Bender_RMP_v3.Z_c[i,j,k] + model_Bender_RMP_v3.Z_s[i,j,k]) \
    #                         for k in Trains[j]['containers']) \
    #                 >= Power_train[j][i][i_next] * (1 - sum(model_Bender_RMP_v3.B[i,j,k] \
    #                                                         for k in Trains[j]['containers']) )
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_Z_b_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = model_B_Z_b_feas)
    
    # # Feasibility cuts for the connection between B and beta variables:
    # # If B[ijk]=0, then S_arrive[ijk]=0, and we have beta[ijk0]=1
    # # print ('opt7...')
    # def model_B_beta_a_rule_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         return sum(model_Bender_RMP_v3.beta[i,j,kp,min(model_Bender_RMP_v3.u)] \
    #                     for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) <= k_index) \
    #                 + (k_index+1) * model_Bender_RMP_v3.B[i,j,k] \
    #                 >= k_index+1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_beta_a_rule_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = model_B_beta_a_rule_feas)
    
    # # If B[ijk]=1, then S_arrive[i,j,k+1]=1, and we have beta[i,j,k+1,n]=0
    # # print ('opt8...')
    # def model_B_beta_b_rule_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         if k_index <= len(Trains[j]['containers'])-2:
    #             return sum(model_Bender_RMP_v3.beta[i,j,kp,max(model_Bender_RMP_v3.u)] \
    #                         for kp in Trains[j]['containers'] if Trains[j]['containers'].index(kp) >= k_index+1) \
    #                     - (len(Trains[j]['containers']) - (k_index+1)) * model_Bender_RMP_v3.B[i,j,k] \
    #                     >= 0
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_beta_b_rule_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = model_B_beta_b_rule_feas)        

    # def model_B_beta_c_rule_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_RMP_v3.beta[i,j,k,min(model_Bender_RMP_v3.u)] \
    #                 + model_Bender_RMP_v3.B[i,j,k] \
    #                 >= 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_beta_c_rule_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = model_B_beta_c_rule_feas)
    
    # # If B[ijk]=1, then S_arrive[i,j,k+1]=1, and we have beta[i,j,k+1,n]=0
    # # print ('opt8...')
    # def model_B_beta_d_rule_feas(model_Bender_RMP_v3, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         if k_index <= len(Trains[j]['containers'])-2:
    #             k_next = Trains[j]['containers'][k_index+1]
    #             return model_Bender_RMP_v3.beta[i,j,k_next,max(model_Bender_RMP_v3.u)] \
    #                     - model_Bender_RMP_v3.B[i,j,k] \
    #                     >= 0
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.model_B_beta_d_rule_feas = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = model_B_beta_d_rule_feas)        


    '''original constraints'''
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # print ('b...')
    def deploy_swap_charge_rule_orig(model_Bender_RMP_v3, i):
        return sum(sum((model_Bender_RMP_v3.Z_c[i,j,k] + model_Bender_RMP_v3.Z_s[i,j,k]) \
                       for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                   for j in model_Bender_RMP_v3.j) \
                <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * model_Bender_RMP_v3.X[i]
    model_Bender_RMP_v3.deploy_swap_charge_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, rule = deploy_swap_charge_rule_orig)        
    
    # # If station is deployed, then at least one battery must be swapped nor charged in station i
    # print ('opt2...')
    def deploy_swap_charge_must_rule_orig_feas(model_Bender_RMP_v3, i):
        return sum(sum((model_Bender_RMP_v3.Z_c[i,j,k] + model_Bender_RMP_v3.Z_s[i,j,k]) \
                        for k in model_Bender_RMP_v3.k if k in Trains[j]['containers']) \
                    for j in model_Bender_RMP_v3.j) \
                >= model_Bender_RMP_v3.X[i]
    model_Bender_RMP_v3.deploy_swap_charge_must_rule_orig_feas = pe.Constraint(model_Bender_RMP_v3.i, rule = deploy_swap_charge_must_rule_orig_feas)   

    # Train j cannot both swap and charge batteries in station i.
    # print ('c...')
    def noboth_swap_charge_rule_orig(model_Bender_RMP_v3, i, j, k1, k2):
        if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
            return model_Bender_RMP_v3.Z_s[i,j,k1] + model_Bender_RMP_v3.Z_c[i,j,k2] <= 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.noboth_swap_charge_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k1, model_Bender_RMP_v3.k2, rule = noboth_swap_charge_rule_orig)            
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # print ('e0...')
    def consecutive_battery_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            kid = int(((k.split('container '))[1].split(' in')[0]))
            if kid < len(Trains[j]['containers']):
                return sum(model_Bender_RMP_v3.Y[j,kp] \
                           for kp in model_Bender_RMP_v3.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                       <= sum(1 for kp in model_Bender_RMP_v3.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
                          * model_Bender_RMP_v3.Y[j,k]
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.consecutive_battery_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = consecutive_battery_rule_orig)  
        
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # print ('e...')
    def max_number_batteries_station_rule_orig(model_Bender_RMP_v3, i, j):
        return sum(model_Bender_RMP_v3.Z_s[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    model_Bender_RMP_v3.max_number_batteries_station_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = max_number_batteries_station_rule_orig)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # print ('f...')
    def max_number_chargers_station_rule_orig(model_Bender_RMP_v3, i, j):
        return sum(model_Bender_RMP_v3.Z_c[i,j,k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    model_Bender_RMP_v3.max_number_chargers_station_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, rule = max_number_chargers_station_rule_orig)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # *** print ('g...')
    def power_rule_orig_ub(model_Bender_RMP_v3, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_Bender_RMP_v3.S_depart[i1,j,k] for k in Trains[j]['containers']) \
                   - sum(model_Bender_RMP_v3.S_arrive[i2,j,k] for k in Trains[j]['containers']) \
                   <= Power_train[j][i1][i2] + epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.power_rule_orig_ub = pe.Constraint(model_Bender_RMP_v3.i1, model_Bender_RMP_v3.i2, model_Bender_RMP_v3.j, rule = power_rule_orig_ub)          
    
    def power_rule_orig_lb(model_Bender_RMP_v3, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return sum(model_Bender_RMP_v3.S_depart[i1,j,k] for k in Trains[j]['containers']) \
                   - sum(model_Bender_RMP_v3.S_arrive[i2,j,k] for k in Trains[j]['containers']) \
                   >= Power_train[j][i1][i2] - epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.power_rule_orig_lb = pe.Constraint(model_Bender_RMP_v3.i1, model_Bender_RMP_v3.i2, model_Bender_RMP_v3.j, rule = power_rule_orig_lb)       
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # *** print ('g1...')
    def battery_sequential_rule_linear_a_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                return M * model_Bender_RMP_v3.B[i,j,k] - model_Bender_RMP_v3.S_arrive[i,j,k] >= 0
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.battery_sequential_rule_linear_a_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = battery_sequential_rule_linear_a_orig)          

    # *** print ('g2...')
    def battery_sequential_rule_linear_b_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            k_index = Trains[j]['containers'].index(k)
            if k_index < len(Trains[j]['containers'])-1:
                k_next = Trains[j]['containers'][k_index+1]
                return model_Bender_RMP_v3.S_arrive[i,j,k_next] - M * model_Bender_RMP_v3.B[i,j,k] >= -M+1 
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.battery_sequential_rule_linear_b_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = battery_sequential_rule_linear_b_orig)          

    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # *** print ('h...')
    def soc_increase_rule_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.S_depart[i,j,k] - model_Bender_RMP_v3.S_arrive[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_increase_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_increase_rule_orig)          

    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # *** print ('i...')
    def swap_full_soc_rule_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.S_depart[i,j,k] - model_Bender_RMP_v3.Z_s[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.swap_full_soc_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = swap_full_soc_rule_orig)     

    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # *** print ('i1...')
    def soc_depart_arrive_between_stations_rule_orig(model_Bender_RMP_v3, i1, i2, j, k):
        if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
            return - model_Bender_RMP_v3.S_arrive[i2,j,k] + model_Bender_RMP_v3.S_depart[i1,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_depart_arrive_between_stations_rule_orig = pe.Constraint(model_Bender_RMP_v3.i1, model_Bender_RMP_v3.i2, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_depart_arrive_between_stations_rule_orig)     

    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # *** print ('k0...')
    def soc_time_charge_rule_PLA_Zs_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.S_depart[i,j,k] \
                   - model_Bender_RMP_v3.F[i,j,k] \
                   + M * model_Bender_RMP_v3.Z_s[i,j,k] \
                   >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_Zs_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_Zs_orig)        
    
    # *** print ('k1...')
    def soc_time_charge_rule_PLA_Zc_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.S_depart[i,j,k] \
                   + model_Bender_RMP_v3.F[i,j,k] \
                   + epsilon * model_Bender_RMP_v3.Z_c[i,j,k] \
                   >= 1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_Zc_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_Zc_orig)        
    
    # print ('k2...')
    def soc_time_charge_rule_PLA_beta1_orig_ub(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.beta[i,j,k,u] for u in model_Bender_RMP_v3.u if u != max(model_Bender_RMP_v3.u)) <= 1 + epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_beta1_orig_ub = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_beta1_orig_ub)        

    def soc_time_charge_rule_PLA_beta1_orig_lb(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.beta[i,j,k,u] for u in model_Bender_RMP_v3.u if u != max(model_Bender_RMP_v3.u)) >= 1 - epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_beta1_orig_lb = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_beta1_orig_lb)        

    # *** print ('k3...')
    def soc_time_charge_rule_PLA_gamma1_orig_ub(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.gamma[i,j,k,u] for u in model_Bender_RMP_v3.u) <= 1 + epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_gamma1_orig_ub = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_gamma1_orig_ub)        

    def soc_time_charge_rule_PLA_gamma1_orig_lb(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.gamma[i,j,k,u] for u in model_Bender_RMP_v3.u) >= 1 - epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_gamma1_orig_lb = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_gamma1_orig_lb)        

    # print ('k4...')
    def soc_time_charge_rule_PLA_tau1_orig_ub(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.tau[i,j,k,v] for v in model_Bender_RMP_v3.v if v != max(model_Bender_RMP_v3.v)) <= 1 + epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_tau1_orig_ub = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_tau1_orig_ub)        
    
    def soc_time_charge_rule_PLA_tau1_orig_lb(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum(model_Bender_RMP_v3.tau[i,j,k,v] for v in model_Bender_RMP_v3.v if v != max(model_Bender_RMP_v3.v)) >= 1 - epsilon
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_tau1_orig_lb = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_tau1_orig_lb)        

    # *** print ('k5...')
    def soc_time_charge_rule_PLA_gamma_Sarrive_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return sum((Segments_SOC[u]['SOC'] * model_Bender_RMP_v3.gamma[i,j,k,u]) \
                       for u in model_Bender_RMP_v3.u) \
                   - model_Bender_RMP_v3.S_arrive[i,j,k] \
                   == 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_gamma_Sarrive_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive_orig)        

    # @@@ print ('k6...')
    def soc_time_charge_rule_PLA_tau_eta_Tc(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.T_c[i,j,k] \
                   - sum((Segments_hour_charge[v]['hour'] * model_Bender_RMP_v3.tau[i,j,k,v] \
                          + (Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * model_Bender_RMP_v3.eta[i,j,k,v]) \
                         for v in model_Bender_RMP_v3.v if v != max(model_Bender_RMP_v3.v)) \
                   == 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # *** print ('k7...')
    def soc_time_charge_rule_PLA_gamma_beta_orig(model_Bender_RMP_v3, i, j, k, u):
        if k in Trains[j]['containers'] and u != min(model_Bender_RMP_v3.u) and u != max(model_Bender_RMP_v3.u):
            return - model_Bender_RMP_v3.gamma[i,j,k,u] + model_Bender_RMP_v3.beta[i,j,k,u-1] + model_Bender_RMP_v3.beta[i,j,k,u] >= 0
        elif k in Trains[j]['containers'] and u == min(model_Bender_RMP_v3.u):
            return - model_Bender_RMP_v3.gamma[i,j,k,u] + model_Bender_RMP_v3.beta[i,j,k,u] >= 0
        elif k in Trains[j]['containers'] and u == max(model_Bender_RMP_v3.u):
            return - model_Bender_RMP_v3.gamma[i,j,k,u] + model_Bender_RMP_v3.beta[i,j,k,u-1] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_gamma_beta_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, rule = soc_time_charge_rule_PLA_gamma_beta_orig)        

    # *** print ('k8...')
    def soc_time_charge_rule_PLA_eta_tau_orig(model_Bender_RMP_v3, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_RMP_v3.v) and v != max(model_Bender_RMP_v3.v):
            return - model_Bender_RMP_v3.eta[i,j,k,v] + model_Bender_RMP_v3.tau[i,j,k,v-1] + model_Bender_RMP_v3.tau[i,j,k,v] >= 0
        elif k in Trains[j]['containers'] and v == min(model_Bender_RMP_v3.v):
            return - model_Bender_RMP_v3.eta[i,j,k,v] + model_Bender_RMP_v3.tau[i,j,k,v] >= 0
        elif k in Trains[j]['containers'] and v == max(model_Bender_RMP_v3.v):
            return - model_Bender_RMP_v3.eta[i,j,k,v] + model_Bender_RMP_v3.tau[i,j,k,v-1] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_eta_tau_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.v, rule = soc_time_charge_rule_PLA_eta_tau_orig)        

    # *** print ('k4...')
    def soc_time_charge_rule_PLA_eta_tau1_orig(model_Bender_RMP_v3, i, j, k, v):
        if k in Trains[j]['containers'] and v != min(model_Bender_RMP_v3.v) and v != max(model_Bender_RMP_v3.v):
            return - model_Bender_RMP_v3.eta[i,j,k,v] + model_Bender_RMP_v3.tau[i,j,k,v] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_eta_tau1_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.v, rule = soc_time_charge_rule_PLA_eta_tau1_orig)        

    # *** print ('k9...')
    def soc_time_charge_rule_PLA_F_leq_orig(model_Bender_RMP_v3, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_RMP_v3.u) and v != max(model_Bender_RMP_v3.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return sum((model_Bender_RMP_v3.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                       for w in model_Bender_RMP_v3.u) \
                   + w * model_Bender_RMP_v3.eta[i,j,k,v] \
                   - model_Bender_RMP_v3.F[i,j,k] \
                   - M * model_Bender_RMP_v3.tau[i,j,k,v] \
                   - M * model_Bender_RMP_v3.beta[i,j,k,u] \
                   >= -2*M
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_F_leq_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, model_Bender_RMP_v3.v, rule = soc_time_charge_rule_PLA_F_leq_orig)        

    # *** print ('k10...')
    def soc_time_charge_rule_PLA_F_geq_orig(model_Bender_RMP_v3, i, j, k, u, v):
        if k in Trains[j]['containers'] and u != max(model_Bender_RMP_v3.u) and v != max(model_Bender_RMP_v3.v):
            s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
            s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
            g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
            g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
            g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
            g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
            w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
            return model_Bender_RMP_v3.F[i,j,k] \
                   - sum((model_Bender_RMP_v3.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
                         for w in model_Bender_RMP_v3.u) \
                   - w * model_Bender_RMP_v3.eta[i,j,k,v] \
                   - M * model_Bender_RMP_v3.tau[i,j,k,v] \
                   - M * model_Bender_RMP_v3.beta[i,j,k,u] \
                   >= -2*M
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_PLA_F_geq_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, model_Bender_RMP_v3.v, rule = soc_time_charge_rule_PLA_F_geq_orig)        

    # @@@ print ('n...')
    def soc_time_charge_rule_d(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.T_c[i,j,k] + M * model_Bender_RMP_v3.Z_c[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_d = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_d)
    
    # @@@ print ('n1...')
    def soc_time_charge_rule_e(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return M * model_Bender_RMP_v3.T_c[i,j,k] - model_Bender_RMP_v3.Z_c[i,j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_time_charge_rule_e = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_time_charge_rule_e)

    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # *** print ('o...')
    def no_swap_charge_soc_rule_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.S_depart[i,j,k] + model_Bender_RMP_v3.S_arrive[i,j,k] \
                   + model_Bender_RMP_v3.Z_c[i,j,k] + model_Bender_RMP_v3.Z_s[i,j,k] \
                   >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.no_swap_charge_soc_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = no_swap_charge_soc_rule_orig)

    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # *** print ('o2...')
    def soc_depart_origin_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.S_depart['origin',j,k] - model_Bender_RMP_v3.Y[j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_depart_origin_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_depart_origin_rule_orig)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # *** print ('o3...')
    def soc_arrive_origin_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_RMP_v3.S_arrive['origin',j,k] - model_Bender_RMP_v3.Y[j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.soc_arrive_origin_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = soc_arrive_origin_rule_orig)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    def origin_no_chargeswap_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            return 2 * model_Bender_RMP_v3.Y[j,k] - model_Bender_RMP_v3.Z_c['origin',j,k] - model_Bender_RMP_v3.Z_s['origin',j,k] >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.origin_no_chargeswap_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = origin_no_chargeswap_rule_orig)

    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # print ('o6...')
    def nobattery_nochargeswap_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            return 2*len(Stations) * model_Bender_RMP_v3.Y[j,k] \
                   - sum(model_Bender_RMP_v3.Z_c[i,j,k] for i in model_Bender_RMP_v3.i) \
                   - sum(model_Bender_RMP_v3.Z_s[i,j,k] for i in model_Bender_RMP_v3.i) \
                   >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.nobattery_nochargeswap_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = nobattery_nochargeswap_rule_orig)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # *** print ('o7...')
    def nobattery_zerosoc_rule_orig(model_Bender_RMP_v3, j, k):
        if k in Trains[j]['containers']:
            return - sum(model_Bender_RMP_v3.S_arrive[i,j,k] for i in model_Bender_RMP_v3.i) \
                   - sum(model_Bender_RMP_v3.S_depart[i,j,k] for i in model_Bender_RMP_v3.i) \
                   + 2*len(Stations) * model_Bender_RMP_v3.Y[j,k] \
                   >= 0
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.nobattery_zerosoc_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = nobattery_zerosoc_rule_orig)

    # The SOC of each battery must not exceed 100%.
    # *** print ('p...')
    def ub_soc_arrive_rule_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.S_arrive[i,j,k] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.ub_soc_arrive_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = ub_soc_arrive_rule_orig)
    
    # *** print ('q...')
    def ub_soc_depart_rule_orig(model_Bender_RMP_v3, i, j, k):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.S_depart[i,j,k] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.ub_soc_depart_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = ub_soc_depart_rule_orig)

    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_RMP_v3.
    # print ('r...')
    # def preprocessing_Y0_rule_orig(model_Bender_RMP_v3, j, k):
    #     if k not in Trains[j]['containers']:
    #         return model_Bender_RMP_v3.Y[j,k] == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_RMP_v3.preprocessing_Y0_rule_orig = pe.Constraint(model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, rule = preprocessing_Y0_rule_orig)
    
    # the value of gamma an eta variables are no greater than 1
    # *** print ('t...')
    def ub_gamma_rule_orig(model_Bender_RMP_v3, i, j, k, u):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.gamma[i,j,k,u] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.ub_gamma_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.u, rule = ub_gamma_rule_orig)
    
    def ub_eta_rule_orig(model_Bender_RMP_v3, i, j, k, v):
        if k in Trains[j]['containers']:
            return - model_Bender_RMP_v3.eta[i,j,k,v] >= -1
        else:
            return pe.Constraint.Skip
    model_Bender_RMP_v3.ub_eta_rule_orig = pe.Constraint(model_Bender_RMP_v3.i, model_Bender_RMP_v3.j, model_Bender_RMP_v3.k, model_Bender_RMP_v3.v, rule = ub_eta_rule_orig)
    
    
    # solve the model
    print('Solving...')
    # solve by gurobi
    opt = pyomo.opt.SolverFactory('gurobi')
    opt.options['MIPGap'] = gap_RMP  # 1% optimality gap
    opt.options['TimeLimit'] = time_RMP
    results = opt.solve(model_Bender_RMP_v3, tee=False, keepfiles=False)
    # results.write()
    
    # solve by cplex:
    # opt= pyomo.opt.SolverFactory("cplex")
    # # optimality_gap = 0.05
    # # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 900
    # results=opt.solve(model_Bender_RMP_v3, tee=False, keepfiles=False)
    # # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_Bender_RMP_v3, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_Bender_RMP_v3)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_Bender_RMP_v3, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    
    time_end = time.time()
    time_model = time_end - time_start
    # upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
    # gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
    
    '''Record variables'''  
    # print ('Recording variables...')
    W.update({itr_while: model_Bender_RMP_v3.W[0].value})
    
    X.update({itr_while:{}})
    Y.update({itr_while:{}})
    Z_c.update({itr_while:{}})
    Z_s.update({itr_while:{}})
    S_arrive.update({itr_while:{}})
    S_depart.update({itr_while:{}})
    T_c.update({itr_while:{}})
    for i in Stations:
        X[itr_while].update({i: model_Bender_RMP_v3.X[i].value})
        Z_c[itr_while].update({i: {}})
        Z_s[itr_while].update({i: {}})
        S_arrive[itr_while].update({i: {}})
        S_depart[itr_while].update({i: {}})
        T_c[itr_while].update({i: {}})
        for j in Trains:
            Z_c[itr_while][i].update({j: {}})
            Z_s[itr_while][i].update({j: {}})
            S_arrive[itr_while][i].update({j: {}})
            S_depart[itr_while][i].update({j: {}})
            T_c[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                Z_c[itr_while][i][j].update({k: model_Bender_RMP_v3.Z_c[i,j,k].value})
                Z_s[itr_while][i][j].update({k: model_Bender_RMP_v3.Z_s[i,j,k].value})
                S_arrive[itr_while][i][j].update({k: model_Bender_RMP_v3.S_arrive[i,j,k].value})
                S_depart[itr_while][i][j].update({k: model_Bender_RMP_v3.S_depart[i,j,k].value})
                T_c[itr_while][i][j].update({k: model_Bender_RMP_v3.T_c[i,j,k].value})
    for j in Trains:
        Y[itr_while].update({j: {}})
        for k in Trains[j]['containers']:
            Y[itr_while][j].update({k: model_Bender_RMP_v3.Y[j,k].value})
            
    beta.update({itr_while:{}})
    tau.update({itr_while:{}})
    gamma.update({itr_while:{}})
    eta.update({itr_while:{}})
    B.update({itr_while:{}})
    F.update({itr_while:{}})
    for i in Stations:
        beta[itr_while].update({i: {}})
        tau[itr_while].update({i: {}})
        gamma[itr_while].update({i: {}})
        eta[itr_while].update({i: {}})
        B[itr_while].update({i: {}})
        F[itr_while].update({i: {}})
        for j in Trains:
            beta[itr_while][i].update({j: {}})
            tau[itr_while][i].update({j: {}})
            gamma[itr_while][i].update({j: {}})
            eta[itr_while][i].update({j: {}})
            B[itr_while][i].update({j: {}})
            F[itr_while][i].update({j: {}})
            for k in Trains[j]['containers']:
                beta[itr_while][i][j].update({k: {}})
                tau[itr_while][i][j].update({k: {}})
                gamma[itr_while][i][j].update({k: {}})
                eta[itr_while][i][j].update({k: {}})
                if model_Bender_RMP_v3.B[i,j,k].value == None:
                    B[itr_while][i][j].update({k: 1})
                    F[itr_while][i][j].update({k: 1})
                else:
                    B[itr_while][i][j].update({k: model_Bender_RMP_v3.B[i,j,k].value})
                    F[itr_while][i][j].update({k: model_Bender_RMP_v3.F[i,j,k].value})
                for u in Segments_SOC:
                    if u != max(Segments_SOC):
                        beta[itr_while][i][j][k].update({u: model_Bender_RMP_v3.beta[i,j,k,u].value})
                    gamma[itr_while][i][j][k].update({u: model_Bender_RMP_v3.gamma[i,j,k,u].value})
                for v in Segments_hour_charge:
                    if v != max(Segments_hour_charge):
                        tau[itr_while][i][j][k].update({v: model_Bender_RMP_v3.tau[i,j,k,v].value})
                        eta[itr_while][i][j][k].update({v: model_Bender_RMP_v3.eta[i,j,k,v].value})
    
    results_RMP.update({itr_while: results})

    if results.solver.termination_condition != TerminationCondition.infeasible and results.solver.termination_condition != TerminationCondition.infeasibleOrUnbounded:  
        # calculate the objective function value
        cost_fix = sum(Stations[i]['cost_fix'] * model_Bender_RMP_v3.X[i].value for i in model_Bender_RMP_v3.i)
        constant = - penalty_delay * sum(sum(Trains[j]['stations'][i]['time_wait'] for j in model_Bender_RMP_v3.j) for i in model_Bender_RMP_v3.i)
        obj = penalty_fix_cost * cost_fix + model_Bender_RMP_v3.W[0].value + constant
        obj_RMP.update({itr_while: obj})
        if itr_while > 0:
            if obj > LB[itr_while-1]:
                LB[itr_while] = obj
            else:
                LB[itr_while] = LB[itr_while-1]
    else:
        obj = 'None'
        obj_RMP.update({itr_while: obj})
        if itr_while > 0:
            LB[itr_while] = LB[itr_while-1]
    
    return obj_RMP, LB, W, X, Y, Z_c, Z_s, beta, tau, B, S_arrive, S_depart, T_c, gamma, eta, F, results_RMP, time_model
















'''This model is based on Model_Bender_SP_v1. 
The difference between this model (v2) and Model_Bender_SP_v1 is: this model has removed variables S, gamma, eta and F. Now it includes T and D variables only.
'''
def Model_Bender_SP_v2(Stations, Trains, Containers, Power_train, TravelTime_train, hour_battery_swap, rate_charge_empty, penalty_delay, penalty_fix_cost, \
                       M, epsilon, buffer_soc_estimate, Segments_SOC, Segments_hour_charge, \
                       X, Y, Z_c, Z_s, beta, tau, B, D, S_arrive, S_depart, T_arrive, T_depart, T_c, F, gamma, eta, \
                       pi, Q, ER, EP, itr_while, ratio_ER, obj_SP, results_SP, UB):

    time_start = time.time()
    model_Bender_SP_v2 = pe.ConcreteModel()
    # print ('Defining indices and sets...')
    # Sets and indices
    model_Bender_SP_v2.i = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v2.i1 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v2.i2 = pe.Set(initialize = sorted(set(Stations.keys())))
    model_Bender_SP_v2.j = pe.Set(initialize = sorted(set(Trains.keys())))
    model_Bender_SP_v2.k = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v2.k1 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v2.k2 = pe.Set(initialize = sorted(set(Containers.keys())))
    model_Bender_SP_v2.u = pe.Set(initialize = sorted(set(Segments_SOC.keys())))
    model_Bender_SP_v2.v = pe.Set(initialize = sorted(set(Segments_hour_charge.keys())))
    
    # Variables
    # print ('Defining variables...')
    model_Bender_SP_v2.D = pe.Var(model_Bender_SP_v2.i, model_Bender_SP_v2.j, within = pe.NonNegativeReals)
    model_Bender_SP_v2.T_arrive = pe.Var(model_Bender_SP_v2.i, model_Bender_SP_v2.j, within = pe.NonNegativeReals)
    model_Bender_SP_v2.T_depart = pe.Var(model_Bender_SP_v2.i, model_Bender_SP_v2.j, within = pe.NonNegativeReals)
    # preprocessing  # since it is AbstractModel, we move preprocessing constraints to formal constraints
    # print ('Preprocessing') 
    # For each train j, the depart and arrival time at the origin are both zero.
    for j in model_Bender_SP_v2.j:
        model_Bender_SP_v2.T_arrive['origin',j].fix(0)
        model_Bender_SP_v2.T_depart['origin',j].fix(0)
    
    # objective function
    # print ('Reading objective...')
    def obj_rule(model_Bender_SP_v2):
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SP_v2.i)
        cost_delay = sum(sum((model_Bender_SP_v2.D[i,j] - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SP_v2.j) for i in model_Bender_SP_v2.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        return obj  
    model_Bender_SP_v2.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
    
    
    # constraints
    # define delay time for train j in station i
    # print ('a...')
    def delay_define_rule(model_Bender_SP_v2, i, j):
        return model_Bender_SP_v2.D[i,j] - model_Bender_SP_v2.T_depart[i,j] + model_Bender_SP_v2.T_arrive[i,j] >= 0
    model_Bender_SP_v2.delay_define_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, rule = delay_define_rule)
    
    # If station is not deployed, then batteries can be neither swapped nor charged in station i
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('b...')
    # def deploy_swap_charge_rule(model_Bender_SP_v2, i):
    #     return sum(sum((Z_c[itr_while][i][j][k] + Z_s[itr_while][i][j][k]) \
    #                    for k in model_Bender_SP_v2.k if k in Trains[j]['containers']) \
    #                for j in model_Bender_SP_v2.j) \
    #             <= 2 * sum(sum(1 for k in Containers if k in Trains[j]['containers']) for j in Trains) * X[i][itr_while]
    # model_Bender_SP_v2.deploy_swap_charge_rule = pe.Constraint(model_Bender_SP_v2.i, rule = deploy_swap_charge_rule)        
    
    # Train j cannot both swap and charge batteries in station i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('c...')
    # def noboth_swap_charge_rule(model_Bender_SP_v2, i, j, k1, k2):
    #     if k1 in Trains[j]['containers'] and k2 in Trains[j]['containers']:
    #         return Z_s[itr_while][i][j][k1] + Z_c[itr_while][i][j][k2] <= 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.noboth_swap_charge_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k1, model_Bender_SP_v2.k2, rule = noboth_swap_charge_rule)        
    
    # Train j cannot depart station i until passenger trains have passed.
    # print ('d...')
    def wait_passenger_rule(model_Bender_SP_v2, i, j):
        return model_Bender_SP_v2.T_depart[i,j] - model_Bender_SP_v2.T_arrive[i,j] >= Trains[j]['stations'][i]['time_wait']
    model_Bender_SP_v2.wait_passenger_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, rule = wait_passenger_rule)        
    
    # Upper bound on the number of batteries that train j can carry. 
    # This set of constraints are addressed in preprocessing "If container k does not belong to train j, then Y[j,k] = 0."
    
    # In train j, if container k does not carry a battery, then all the containers behind k do not carry batteries, either. In other words, for each train, batteries can only be stored in the first few consecutive containers. Assume the containers closer to the locomotive have smaller indices.
    # this constraint should be removed from SP, because it does not contain any variables (Y are constants).
    # print ('e0...')
    # def consecutive_battery_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         kid = int(((k.split('container '))[1].split(' in')[0]))
    #         if kid < len(Trains[j]['containers']):
    #             return sum(Y[itr_while][j][kp] \
    #                        for kp in model_Bender_SP_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                    <= sum(1 for kp in model_Bender_SP_v2.k if kp in Trains[j]['containers'] and int(((kp.split('container '))[1].split(' in')[0])) >= kid+1) \
    #                       * Y[itr_while][j][k]
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.consecutive_battery_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = consecutive_battery_rule)  
    
    # If train j chooses to swap batteries in station i, the number of swapped batteries cannot exceed the number of available batterries in i.
    # This constraint should be removed from SP, because it does not have variables.
    # print ('e...')
    # def max_number_batteries_station_rule(model_Bender_SP_v2, i, j):
    #     return sum(Z_s[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_batteries']
    # model_Bender_SP_v2.max_number_batteries_station_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, rule = max_number_batteries_station_rule)  
    
    # if train j chooses to charge batteries in station i, the number of charged batteries cannot exceed teh number of available chargers in i.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('f...')
    # def max_number_chargers_station_rule(model_Bender_SP_v2, i, j):
    #     return sum(Z_c[itr_while][i][j][k] for k in Trains[j]['containers']) <= Stations[i]['max_number_chargers']
    # model_Bender_SP_v2.max_number_chargers_station_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, rule = max_number_chargers_station_rule)  
    
    # If station i2 is immediately after station i1 along the route, then for train j, the SOC at its arrival at i2 must equal the SOC at its departure from i1 minus the amount of power required for train j to travel from stations i1 to i2.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('g...')
    # def power_rule(model_Bender_SP_v2, i1, i2, j):
    #     if i2 == Stations[i1]['station_after']:
    #         return sum(model_Bender_SP_v2.S_depart[i1,j,k] for k in Trains[j]['containers']) \
    #                - sum(model_Bender_SP_v2.S_arrive[i2,j,k] for k in Trains[j]['containers']) \
    #                == Power_train[j][i1][i2]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.power_rule = pe.Constraint(model_Bender_SP_v2.i1, model_Bender_SP_v2.i2, model_Bender_SP_v2.j, rule = power_rule)          
    
    # For each train j, electricity in each battery is consumed sequentially, starting from the battery closest to the locomotive. If container k1 is closer to the locomotive than k2, then the battery in k2 will not be used before the battery in k1 is run out.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('g1...')
    # def battery_sequential_rule_linear_a(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         if k_index < len(Trains[j]['containers'])-1:
    #             return - model_Bender_SP_v2.S_arrive[i,j,k] >=  -M * B[itr_while][i][j][k]
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.battery_sequential_rule_linear_a = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = battery_sequential_rule_linear_a)          

    # print ('g2...')
    # def battery_sequential_rule_linear_b(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         k_index = Trains[j]['containers'].index(k)
    #         if k_index < len(Trains[j]['containers'])-1:
    #             k_next = Trains[j]['containers'][k_index+1]
    #             return model_Bender_SP_v2.S_arrive[i,j,k_next] >=  -M+1 + M * B[itr_while][i][j][k]
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.battery_sequential_rule_linear_b = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = battery_sequential_rule_linear_b)          
    
    # When train j departs station i, the SOC of battery in container k of train j cannot be lower than the SOC at its arrival.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('h...')
    # def soc_increase_rule(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.S_depart[i,j,k] - model_Bender_SP_v2.S_arrive[i,j,k] >= 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_increase_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_increase_rule)          
    
    # If the battery in container k of train j is swapped in station i, then the SOC is 100% when train j leaves station i.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('i...')
    # def swap_full_soc_rule(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.S_depart[i,j,k] >= Z_s[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.swap_full_soc_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = swap_full_soc_rule)     
    
    # If station i2 is immediately after i1, then for each battery in container k of train j, the SOC at its arrival in station i2 must be no higher than that at its departure from station i1.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('i1...')
    # def soc_depart_arrive_between_stations_rule(model_Bender_SP_v2, i1, i2, j, k):
    #     if k in Trains[j]['containers'] and i2 == Stations[i1]['station_after']:
    #         return - model_Bender_SP_v2.S_arrive[i2,j,k] + model_Bender_SP_v2.S_depart[i1,j,k] >= 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_depart_arrive_between_stations_rule = pe.Constraint(model_Bender_SP_v2.i1, model_Bender_SP_v2.i2, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_depart_arrive_between_stations_rule)     

    # If the battery in container k of train j is swapped in station i, then train j must stay in i for at least      
    # print ('j...')
    def time_battery_swap_rule(model_Bender_SP_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v2.T_depart[i,j] - model_Bender_SP_v2.T_arrive[i,j] >= hour_battery_swap * Z_s[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v2.time_battery_swap_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = time_battery_swap_rule)     
    
    # If the battery in container k of train j is charged in station i, then when train j leaves station i, the SOC of container k is a function of the SOC when j arrives at i, and the charging time.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('k0...')
    # def soc_time_charge_rule_PLA_Zs(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.S_depart[i,j,k] \
    #                - model_Bender_SP_v2.F[i,j,k] \
    #                >= -1 - M * Z_s[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_Zs = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_Zs)        
    
    # print ('k1...')
    # def soc_time_charge_rule_PLA_Zc(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.S_depart[i,j,k] \
    #                + model_Bender_SP_v2.F[i,j,k] \
    #                >= 1 - epsilon * Z_c[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_Zc = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_Zc)        

    # This constraint should be removed from SP,  because it does not have variables.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('k2...')
    # def soc_time_charge_rule_PLA_beta1(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(beta[itr_while][i][j][k][u] for u in model_Bender_SP_v2.u if u != max(model_Bender_SP_v2.u)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_beta1 = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_beta1)        

    # print ('k3...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_gamma1(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(model_Bender_SP_v2.gamma[i,j,k,u] for u in model_Bender_SP_v2.u) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_gamma1 = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_gamma1)        

    # This constraint should be removed from SP,  because it does not have variables.
    # print ('k4...')
    # def soc_time_charge_rule_PLA_tau1(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(tau[itr_while][i][j][k][v] for v in model_Bender_SP_v2.v if v != max(model_Bender_SP_v2.v)) == 1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_tau1 = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_tau1)        

    # print ('k5...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_gamma_Sarrive(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum((Segments_SOC[u]['SOC'] * model_Bender_SP_v2.gamma[i,j,k,u]) \
    #                    for u in model_Bender_SP_v2.u) \
    #                - model_Bender_SP_v2.S_arrive[i,j,k] \
    #                == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_gamma_Sarrive = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_gamma_Sarrive)        

    # print ('k6...')
    # @@@ This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_tau_eta_Tc(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.T_c[i,j,k] \
    #                == sum(((Segments_hour_charge[v+1]['hour']-Segments_hour_charge[v]['hour']) * eta[itr_while][i][j][k][v]) \
    #                       for v in model_Bender_SP_v2.v if v != max(model_Bender_SP_v2.v)) \
    #                   + sum((Segments_hour_charge[v]['hour'] * tau[itr_while][i][j][k][v]) \
    #                         for v in model_Bender_SP_v2.v if v != max(model_Bender_SP_v2.v))
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_tau_eta_Tc = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_PLA_tau_eta_Tc)        

    # print ('k7...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_gamma_beta(model_Bender_SP_v2, i, j, k, u):
    #     if k in Trains[j]['containers'] and u != min(model_Bender_SP_v2.u) and u != max(model_Bender_SP_v2.u):
    #         return - model_Bender_SP_v2.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u-1] - beta[itr_while][i][j][k][u]
    #     elif k in Trains[j]['containers'] and u == min(model_Bender_SP_v2.u):
    #         return - model_Bender_SP_v2.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u]
    #     elif k in Trains[j]['containers'] and u == max(model_Bender_SP_v2.u):
    #         return - model_Bender_SP_v2.gamma[i,j,k,u] >= -beta[itr_while][i][j][k][u-1]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_gamma_beta = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.u, rule = soc_time_charge_rule_PLA_gamma_beta)        

    # print ('k8...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_eta_tau(model_Bender_SP_v2, i, j, k, v):
    #     if k in Trains[j]['containers'] and v != min(model_Bender_SP_v2.v) and v != max(model_Bender_SP_v2.v):
    #         return - model_Bender_SP_v2.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v-1] - tau[itr_while][i][j][k][v]
    #     elif k in Trains[j]['containers'] and v == min(model_Bender_SP_v2.v):
    #         return - model_Bender_SP_v2.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v]
    #     elif k in Trains[j]['containers'] and v == max(model_Bender_SP_v2.v):
    #         return - model_Bender_SP_v2.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v-1]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_eta_tau = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.v, rule = soc_time_charge_rule_PLA_eta_tau)        

    # print ('k4...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_eta_tau1(model_Bender_SP_v2, i, j, k, v):
    #     if k in Trains[j]['containers'] and v != min(model_Bender_SP_v2.v) and v != max(model_Bender_SP_v2.v):
    #         return - model_Bender_SP_v2.eta[i,j,k,v] >= -tau[itr_while][i][j][k][v]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_eta_tau1 = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.v, rule = soc_time_charge_rule_PLA_eta_tau1)        

    # print ('k9...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_F_leq(model_Bender_SP_v2, i, j, k, u, v):
    #     if k in Trains[j]['containers'] and u != max(model_Bender_SP_v2.u) and v != max(model_Bender_SP_v2.v):
    #         s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
    #         s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
    #         g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
    #         g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
    #         g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
    #         g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
    #         w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
    #         return sum((model_Bender_SP_v2.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
    #                    for w in model_Bender_SP_v2.u) \
    #                + w * model_Bender_SP_v2.eta[i,j,k,v] \
    #                - model_Bender_SP_v2.F[i,j,k] \
    #                >= M * (-2 + tau[itr_while][i][j][k][v] + beta[itr_while][i][j][k][u])
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_F_leq = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.u, model_Bender_SP_v2.v, rule = soc_time_charge_rule_PLA_F_leq)        

    # print ('k10...')
    # *** This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_PLA_F_geq(model_Bender_SP_v2, i, j, k, u, v):
    #     if k in Trains[j]['containers'] and u != max(model_Bender_SP_v2.u) and v != max(model_Bender_SP_v2.v):
    #         s0, t0 = Segments_SOC[u]['SOC'], Segments_hour_charge[v]['hour']
    #         s1, t1 = Segments_SOC[u+1]['SOC'], Segments_hour_charge[v+1]['hour']
    #         g_0_0 = (1-s0) * ((1-rate_charge_empty)**t0)
    #         g_0_1 = (1-s0) * ((1-rate_charge_empty)**t1)
    #         g_1_0 = (1-s1) * ((1-rate_charge_empty)**t0)
    #         g_1_1 = (1-s1) * ((1-rate_charge_empty)**t1)
    #         w = min(g_0_1 - g_0_0, g_1_1 - g_1_0)
    #         return model_Bender_SP_v2.F[i,j,k] \
    #                - sum((model_Bender_SP_v2.gamma[i,j,k,w] * ((1-Segments_SOC[w]['SOC'])*((1-rate_charge_empty)**t0) )) \
    #                      for w in model_Bender_SP_v2.u) \
    #                - w * model_Bender_SP_v2.eta[i,j,k,v] \
    #                >= M * (-2 + tau[itr_while][i][j][k][v] + beta[itr_while][i][j][k][u])
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_PLA_F_geq = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.u, model_Bender_SP_v2.v, rule = soc_time_charge_rule_PLA_F_geq)        
    
    # This constraint shoule be removed because it doesn't suite changing charge rate.
    # print ('l...')
    # def soc_time_charge_rule_b(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return rate_charge * model_Bender_SP_v2.T_c[i,j,k] <= 1 - model_Bender_SP_v2.S_arrive[i,j,k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_b = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_b)
    
    # print ('m...')
    def soc_time_charge_rule_c(model_Bender_SP_v2, i, j, k):
        if k in Trains[j]['containers']:
            return model_Bender_SP_v2.T_depart[i,j] - model_Bender_SP_v2.T_arrive[i,j] >= T_c[itr_while][i][j][k]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v2.soc_time_charge_rule_c = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_c)
    
    # print ('n...')
    # @@@ This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_d(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.T_c[i,j,k] >= -M * Z_c[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_d = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_d)
    
    # print ('n1...')
    # @@@ This constraint should be removed from SP, because it does not have variables.
    # def soc_time_charge_rule_e(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return M * model_Bender_SP_v2.T_c[i,j,k] >= Z_c[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_time_charge_rule_e = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_time_charge_rule_e)
    
    # If the battery in container k of train j is neither charged nor swapped in station i, then the SOC at its departure an arrival should be same.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('o...')
    # def no_swap_charge_soc_rule(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.S_depart[i,j,k] + model_Bender_SP_v2.S_arrive[i,j,k] >= -Z_c[itr_while][i][j][k] - Z_s[itr_while][i][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.no_swap_charge_soc_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = no_swap_charge_soc_rule)
    
    # If container k of train j carries a battery, then it is assumed to be fully charged when the train departs the origin.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('o2...')
    # def soc_depart_origin_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.S_depart['origin',j,k] >= Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_depart_origin_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_depart_origin_rule)
    
    # To reduce CPLEX processing time, if container k of train j carries a battery, then its SOC is assumed to be 100% when the train arrives the origin.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('o3...')
    # def soc_arrive_origin_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         return model_Bender_SP_v2.S_arrive['origin',j,k] >= Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.soc_arrive_origin_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = soc_arrive_origin_rule)
    
    # If container k of train j carries a battery, then the battery is neither charged nor swapped at the origin.
    # print ('o4...')
    # This constraint should be removed from SP,  because it does not have variables.
    # def origin_no_chargeswap_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         return Z_c[itr_while]['origin'][j][k] + Z_s[itr_while]['origin'][j][k] <= 2 * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.origin_no_chargeswap_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = origin_no_chargeswap_rule)
    
    # For train j, if station i2 is immediately after station i1 on the route, then the arrival time at i2 should equal the departure time at i1 plus the travel time from stations i1 to i2.
    # print ('o5...')
    def T_traveltime_rule(model_Bender_SP_v2, i1, i2, j):
        if i2 == Stations[i1]['station_after']:
            return model_Bender_SP_v2.T_arrive[i2,j] - model_Bender_SP_v2.T_depart[i1,j] == TravelTime_train[j][i1][i2]
        else:
            return pe.Constraint.Skip
    model_Bender_SP_v2.T_traveltime_rule = pe.Constraint(model_Bender_SP_v2.i1, model_Bender_SP_v2.i2, model_Bender_SP_v2.j, rule = T_traveltime_rule)
    
    # If container k in train j does not carry a battery, then we can neither charge nor swap the battery of container k in train j at any stations.
    # This constraint should be removed from SP,  because it does not have variables.
    # print ('o6...')
    # def nobattery_nochargeswap_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         return sum(Z_c[itr_while][i][j][k] for i in model_Bender_SP_v2.i) + sum(Z_s[itr_while][i][j][k] for i in model_Bender_SP_v2.i) <= 2*len(Stations) * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.nobattery_nochargeswap_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = nobattery_nochargeswap_rule)
    
    # If container k of train j does not carry a battery, then the SOC of the "dummy" battery in container k of train j must equal zero at all stations.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('o7...')
    # def nobattery_zerosoc_rule(model_Bender_SP_v2, j, k):
    #     if k in Trains[j]['containers']:
    #         return - sum(model_Bender_SP_v2.S_arrive[i,j,k] for i in model_Bender_SP_v2.i) \
    #                - sum(model_Bender_SP_v2.S_depart[i,j,k] for i in model_Bender_SP_v2.i) \
    #                >= -2*len(Stations) * Y[itr_while][j][k]
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.nobattery_zerosoc_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = nobattery_zerosoc_rule)
    
    # The SOC of each battery must not exceed 100%.
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('p...')
    # def ub_soc_arrive_rule(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.S_arrive[i,j,k] >= -1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.ub_soc_arrive_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = ub_soc_arrive_rule)
    
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('q...')
    # def ub_soc_depart_rule(model_Bender_SP_v2, i, j, k):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.S_depart[i,j,k] >= -1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.ub_soc_depart_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = ub_soc_depart_rule)
    
    # print ('r...')
    # If container k does not belong to train j, then Y[j,k] = 0. This rule replaces constraints "Upper bound on the number of batteries that train j can carry" in the latex model_Bender_SP_v2.
    # This constraint should be removed from SP,  because it does not have variables.
    # def preprocessing_Y0_rule(model_Bender_SP_v2, j, k):
    #     if k not in Trains[j]['containers']:
    #         return Y[itr_while][j][k] == 0
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.preprocessing_Y0_rule = pe.Constraint(model_Bender_SP_v2.j, model_Bender_SP_v2.k, rule = preprocessing_Y0_rule)
       
    # print ('s...')
    # For each train j, the depart and arrival time at the origin are both zero.
    def preprocessing_T0_rule(model_Bender_SP_v2, j):
        return model_Bender_SP_v2.T_arrive['origin',j] + model_Bender_SP_v2.T_depart['origin',j] == 0
    model_Bender_SP_v2.preprocessing_T0_rule = pe.Constraint(model_Bender_SP_v2.j, rule = preprocessing_T0_rule)
    
    # the value of gamma an eta variables are no greater than 1
    # *** This constraint should be removed from SP, because it does not have variables.
    # print ('t...')
    # def ub_gamma_rule(model_Bender_SP_v2, i, j, k, u):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.gamma[i,j,k,u] >= -1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.ub_gamma_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.u, rule = ub_gamma_rule)
    
    # *** This constraint should be removed from SP, because it does not have variables.
    # def ub_eta_rule(model_Bender_SP_v2, i, j, k, v):
    #     if k in Trains[j]['containers']:
    #         return - model_Bender_SP_v2.eta[i,j,k,v] >= -1
    #     else:
    #         return pe.Constraint.Skip
    # model_Bender_SP_v2.ub_eta_rule = pe.Constraint(model_Bender_SP_v2.i, model_Bender_SP_v2.j, model_Bender_SP_v2.k, model_Bender_SP_v2.v, rule = ub_eta_rule)
    
        
    # solve the model
    # print('Solving...')    
    # Solve with Gurobi
    solver = SolverFactory('gurobi')
    solver.options['DualReductions'] = 0  # Ensure extreme ray is available
    solver.options['PreSolve'] = 0  # Disable preprocessing
    solver.options["LogToConsole"] = 0  # Disable Gurobi console output
    results = solver.solve(model_Bender_SP_v2, tee=False)
    
    # Check for infeasibility
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("\nPrimal LP is infeasible. Extracting dual information...")
        upper_bound, lower_bound = 'NA', 'NA'
        gap = 'NA'
        
        # Step 1: Save Pyomo model in LP format
        lp_filename = "pyomo_gurobi_model_Bender_SP_v2.lp"
        model_Bender_SP_v2.write(lp_filename, format='lp')

        # Step 2: Load into Gurobi for advanced analysis
        gurobi_model_Bender_SP_v2 = gp.read(lp_filename)
        gurobi_model_Bender_SP_v2.setParam("Method", 1)  # Ensure Dual Simplex is used
        gurobi_model_Bender_SP_v2.setParam("InfUnbdInfo", 1)  # Enable infeasibility certificate
        gurobi_model_Bender_SP_v2.setParam("OutputFlag", 0)  # Disable Gurobi console output
        gurobi_model_Bender_SP_v2.optimize()

        # constraint name mapping
        constraint_mapping = {}
        index = 0
        for pyomo_constr in model_Bender_SP_v2.component_objects(Constraint, active=True):
            pyomo_name = pyomo_constr.name
            for index_pyomo in pyomo_constr:
                # print (pyomo_name, index_pyomo)
                constraint = pyomo_constr[index_pyomo]
                gurobi_name = gurobi_model_Bender_SP_v2.getConstrs()[index].ConstrName  # Gurobi-generated name
                constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                index +=1
                
        # Step 4: Extract the Farkas dual extreme ray (dual infeasibility certificate)
        if gurobi_model_Bender_SP_v2.status == gp.GRB.INFEASIBLE:
            print("Gurobi confirms infeasibility.")

            '''-------------extract a dual extreme point-----------'''
            Q += 1
            pi.update({Q: {}})
            pi[Q].update({'source_iteration': itr_while})
            pi[Q].update({'source_type': 'extreme_point'})
            pi[Q].update({'source_model': 'SP'})
            EP.update({'%d'%itr_while + 'SP': {}})
            EP['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
            EP['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
            # Step 1: Save Pyomo model to a file
            lp_filename = "pyomo_gurobi_model_Bender_SP_v2.lp"
            model_Bender_SP_v2.write(lp_filename, format='lp')
        
            # Step 2: Load the model_Bender_SP_v2 into Gurobi
            gurobi_model_Bender_SP_v2 = gp.read(lp_filename)
            gurobi_model_Bender_SP_v2.setParam("DualReductions", 0)  # Ensure extreme ray is available
            gurobi_model_Bender_SP_v2.setParam("PreSolve", 0)  # Disable preprocessing
            gurobi_model_Bender_SP_v2.setParam("OutputFlag", 0)  # Disable Gurobi console output
        
            # Step 3: Optimize the model_Bender_SP_v2 with Gurobi (to compute infeasibility certificate)
            gurobi_model_Bender_SP_v2.optimize()
        
            # Step 4: Extract dual values (for constraints) if available    
            try:
                # Fetch duals from the constraints
                dual_values = gurobi_model_Bender_SP_v2.getAttr("Pi", gurobi_model_Bender_SP_v2.getConstrs())
                # Step 5: Map Pyomo constraint names to Gurobi constraint names
                constraint_mapping = {}
                index = 0
                for pyomo_constr in model_Bender_SP_v2.component_objects(Constraint, active=True):
                    pyomo_name = pyomo_constr.name
                    for index_pyomo in pyomo_constr:
                        # print (pyomo_name, index_pyomo)
                        constraint = pyomo_constr[index_pyomo]
                        gurobi_name = gurobi_model_Bender_SP_v2.getConstrs()[index].ConstrName  # Gurobi-generated name
                        constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                        index +=1
                # Step 6: Print dual values with mapped names
                # print("\nDual values (Dual Infeasibility Certificate):")
                for constr, dual_value in zip(gurobi_model_Bender_SP_v2.getConstrs(), dual_values):
                    original_name = constraint_mapping.get(constr.ConstrName, constr.ConstrName)
                    # print(f"Solver constraint: {constr.ConstrName} <==> Pyomo constraint: {original_name}, Dual: {dual_value}")
                    pyomo_name = original_name.partition(' -- ')[0]
                    index_pyomo = original_name.partition(' -- ')[2]
                    if pyomo_name not in pi[Q]:                    
                        pi[Q].update({pyomo_name: {}})
                    if pyomo_name not in EP['%d'%itr_while + 'SP']:                    
                        EP['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                    # split index_pyomo by ", " and "(" and ")"
                    index_list_quote = index_pyomo.split(', ')
                    if len(index_list_quote) >= 2:
                        index_list_quote[0] = index_list_quote[0][1:]
                        index_list_quote[-1] = index_list_quote[-1][:-1]
                    index_list = [item.strip("'") for item in index_list_quote]
                    pi[Q][pyomo_name].update({tuple(index_list): dual_value})
                    EP['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): dual_value})
            except gp.GurobiError as e:
                print(f"GurobiError while extracting duals: {e}")

            '''-------------extract a dual extreme ray-------------'''
            ER.update({'%d'%itr_while + 'SP': {}})
            ER['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
            ER['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
            # Get Farkas dual values
            print("\nDual Extreme Ray (Farkas Certificate):")
            for constr in gurobi_model_Bender_SP_v2.getConstrs():
                try:
                    farkas_dual = -constr.getAttr("FarkasDual")  # Extract Farkas dual values
                    original_name = constraint_mapping[constr.constrName]
                    pyomo_name = original_name.partition(' -- ')[0]
                    index_pyomo = original_name.partition(' -- ')[2]
                    if pyomo_name not in ER['%d'%itr_while + 'SP']:                    
                        ER['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                    # split index_pyomo by ", " and "(" and ")"
                    index_list_quote = index_pyomo.split(', ')
                    if len(index_list_quote) >= 2:
                        index_list_quote[0] = index_list_quote[0][1:]
                        index_list_quote[-1] = index_list_quote[-1][:-1]
                    index_list = [item.strip("'") for item in index_list_quote]
                    ER['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): farkas_dual})
                    # print(f"Constraint {original_name}: Dual Extreme Ray = {farkas_dual}")
                except gp.GurobiError as e:
                    print(f"Could not retrieve FarkasDual for constraint {constr.constrName}: {e}")

            '''-------------add vector ER['%d'%itr_while + 'SP'] to pi[Q]-------------'''
            Q += 1
            pi.update({Q: {}})
            pi[Q].update({'source_iteration': itr_while})
            pi[Q].update({'source_type': 'extreme_ray'})
            pi[Q].update({'source_model': 'SP'})
            for name in ER['%d'%itr_while + 'SP']:
                if name != 'source_model' and name != 'source_iteration':
                    if name not in pi[Q]:
                        pi[Q].update({name: {}})
                    for ind in ER['%d'%itr_while + 'SP'][name]:
                        er_value = ER['%d'%itr_while + 'SP'][name][ind]
                        pi[Q][name].update({ind: (ratio_ER**itr_while)*er_value})
            
    else:
        print("Primal LP is feasible or has another issue.")
        # calculate optimality gap
        upper_bound, lower_bound = results.problem.upper_bound, results.problem.lower_bound
        gap = (results.problem.upper_bound - results.problem.lower_bound)/float(results.problem.upper_bound)
        
        '''extract a dual extreme point'''
        Q += 1
        pi.update({Q: {}})
        pi[Q].update({'source_iteration': itr_while})
        pi[Q].update({'source_type': 'extreme_point'})
        pi[Q].update({'source_model': 'SP'})
        EP.update({'%d'%itr_while + 'SP': {}})
        EP['%d'%itr_while + 'SP'].update({'source_model': 'SP'})
        EP['%d'%itr_while + 'SP'].update({'source_iteration': itr_while})
        # Step 1: Save Pyomo model to a file
        lp_filename = "pyomo_gurobi_model_Bender_SP_v2.lp"
        model_Bender_SP_v2.write(lp_filename, format='lp')
    
        # Step 2: Load the model_Bender_SP_v2 into Gurobi
        gurobi_model_Bender_SP_v2 = gp.read(lp_filename)
        gurobi_model_Bender_SP_v2.setParam("DualReductions", 0)  # Ensure extreme ray is available
        gurobi_model_Bender_SP_v2.setParam("PreSolve", 0)  # Disable preprocessing
        gurobi_model_Bender_SP_v2.setParam("OutputFlag", 0)  # Disable Gurobi console output
    
        # Step 3: Optimize the model_Bender_SP_v2 with Gurobi (to compute infeasibility certificate)
        gurobi_model_Bender_SP_v2.optimize()
        # Step 4: Extract dual values (for constraints) if available    
        try:
            # Fetch duals from the constraints
            dual_values = gurobi_model_Bender_SP_v2.getAttr("Pi", gurobi_model_Bender_SP_v2.getConstrs())
            # Step 5: Map Pyomo constraint names to Gurobi constraint names
            constraint_mapping = {}
            index = 0
            for pyomo_constr in model_Bender_SP_v2.component_objects(Constraint, active=True):
                pyomo_name = pyomo_constr.name
                for index_pyomo in pyomo_constr:
                    # print (pyomo_name, index_pyomo)
                    constraint = pyomo_constr[index_pyomo]
                    gurobi_name = gurobi_model_Bender_SP_v2.getConstrs()[index].ConstrName  # Gurobi-generated name
                    constraint_mapping[gurobi_name] = pyomo_name + ' -- %s' %str(index_pyomo)
                    index +=1
            # Step 6: Print dual values with mapped names
            # print("\nDual values (Dual Infeasibility Certificate):")
            for constr, dual_value in zip(gurobi_model_Bender_SP_v2.getConstrs(), dual_values):
                original_name = constraint_mapping.get(constr.ConstrName, constr.ConstrName)
                # print(f"Solver constraint: {constr.ConstrName} <==> Pyomo constraint: {original_name}, Dual: {dual_value}")
                pyomo_name = original_name.partition(' -- ')[0]
                index_pyomo = original_name.partition(' -- ')[2]
                if pyomo_name not in pi[Q]:                    
                    pi[Q].update({pyomo_name: {}})
                if pyomo_name not in EP['%d'%itr_while + 'SP']:                    
                    EP['%d'%itr_while + 'SP'].update({pyomo_name: {}})
                # split index_pyomo by ", " and "(" and ")"
                index_list_quote = index_pyomo.split(', ')
                if len(index_list_quote) >= 2:
                    index_list_quote[0] = index_list_quote[0][1:]
                    index_list_quote[-1] = index_list_quote[-1][:-1]
                index_list = [item.strip("'") for item in index_list_quote]
                pi[Q][pyomo_name].update({tuple(index_list): dual_value})
                EP['%d'%itr_while + 'SP'][pyomo_name].update({tuple(index_list): dual_value})
        except gp.GurobiError as e:
            print(f"GurobiError while extracting duals: {e}")
       
        
    # opt= pyomo.opt.SolverFactory("cplex")
    # # optimality_gap = 0.05
    # # opt.options["mip_tolerances_mipgap"] = optimality_gap
    # opt.options["timelimit"] = 900
    # results=opt.solve(model_Bender_SP_v2, tee=False, keepfiles=False)
    # results.write()
    
    # # solve by ipopt: treats the variables as continuous
    # solver = SolverFactory('ipopt')
    # #solver.options['max_iter']= 10000
    # results= solver.solve(model_Bender_SP_v2, tee=True)    
    # results.write()

    # opt=SolverFactory('apopt.py')
    # results=opt.solve(model_Bender_SP_v2)
    # results.write()
    # instance.load(results)
    
    # solve by Couenne: an open solver for mixed integer nonlinear problems
    # # import os
    # # from pyomo.environ import *
    # couenne_dir = r'C:\\Users\\z3547138\\AMPL'
    # os.environ['PATH'] = couenne_dir + os.pathsep + os.environ['PATH']
    # opt = SolverFactory("couenne") 
    # # opt.options['logLevel'] = 5  # Verbosity level (higher = more output)
    # # opt.options['timeLimit'] = 120  # Set maximum time limit (in seconds)
    # # opt.options['logFile'] = 'couenne_log.txt'
    # results=opt.solve(model_Bender_SP_v2, timelimit=300, logfile = 'mylog.txt', tee=True)
    # results.write()
    
    # calculate obj
    if results.solver.termination_condition == TerminationCondition.optimal:
        cost_fix = sum(Stations[i]['cost_fix'] * X[itr_while][i] for i in model_Bender_SP_v2.i)
        cost_delay = sum(sum((model_Bender_SP_v2.D[i,j].value - Trains[j]['stations'][i]['time_wait']) for j in model_Bender_SP_v2.j) for i in model_Bender_SP_v2.i)
        obj = penalty_fix_cost * cost_fix + penalty_delay * cost_delay
        obj_SP.update({itr_while:obj})
        if itr_while > 0:
            if obj < UB[itr_while-1]:
                UB[itr_while] = obj
            else:
                UB[itr_while] = UB[itr_while-1]
    else:
        UB[itr_while] = UB[itr_while-1]
        obj_SP.update({itr_while: 'None'})
    
    results_SP.update({itr_while: results})
    
    time_end = time.time()
    time_model = time_end - time_start

    
    
    '''Record variables'''
    # print ('Recording variables...')
    D.update({itr_while:{}})
    T_arrive.update({itr_while:{}})
    T_depart.update({itr_while:{}})
    
    for i in Stations:
        D[itr_while].update({i: {}})
        T_arrive[itr_while].update({i: {}})
        T_depart[itr_while].update({i: {}})
        for j in Trains:
            D[itr_while][i].update({j: model_Bender_SP_v2.D[i,j].value})
            T_arrive[itr_while][i].update({j: model_Bender_SP_v2.T_arrive[i,j].value})
            T_depart[itr_while][i].update({j: model_Bender_SP_v2.T_depart[i,j].value})
    
    
    
    return obj_SP, UB, D, T_arrive, T_depart, results_SP, time_model, gap, upper_bound, lower_bound, pi, Q, ER, EP
    





