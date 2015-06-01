from pom3_teams import *
from pom3_requirements import *
import random

"""##############################################################################
   ### -@author: Joseph Krall
   ### -@note: POM3 Simulation Module
   ### -@note: This work done in affiliation with the West Virginia University
   ### -@contact: kralljoe@gmail.com
   ### -@copyright: This work is made for academic research.  Do not edit without
   ### -@copyright: citing POM3 as a reference.
   ##############################################################################"""

class pom3_decisions:
    def __init__(p3d, X):
        p3d.culture = X[0]
        p3d.criticality = X[1]
        p3d.criticality_modifier = X[2]
        p3d.initial_known = X[3]
        p3d.interdependency = X[4]
        p3d.dynamism = X[5]
        p3d.size = int(X[6])
        p3d.plan = int(X[7])
        p3d.team_size = X[8]
        
class pom3:
    
    def __init__(p3):
     pass
            
    def simulate(p3, inputs):
    
        # # # # # # # # # # #
        # 0) Initialization #
        # # # # # # # # # # #
        
        POM3_DECISIONS = pom3_decisions(inputs)
        numberOfShuffles = random.randint(2,6)
    
        # # # # # # # # # # # # # # #
        # 1) Generate Requirements  #
        # # # # # # # # # # # # # # #
        
        POM3_REQUIREMENTS = pom3_requirements(POM3_DECISIONS)
        
        # # # # # # # # # # #
        # 2) Generate Teams #
        # # # # # # # # # # #
        
        POM3_TEAMS = pom3_teams(POM3_REQUIREMENTS, POM3_DECISIONS)
        
        # # # # # # # #
        # 3) Shuffle  #
        # # # # # # # #
        
        
        for shufflingIteration in range(numberOfShuffles):
            for team in POM3_TEAMS.teams:
                team.updateBudget(numberOfShuffles)
                team.collectAvailableTasks(POM3_REQUIREMENTS)
                team.applySortingStrategy()
                team.executeAvailableTasks()
                team.discoverNewTasks()
                team.updateTasks()


            
        # # # # # # # # # # # # #
        # 4) Objective Scoring  #
        # # # # # # # # # # # # #
        
        cost_sum,value_sum,god_cost_sum,god_value_sum,completion_sum,available_sum,total_tasks = 0.0, 0.0, 0.0, 0.0, 0,0,0
        for team in POM3_TEAMS.teams:
            cost_sum += team.cost_total
            value_sum += team.value_total
            available_sum += team.numAvailableTasks
            completion_sum += team.numCompletedTasks
            for task in team.tasks:
                if task.val.visible:
                    total_tasks += 1
            
            for task in team.tasks:
                if task.val.done == True:
                    god_cost_sum += task.val.cost
                    god_value_sum += task.val.value
                    
        if cost_sum == 0: our_frontier = 0.0
        else: our_frontier =     value_sum /     cost_sum
        
        if god_cost_sum == 0: god_frontier = 0.0
        else: god_frontier = god_value_sum / god_cost_sum
        
        if god_frontier == 0.0: score = 0.0
        else: score        =  our_frontier / god_frontier
        
        #print "cost",cost_sum,"value",value_sum,"completion",completion_sum,"avaiable",available_sum,"tot tasks",total_tasks
        if completion_sum == 0: cost = 0
        else: cost = cost_sum/completion_sum
        
        if available_sum == 0: idle = 0
        else: idle = 1 - completion_sum/float(available_sum)
        
        if total_tasks == 0: completion = 0
        else: completion = completion_sum/float(total_tasks)
        
        
        
        #return [cost, score, completion, idle]
        return [cost, completion, idle]
        
        

# Test Code 
#p3 = pom3()
#print p3.simulate([0.20, 1.26, 8, 0.95, 100, 10, 2, 5, 20])

 
           
