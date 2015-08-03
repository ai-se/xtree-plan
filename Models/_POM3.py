import sys
import os
_pwd = os.getcwd()
sys.path.insert(0, _pwd + '/pom3/')
from pom3.pom3_teams import *
from pom3.pom3_requirements import *
import random
from numpy.random.mtrand import shuffle

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


class pom3():
  # Initialization

  def __init__(my):
    my.cost_sum, my.value_sum, my.god_cost_sum = 0.0, 0.0, 0.0
    my.god_value_sum, my.completion_sum = 0.0, 0.0
    my.available_sum, my.total_tasks = 0.0, 0.0

  def POM3_DECISIONS(my, x):
    return pom3_decisions(x)

  # Generate Requirements

  def POM3_REQUIREMENTS(my, x):
    return pom3_requirements(my.POM3_DECISIONS(x))

  # Generate Teams

  def POM3_TEAMS(my, x):
    return pom3_teams(my.POM3_REQUIREMENTS(x),
                      my.POM3_DECISIONS(x))

  # Shuffle

  def shuffle(my, x):
    numberOfShuffles = random.randint(2, 6)
    for shufflingIteration in range(numberOfShuffles):
      for team in my.POM3_TEAMS(x).teams:
        team.updateBudget(numberOfShuffles)
        team.collectAvailableTasks(my.POM3_REQUIREMENTS(x))
        team.applySortingStrategy()
        team.executeAvailableTasks()
        team.discoverNewTasks()
        team.updateTasks()

  # Objective Scoring

  def objective(my, x):
    my.cost_sum, my.value_sum, my.god_cost_sum = 0.0, 0.0, 0.0
    my.god_value_sum, my.completion_sum = 0.0, 0.0
    my.available_sum, my.total_tasks = 0.0, 0.0

    my.shuffle(x)
    for team in my.POM3_TEAMS(x).teams:
      my.cost_sum += team.cost_total
      print my.cost_sum
      my.value_sum += team.value_total
      my.available_sum += team.numAvailableTasks
      my.completion_sum += team.numCompletedTasks
      for task in team.tasks:
        if task.val.visible:
          my.total_tasks += 1

      for task in team.tasks:
        if task.val.done:
          my.god_cost_sum += task.val.cost
          my.god_value_sum += task.val.value

    return {
        'cost_sum': my.cost_sum,
        'value_sum': my.value_sum,
        'god_cost_sum': my.god_cost_sum,
        'god_value_sum': my.god_value_sum,
        'completion_sum': my.completion_sum,
        'available_sum': my.available_sum,
        'total_tasks': my.total_tasks}

    if my.cost_sum == 0:
      my.our_frontier = 0.0
    else:
      my.our_frontier = my.value_sum / my.cost_sum

    if my.god_cost_sum == 0:
      my.our_frontier = 0.0
    else:
      my.our_frontier = my.god_value_sum / my.god_cost_sum

    if my.our_frontier == 0.0:
      my.score = 0.0
    else:
      my.score = my.our_frontier / my.our_frontier

  # print
  # "cost",cost_sum,"value",value_sum,"completion",completion_sum,"avaiable",available_sum,"tot
  # tasks",total_tasks

  def costFxn(my):

    if my.completion_sum == 0:
      cost = 0
    else:
      cost = my.cost_sum / my.completion_sum

    if my.available_sum == 0:
      idle = 0
    else:
      idle = 1 - my.completion_sum / float(my.available_sum)

    if my.total_tasks == 0:
      completion = 0
    else:
      completion = my.completion_sum / float(my.total_tasks)

    return [cost, idle, completion]

  # return [cost, my.score, completion, idle]
  def simulate(p3, inputs):

    # # # # # # # # # # #
    # 0) Initialization #
    # # # # # # # # # # #

    POM3_DECISIONS = pom3_decisions(inputs)
    numberOfShuffles = random.randint(2, 6)

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

    cost_sum, value_sum, god_cost_sum, god_value_sum, completion_sum, available_sum, total_tasks = 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    for team in POM3_TEAMS.teams:
      cost_sum += team.cost_total
      value_sum += team.value_total
      available_sum += team.numAvailableTasks
      completion_sum += team.numCompletedTasks
      for task in team.tasks:
        if task.val.visible:
          total_tasks += 1

      for task in team.tasks:
        if task.val.done:
          god_cost_sum += task.val.cost
          god_value_sum += task.val.value

    if cost_sum == 0:
      our_frontier = 0.0
    else:
      our_frontier = value_sum / cost_sum

    if god_cost_sum == 0:
      god_frontier = 0.0
    else:
      god_frontier = god_value_sum / god_cost_sum

    if god_frontier == 0.0:
      score = 0.0
    else:
      score = our_frontier / god_frontier

    # print
    # "cost",cost_sum,"value",value_sum,"completion",completion_sum,"avaiable",available_sum,"tot
    # tasks",total_tasks
    if completion_sum == 0:
      cost = 0
    else:
      cost = cost_sum / completion_sum

    if available_sum == 0:
      idle = 0
    else:
      idle = 1 - completion_sum / float(available_sum)

    if total_tasks == 0:
      completion = 0
    else:
      completion = completion_sum / float(total_tasks)

    # return [cost, score, completion, idle]
    return [cost, completion, idle]


def pom3_do():
  pass


if __name__ == '__main__':
  # Test Code
  # p3 = pom3()
  # print p3.simulate([0.20, 1.26, 8, 0.95, 100, 10, 2, 5, 20])
  pass
