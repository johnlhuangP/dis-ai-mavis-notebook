# Import necessary modules from your files
import os
import subprocess
import numpy as np

from tqdm.notebook import tqdm

from searchclient.agent_types.classic import * 

# Import all action classes (used for hardcoding solutions) and actions libraries
from searchclient.domains.hospital.actions import (
    NoOpAction, MoveAction, PushAction, PullAction, AnyAction, DEFAULT_MAPF_ACTION_LIBRARY, DEFAULT_HOSPITAL_ACTION_LIBRARY
)

# Import state, goal description and level classes for the MAvis hospital environment
from searchclient.domains.hospital.state import HospitalState
from searchclient.domains.hospital.goal_description import HospitalGoalDescription
from searchclient.domains.hospital.level import HospitalLevel

# Import the Graph-Search algorithm
from searchclient.search_algorithms.graph_search import graph_search

# Import the different search strategies for both uninformed and informed search
from searchclient.strategies.bfs import FrontierBFS
from searchclient.strategies.dfs import FrontierDFS
from searchclient.strategies.bestfirst import FrontierBestFirst, FrontierGreedy, FrontierAStar

# Import heuristic classes, to be used in informed search methods
from searchclient.domains.hospital.heuristics import (
    HospitalZeroHeuristic, HospitalGoalCountHeuristics, HospitalAdvancedHeuristics
)
    
from searchclient.domains.hospital.level import HospitalLevel

print(os.getcwd())
def load_level_file_from_path(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.strip(), lines))
        return lines
    
def render_plan(level_path, plan, strategy_name, heuristic_name, num_generated, elapsed_time, sol_length):

    str_plan = convert_plan_to_string(plan) #convert the plan to a string

    # this just makes sure that the meta information is displayed correctly in the visualization
    if strategy_name == 'greedy' or strategy_name == 'astar':
        strategy_name_pygame = strategy_name + ' w. ' + heuristic_name
    else:
        strategy_name_pygame = strategy_name
    
    subprocess.run(["python", 
                    "renderMAvis.py", 
                    "--level", level_path, 
                    "--plan", str_plan, 
                    "--search_strategy", strategy_name_pygame, 
                    "--num_generated", str(num_generated), 
                    "--time_elapsed", str(elapsed_time), 
                    "--sol_length", str(sol_length)])

def test_graph_search(level_path, frontier, frontier_as_string, heuristic_name):
    level_path = level_path
    level_lines = load_level_file_from_path(level_path)
    level = HospitalLevel.parse_level_lines(level_lines)

    num_agents = len(level.initial_agent_positions)
    action_set = [DEFAULT_HOSPITAL_ACTION_LIBRARY for _ in range(num_agents)]
    initial_state = HospitalState(level, level.initial_agent_positions, level.initial_box_positions)
    goal_description = HospitalGoalDescription(level, level.box_goals + level.agent_goals)

    frontier = frontier
    success, plan, num_generated, elapsed_time = graph_search(initial_state, action_set, goal_description, frontier)

    render_plan(level_path, plan, frontier_as_string, heuristic_name, num_generated, elapsed_time, len(plan))


def frontierGreedyTest():
    heuristic = HospitalGoalCountHeuristics()
    greedy_frontier = FrontierGreedy(heuristic)
    test_graph_search("levels/SimpleDebug.lvl", greedy_frontier, "greedy", "goal_count")


    greedy_frontier2 = FrontierGreedy(heuristic)
    test_graph_search("levels/TwoAgentsDebug.lvl", greedy_frontier2, "greedy", "goal_count")

def AStarTest():
    heuristic = HospitalGoalCountHeuristics()
    astar_frontier = FrontierAStar(heuristic)
    test_graph_search("levels/SimpleDebug.lvl", astar_frontier, "astar", "goal_count")

    test_graph_search("levels/TwoAgentsDebug.lvl", astar_frontier, "astar", "goal_count")

frontierGreedyTest()
AStarTest()