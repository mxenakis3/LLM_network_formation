import networkx as nx
import config_utils
import simulation_utils as sim_utils
import numpy as np
from dotenv import load_dotenv
import os
import warnings
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time

warnings.filterwarnings("ignore")

class Simulation:
  def __init__(self, config):
    load_dotenv()
    self.api_key = os.getenv('OPENAI_API_KEY')
    self.n_iters = config['n_iters']
    self.network, self.agents = sim_utils.initialize_agents(config, self.api_key)
    self.new_edges = set() # Set of new edges that are purchased at the end of each time step
    self.new_colors = {} # Set of new colors for each agent at end of each time step
    self.spls = None # Shortest path lengths between agents at a given time step
    self.color_tracker = pd.DataFrame(columns=[i for i in range(config['n_agents'])])
    # Populate tracker with the original colors
    # print(f"og_color_tracker: {self.color_tracker}")
    # print(f"og colors: {[agent.color for agent in self.agents.values()]}")
    self.color_tracker.loc[0] = [agent.color for agent in self.agents.values()]
    self.curr_time = 0


  def run(self):
    """
    Main Loop
    """
    # Store data in DataFrames
    start_time = time.time()
    for t in tqdm(range(self.n_iters), desc="Iteration"):
    # for t in range(self.n_iters):
      print(f"Current iteration: {t}/{self.n_iters}")
      # If a consensus has been reached, return the iteration number
      # if self.reached_consensus():
        # print(f"Consensus has been reached at iteration {t}")
      self.spls = dict(nx.all_pairs_shortest_path_length(self.network))
      print(f"Shortest path lengths calculating: {self.spls}")
      self.update_neighbor_info()
      print(f"Neighbors updated. Showing neighbor_proximity and neighbor_color for agent 0:")
      print(f"\t colors: {self.agents[0].neighbor_colors}")
      print(f"\t proximity: {self.agents[0].neighbor_proximity}")
      self.store_edge_purchases()
      print(f"Edges stored. New edge dictionary: \n \t {self.new_edges}")
      self.agents_choose_colors() # Agents choose based on colors from last iteration
      print(f"Colors stored. New color dictionary: \n \t {self.new_colors}")
      print(f"edges in og network: {len(self.network.edges)}")
      self.update_network()
      print(f"Updating network. edges in new network: {len(self.network.edges)}")
      nx.draw(self.network)
      self.update_colors()
      print(f"Updating colors: {self.new_colors}")
      self.update_time()
      # print(f"Color tracking: {self.color_tracker}")

    ### TIME EXPIRES ###
    # Return iteration number if consensus is reached, otherwise return -1
    end_time = time.time()
    runtime = start_time - end_time
    return runtime


  def reached_consensus(self):
    """
    Check that all nodes have the same color.
    """
    # Get the color of the first agent
    first_agent_color = next(iter(self.agents.values())).color
    
    # Check if all agents have the same color as the first one
    for agent in self.agents.values():
        if agent.color != first_agent_color:
            return False
    return True


  def update_neighbor_info(self):
    """
    Updates 'neighbor_colors' and 'neighbor_proximity' for each agent at the beginning of each timestep.
    """
    adj_matrix = nx.to_numpy_array(self.network)
    # id is the agent id for the agent in the full set of agents that we are currently iterating over. Agent is the instance of the agent class. 
    for id, agent in self.agents.items():
      # Get neighbors = adjacency list (neighbors) of current network corresponding to each agent
      neighbors = adj_matrix[id] # since the agent id's are zero-indexed, the i'th row of the adjacency matrix is the adjacency list for agent 'i'.
      for n_id, x in enumerate(neighbors): # iterate over adjacency list. n_id is the id of the neighbor
        if id == n_id: 
          continue # No self edges
        if x > 0: # Edge exists between agent 'id' and neighbor 'n_id'
          agent.neighbor_colors[n_id] = self.agents[n_id].color # allow the agent to see the current color of its neighbor
        # else: # No edge exists between agent 'id' and neighbor 'n_id'
        agent.neighbor_proximity[n_id] = {'degree': self.network.degree[n_id], # Degree of agent 'n_d' is the current degree of network n_id
                                          'network_distance': self.spls.get(id, {}).get(n_id, 'No direct path exists to you') } # network distance to n_id is the current shortest path length between id and n_id


  def store_edge_purchases(self):
    """
    Ask each agent if they want to buy an edge. If they do, store it in the new_edges dictionary. 
    Then, collect their payment.
    """
    for agent in self.agents.values():
      edge_purchased = agent.buy_edge() # Buy edge returns
      if edge_purchased: # If buy edge returns non-null, an edge has been purchased
        if edge_purchased not in self.new_edges: # If the edge does not already exist in the set of new edges to add
          self.new_edges.add(edge_purchased) # Add tuple representing edge (id, n_id) to set of new edges
          u, v = edge_purchased
          u, v = int(u), int(v)
          for condition, reward in self.agents[u].projected_reward.items():
            self.agents[u].projected_reward[condition] = reward - self.agents[u].edge_cost

        else:
          # Case: two agents decided to buy links to each other.
          # Choose who buys edge randomly -- avoids making agents with low ID's (who choose first) more likely to pay.
          u, v = edge_purchased
          coin_flip = np.random.choice([0,1])
          if coin_flip == 0:
            for condition, reward in self.agents[u].projected_reward.items():
              self.agents[u].projected_reward[condition] = reward + self.agents[u].edge_cost # reimburse u
            for condition, reward in self.agents[v].projected_reward.items():
              self.agents[v].projected_reward[condition] = reward - self.agents[v].edge_cost # charge v


  def agents_choose_colors(self):
    """
    Ask each agent how they want to vote via 'agent.choose_color'.
    Store colors in separate dictionary and update at end of timestep.
    If invalid color choice, don't add to the dictionary 
    Entry: {agent_id: new_color}
    """
    for agent in self.agents.values():
      color_choice = agent.choose_color()
      if color_choice in set([0, 1]):
        self.new_colors[agent.id] = color_choice


  def update_network(self):
    """
    Adds in the newly purchased edges for the current timestep.
    """
    for edge in self.new_edges:
      u, v = edge
      self.network.add_edge(int(u), int(v))
      self.network.add_edge(int(v), int(u))

    self.new_edges = set()

  
  def update_colors(self):
      """
      Updates each agent's color according to their selection via 'new_colors'
      If their color choice was invalid, keep their old color
      """
      # Agent colors initially stores the original agent colors {id: agent.color}
      agent_colors = {agent_id: agent.color for agent_id, agent in self.agents.items()}

      for agent_id, color in self.new_colors.items():
          self.agents[agent_id].color = color
          agent_colors[agent_id] = self.new_colors[agent_id]
      
      self.color_tracker.loc[len(self.color_tracker)] = list(agent_colors.values()) # Log the colors


  def update_time(self):
    """
    Updates time for each agent
    """
    for agent in self.agents.values():
      agent.iters_remaining -= 1
    self.curr_time += 1
