import agent_class
import networkx as nx


def initialize_agents(config, openai_key):
  '''
  Create the initial network and agents for the simulation
  Inputs:
  - config: configuration file to specify how to create the agent
  Outputs:
  - network: Networkx Graph object containing vertices for each agent in range n_agents in config
  - agent_dict: dictionary {agent_id: agent_class Instance}
  '''
  agent_dict = {}
  network = nx.Graph()
  for i in range(config['n_agents']):
    network.add_node(i)
    agent_dict[i] = agent_class.Agent(id = i, config = config, openai_key = openai_key, n_iters = config['n_iters'])
  return network, agent_dict
  