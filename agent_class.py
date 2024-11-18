from click import prompt
import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import time



class Agent:
  def __init__(self, id, config, openai_key, n_iters):
    self.id = id
    self.config = config
    if self.id % 2 == 0:
      self.consensus_0_reward,  self.consensus_1_reward = config['agent_1']['consensus_0_reward'], config['agent_1']['consensus_1_reward']
      self.edge_cost = config['agent_1']['edge_cost']
    else:
      self.consensus_0_reward,  self.consensus_1_reward = config['agent_2']['consensus_0_reward'], config['agent_2']['consensus_1_reward']
      self.edge_cost = config['agent_2']['edge_cost']
    self.llm = OpenAI(api_key = openai_key)
    self.color = None
    self.color = self.choose_color()
    self.neighbor_colors = {}
    self.neighbor_proximity = {}
    self.projected_reward = {0: self.consensus_0_reward,
                             1: self.consensus_1_reward}
    self.n_iters = n_iters
    self.iters_remaining = n_iters
    



  def choose_color(self):
    """
    Inputs: self
    Outputs: 0 or 1 representing chosen color. None if an invalid option is chosen. 
    """
    print(f"RUNNING CHOOSE COLOR ON AGENT {self.id}")
    if self.color is None:
      self.color = np.random.choice([0,1])
      return self.color

    else:
      # Set up ConversationChain
      choose_color = ConversationChain(
        llm = self.llm,
        memory = ConversationBufferMemory(),
        verbose = True
      )

      if choose_color.memory.chat_memory.messages:
        print("Clearing residual memory in choose_color.")
        choose_color.memory.clear()

      preferred_consensus = choose_color.run(f"{self.config['preferred_consensus_prompt']}")
      print(preferred_consensus)

      most_likely_consensus = choose_color.run(f"{self.config['most_likely_consensus_prompt']}")
      print(most_likely_consensus)

      color = choose_color.run(f"{self.config['choose_color_prompt']}")
      # Create separate chain to parse the output of the edge selection: 
      color = color.strip()

      if color.isnumeric():
        try:
          color = int(color)
          choose_color.memory.clear()
          return (color)

        except:
          print(f"Failed to choose {self.id} and {color} to int at iteration {self.n_iters - self.iters_remaining}. Leaving as none")
          choose_color.memory.clear()
          return None
    


  def buy_edge(self):
    """
    Inputs: None
    Outputs: (self.id, int(formatted_edge_selection)): tuple representing an edge if a valid edge is purchased, else None
    
    """
    print(f"Running BUY EDGE on agent {self.id}")
    if (min(self.projected_reward.values()) > self.edge_cost):
      # Set up ConversationChain
      choose_edge = ConversationChain(
        llm = self.llm,
        memory = ConversationBufferMemory(),
        verbose = False
      )

      preferred_consensus = choose_edge.run(f"{self.config['preferred_consensus_prompt']}")
      print(f"Preferred Consensus \n Prompt: {self.config['preferred_consensus_prompt']} \n Response: {preferred_consensus}")

      most_likely_consensus = choose_edge.run(f"{self.config['most_likely_consensus_prompt']}")
      print(f"Most Likely Consensus \n Prompt: {self.config['most_likely_consensus_prompt']} \n Response: {most_likely_consensus}")

      edge_reasoning = choose_edge.run(f"{self.config['edge_reasoning_prompt']}")
      print(f"Edge Reasoning \n Prompt: {self.config['edge_reasoning_prompt']} \n Response: {edge_reasoning}")

      real_edge_evaluation = choose_edge.run(f"{self.config['real_edge_evaluation_prompt']}")
      print(f"Real Edge Evaluation Prompt \n Prompt: {self.config['real_edge_evaluation_prompt']} \n Response: {real_edge_evaluation}")

      real_edge_cost_benefit = choose_edge.run(f"{self.config['real_edge_cost_benefit_prompt']}")
      print(f"Real Edge Cost Benefit \n Prompt: {self.config['real_edge_cost_benefit_prompt']} \n Response: {real_edge_cost_benefit}")

      edge_selection = choose_edge.run(f"{self.config['edge_selection_prompt']}")
      print(f"Edge Selection \n Prompt: {self.config['edge_selection_prompt']} \n Response: {edge_selection}")

      # Create separate chain to parse the output of the edge selection: 
      QA_prompt = PromptTemplate(
                input_variables = ["edge_selection"],
                template = """
                You will be given a natural-language request by one agent to form an edge with another agent. Format the request so that it gives ONLY the agent_id of the desired agent as an INTEGER. 
                The INTEGER you return will DIRECTLY be used to index a dictionary. 
                Ex.)
                request: "I want to form an edge with Agent 5"; Format as: 5
                request: "Agent 5"; Format as: 5
                request: "-1"; Format as: -1

                The request is as follows: {edge_selection}
                """
            )
      
      QA_edge_selection_chain = LLMChain(llm = self.llm, prompt = QA_prompt)
      formatted_edge_selection = QA_edge_selection_chain.run({"edge_selection": edge_selection})
      print(f"formatted edge selection: {formatted_edge_selection}")
      formatted_edge_selection = formatted_edge_selection.strip()
      # print(f"Agent {self.id} Answer: {formatted_edge_selection}")

      # For Testing Purposes
      # if self.id == 3:
      #   print(f"consensus_0_reward: {self.consensus_0_reward}")
      #   print(f"consensus_1_reward: {self.consensus_1_reward}")
      #   print(f"neighbor_colors: {self.neighbor_colors}")
      #   print(f"edge_cost: {self.edge_cost}")
      #   print(f"projected_reward: {min(self.projected_reward[0], self.projected_reward[1])}")
      #   print(f"neighbor_proximity: {self.neighbor_proximity}")

      # Show error if edge selection is not numeric
      if not formatted_edge_selection.isnumeric():
          if formatted_edge_selection == str(-1):
            choose_edge.memory.clear()
            return None
          print(f"Agent {self.id} gives non_numeric answer at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
          choose_edge.memory.clear()
          return None
      
      # If the selected edge is valid return the edge as an int to store in the dictionary of edges
      if int(formatted_edge_selection) in self.neighbor_proximity and  int(formatted_edge_selection) not in self.neighbor_colors:
        choose_edge.memory.clear()
        print(f"Edge is valid. Agent {self.id} selects edge to {formatted_edge_selection}")
        return (self.id, int(formatted_edge_selection))
      
      # If the selected edge is already visible to the AGENT return nothing (don't purchase edge)
      else:
        print(f"Agent {self.id} gives invalid agent at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
        choose_edge.memory.clear()
        return None
    return None
        