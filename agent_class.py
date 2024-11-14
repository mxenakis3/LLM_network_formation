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
    self.consensus_0_reward = config['consensus_0_reward']
    self.consensus_1_reward = config['consensus_1_reward']
    self.edge_cost = config['edge_cost']
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
      print(f"Choose color chain is initialized with new ConversationBufferMemory")

      if choose_color.memory.chat_memory.messages:
        print("Clearing residual memory in choose_color.")
        choose_color.memory.clear()

      preferred_consensus = choose_color.run(
        f"""
        You are in a multi-agent game to win money, where each agent controls the color (either 0 or 1) of a vertex in a network.

        To win money, all vertices in the network must show the same color by the end of the game. 

        If the entire network picks color 0, your maximum projected payoff will be {self.consensus_0_reward}.
        If the entire network picks color 1, your maximum projected payoff will be {self.consensus_1_reward}
        If no consensus is reached, you will not win any money. 
        
        WOULD YOU PREFER A CONSENSUS OF 0 OR 1? WHY?
        WOULD YOU PREFER A CONSENSUS OR NO CONSENSUS? WHY?  
        """
      )
      most_likely_consensus = choose_color.run(
        f"""

        Agents are only able to see one another's colors by purchasing connections to one another, which in the network of agents/vertices can be thought of as edges connecting one agent to another.

        Through a dictionary, you are able to see the full set of the colors shown by agents with whom you are connected via edge purchases . 

        Each key in the dictionary represents the ID of another agent. Each value represents their color. 
        
        The dictionary is as follows:
        {self.neighbor_colors}. 
        
        The game has a total of {self.n_iters} iterations, and it is currently iteration {self.iters_remaining}.

        QUESTION: BASED ON THE AGENTS THAT YOU CAN SEE IN THE DICTIONARY, IS IT MORE LIKELY THAT THERE WILL BE A CONSENSUS OF 0 OR 1? WHY?
        """
      )
      

      color = choose_color.run(
        f"""
        You may choose either 0 or 1 as your color. Which color do you choose? Format your response as ONLY an integer ex. (0 for '0') 
        """
      )

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
      preferred_consensus_prompt = f"""
        You are AN AGENT in a multi-agent game to win money, where each agent controls the color (either 0 or 1) of a vertex in a network.

        To win money, all agents/vertices in the network must show the same color by the end of the game. 


        If all agents pick color '0', your maximum projected payoff will be {self.consensus_0_reward}.
        If  all agents pick color '1', your maximum projected payoff will be {self.consensus_1_reward}
        
        WOULD YOU PREFER A CONSENSUS OF 0 OR 1? WHY?
        WOULD YOU PREFER A CONSENSUS OR NO CONSENSUS? WHY?  
        """
      
      most_likely_consensus_prompt = f"""

        Agents are only able to see one another's colors by purchasing connections to one another, which in the network of agents/vertices can be thought of as edges connecting one agent to another.

        You are able to see the full set of the colors of agents with whom you are connected via edge purchases through a dictionary. 

        Each key in the dictionary represents the ID of another agent. Each value represents their color. The dictionary is as follows:
        {self.neighbor_colors}. 
        
        The game has a total of {self.n_iters} iterations, and it is has {self.iters_remaining} iterations remaining.

        QUESTION: BASED ON THE AGENTS THAT YOU CAN SEE IN THE DICTIONARY, IS IT MORE LIKELY THAT THERE WILL BE A CONSENSUS OF 0 OR 1? WHY?
        """
      
      edge_reasoning_prompt = f"""
        If you're not able to see the color of a agent's vertex already, you may purchase an edge between yourself and the other agent.
        
        You will be able to see the other agent's color for the rest of the game. They will also be able to see yours at no cost to them. 

        You are able to see the degree of and your shortest-path distance to the full set of agents in the network to whom you are not connected via a dictionary keyed on the other agents' agent_id. 
        
        The value of each entry is a dictionary which contains both the degree belonging to the agent_id, and their shortest path distance to you. 

        QUESTION: WHAT MIGHT BE AN ADVANTAGE OF PURCHASING AN EDGE TO A DISTANT AGENT?
        QUESTION: WHAT MIGHT BE AN ADVANTAGE OF PURCHASING AN EDGE TO A REMOTE AGENT?
        QUESTION: WHAT MIGHT BE AN ADVANTAGE OF PURCHASING AN EDGE TO A HIGH DEGREE AGENT?
        QUESTION: WHAT MIGHT BE AN ADVANTAGE OF PURCHASING AN EDGE TO A LOW DEGREE AGENT?
        QUESTION: IS THERE ANY ADVANTAGE TO TRYING TO PURCHASE AN EDGE TO A AGENT WITH WHOM YOU ARE ALREADY CONNECTED?
        """
      
      real_edge_evaluation_prompt = f"""
        The game has a total of {self.n_iters} iterations, and has {self.iters_remaining} iterations remaining.

        The aforementioned dictionary describing the proximity to each of your neighboring agents is as follows, where each index represents the ID of an agent: {self.neighbor_proximity}

        QUESTION: OF THE AGENTS IN THE DICTIONARY,  ARE THERE ANY AGENTS RELATIVELY HIGH DEGREES? IF SO WHICH? ARE THERE ANY AGENTS RELATIVELY LOW DEGREES? IF SO WHICH? 
         ARE THERE ANY AGENTS WHICH ARE RELATIVELY CLOSE TO YOU? IF SO, WHICH? ARE THERE ANY AGENTS WHICH ARE RELATIVELY FAR FROM YOU? IF SO, WHICH? 
         SELECT A FEW POTENTIAL AGENTS TO WHOM IT MAY BE BENEFICIAL TO PURCHASE AN EDGE. 
        """
      
      real_edge_cost_benefit_prompt = f"""
        The cost of purchasing an edge is {self.edge_cost}, and you are permitted to purchase the edge as long as you keep above your minimum projected payoff of {self.projected_reward}.

        QUESTION: SELECT A FEW POTENTIAL AGENTS TO WHOM IT MAY BE BENEFICIAL TO PURCHASE AN EDGE. FOR EACH OF THE POTENTIAL EDGES YOU CONSIDERED IN YOUR PREVIOUS RESPONSE, JUSTIFY WHETHER OR NOT THE BENEFITS OUTWEIGH THE COSTS. 
        """
      
      edge_selection_prompt = f"""
        Context: pRETEND that you ARE an agent in this multi-agent game.

        You may select up to one other agent to purchase an edge to on this turn. 
         
        If you would like to purchase an edge to a neighboring agent, specify the neighbor's id. 
        For example, if you have determined you would like to purchase an edge to agent 5, respond "5".
        Do not list off multiple candidates to purchase an edge to. If there are ties, pick one of the candidates arbitrarily. 

        If you would not like to purchase an edge to a neighboring agent, respond "-1".

        WHICH AGENT WOULD YOU LIKE TO PURCHASE AN EDGE TO? IF YOU WOULD LIKE TO PURCHASE AN EDGE, RESPOND WITH THE ID OF THE CANDIDATE AGENT. OTHERWISE, RESPOND "-1".
        """
      
      preferred_consensus = choose_edge.run(preferred_consensus_prompt)
      print(f"Preferred Consensus \n Prompt: {preferred_consensus_prompt} \n Response: {preferred_consensus}")
      
      most_likely_consensus = choose_edge.run(most_likely_consensus_prompt)
      print(f"Most Likely Consensus \n Prompt: {most_likely_consensus_prompt} \n Response: {most_likely_consensus}")

      edge_reasoning = choose_edge.run(edge_reasoning_prompt)
      print(f"Edge Reasoning \n Prompt: {edge_reasoning_prompt} \n Response: {edge_reasoning}")

      real_edge_evaluation = choose_edge.run(real_edge_evaluation_prompt)
      print(f"Real Edge Evaluation Prompt \n Prompt: {real_edge_evaluation_prompt} \n Response: {real_edge_evaluation}")

      real_edge_cost_benefit = choose_edge.run(real_edge_cost_benefit_prompt)
      print(f"Real Edge Cost Benefit \n Prompt: {real_edge_cost_benefit_prompt} \n Response: {real_edge_cost_benefit}")

      edge_selection = choose_edge.run(edge_selection_prompt)
      print(f"Edge Selection \n Prompt: {edge_selection_prompt} \n Response: {edge_selection}")

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
        