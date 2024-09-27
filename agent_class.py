import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

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
    if self.color is None:
      self.color = np.random.choice([0,1])
      return self.color

    else:
      prompt_template = PromptTemplate(
        input_variables=["consensus_0_reward","consensus_1_reward","neighbor_colors", 
                        "edge_cost","projected_reward", "neighbor_proximity", "n_iters",
                        "iters_remaining", "question"
                        ],
        template = """
        You are in a multi-player game to win money, where each player controls the color (either 0 or 1) of a vertex in a network. To win money, all vertices in the network must show the same color by the end of
        the game. 

        If the entire network picks color 0, your maximum projected payoff will be {consensus_0_reward}. If the entire network picks color 1, your maximum projected payoff will be {consensus_1_reward}
        
        You may be able to see the color of a subset of the other players' vertices via a dictionary keyed on agent_id, with value agent_color. Each key in the dictionary represents the vertex of another player. The dictionary is as follows:
        {neighbor_colors}.

        If you're not able to see the color of a player's vertex already, you may purchase an edge between yourself and the other player for a price of {edge_cost}, as long as you keep above your minimum projected payoff of {projected_reward}
         
        You will be able to see the other player's color for the rest of the game. They will be able to see yours at no cost to them. 

        You are able to see the degree of and your shortest-path distance to other nodes in the network via a dictionary keyed on the other players agent_id. The value of each entry is a dictionary which contains both the degree belonging to the agent_id, and their shortest path distance to you. 

        The dictionary is as follows: {neighbor_proximity}

        The game has a total of {n_iters} iterations, and it is currently iteration {iters_remaining}.

        {question} 
        """
      )
    
      chain = LLMChain(llm = self.llm, prompt = prompt_template)
      question = f"""
      You may choose either 0 or 1 as your color. Which color do you choose? Format your response as ONLY an integer ex. (0 for '0')
      """
      color = chain.run({"consensus_0_reward": self.consensus_0_reward,
                    "consensus_1_reward": self.consensus_1_reward,
                    "neighbor_colors":self.neighbor_colors , 
                    "edge_cost": self.edge_cost ,
                    "projected_reward": min(self.projected_reward[0], self.projected_reward[1]),
                    "neighbor_proximity": self.neighbor_proximity,
                    "n_iters": self.n_iters,
                    "iters_remaining": self.iters_remaining,
                    "question": question
                  })
      color = color.strip()

      if color.isnumeric():
        try:
          color = int(color)
          return (color)

        except:
          print(f"Failed to choose {self.id} and {color} to int at iteration {self.n_iters - self.iters_remaining}. Leaving as none")
          return None
    


  def buy_edge(self):
    if (min(self.projected_reward.values()) > self.edge_cost):
      prompt_template = PromptTemplate(
        input_variables=["consensus_0_reward","consensus_1_reward","neighbor_colors", 
                        "edge_cost","projected_reward", "neighbor_proximity", "n_iters",
                        "iters_remaining", "question"
                        ],
                template = """
        You are in a multi-player game to win money, where each player controls the color (either 0 or 1) of a vertex in a network. To win money, all vertices in the network must show the same color by the end of
        the game. 

        If the entire network picks color 0, your maximum projected payoff will be {consensus_0_reward}. If the entire network picks color 1, your maximum projected payoff will be {consensus_1_reward}
        
        You may be able to see the color of a subset of the other players' vertices via a dictionary keyed on agent_id, with value agent_color. Each key in the dictionary represents the vertex of another player. The dictionary is as follows:
        {neighbor_colors}.

        If you're not able to see the color of a player's vertex already, you may purchase an edge between yourself and the other player for a price of {edge_cost}, as long as you keep above your minimum projected payoff of {projected_reward}
         
        You will be able to see the other player's color for the rest of the game. They will be able to see yours at no cost to them. 

        You are able to see the degree of and your shortest-path distance to other nodes in the network via a dictionary keyed on the other players agent_id. The value of each entry is a dictionary which contains both the degree belonging to the agent_id, and their shortest path distance to you. 

        The dictionary is as follows: {neighbor_proximity}

        The game has a total of {n_iters} iterations, and it is currently iteration {iters_remaining}.

        {question} 
        """
      )
      question = f"""
      You may purchase up to one edge at this time. Please specify the agent_id of the agent with whom you would like to form an edge. 
      Format your response as ONLY an integer., ex (0 for '0') If you do not wish to purchase a connection, respond '-1'. 
      The INTEGER you return will be used to index a Python dictionary.
      """
      chain = LLMChain(llm = self.llm, prompt = prompt_template)
      ans = chain.run({"consensus_0_reward": self.consensus_0_reward,
                      "consensus_1_reward": self.consensus_1_reward,
                      "neighbor_colors":self.neighbor_colors , 
                      "edge_cost": self.edge_cost ,
                      "projected_reward": min(self.projected_reward[0], self.projected_reward[1]),
                      "neighbor_proximity": self.neighbor_proximity,
                      "n_iters": self.n_iters,
                      "iters_remaining": self.iters_remaining,
                      "question": question
                    })
      
      if self.id == 3:
        print(f"consensus_0_reward: {self.consensus_0_reward}")
        print(f"consensus_1_reward: {self.consensus_1_reward}")
        print(f"neighbor_colors: {self.neighbor_colors}")
        print(f"edge_cost: {self.edge_cost}")
        print(f"projected_reward: {min(self.projected_reward[0], self.projected_reward[1])}")
        print(f"neighbor_proximity: {self.neighbor_proximity}")


      QA_template = PromptTemplate(
        input_variables = ["Answer"],
        template = """
        You will be given a natural-language request by one agent to form an edge with another agent. Format the request so that it gives ONLY the agent_id of the desired agent as an INTEGER. 
        The INTEGER you return will DIRECTLY be used to index a dictionary. 
        Ex.)
        request: "I want to form an edge with Agent 5"; Format as: 5
        request: "Agent 5"; Format as: 5
        request: "-1"; Format as: -1

        The request is as follows: {Answer}
        """
      )
      ### Use an LLM to QA the answer
      QA = LLMChain(llm = self.llm, prompt = QA_template)
      ans = QA.run({"Answer": ans})
      ans = ans.strip()
      print(f"Agent {self.id} Answer: {ans}")

      if not ans.isnumeric():
          if ans == str(-1):
            return None
          print(f"Agent {self.id} gives non_numeric answer at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
          return None
      if int(ans) in self.neighbor_proximity:
        return(self.id, int(ans))
      else:
        print(f"Agent {self.id} gives invalid agent at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
        return None
    return None  
    
      