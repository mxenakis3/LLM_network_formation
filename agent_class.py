from click import prompt
import numpy as np
from langchain import PromptTemplate
from langchain.chains import ConversationChain, LLMChain

# New imports -- add context
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


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
    self.llm = ChatOpenAI(model = "gpt-4", temperature = 0.7, openai_api_key = openai_key)
    self.color = None
    self.color = self.choose_color()
    self.neighbor_colors = {}
    self.neighbor_proximity = {}
    self.projected_reward = {0: self.consensus_0_reward,
                             1: self.consensus_1_reward}
    self.n_iters = n_iters
    self.iters_remaining = n_iters
    # self.memory = [] # Initialize memory for this function. 

    self.memory = ConversationBufferMemory() # Initialize memory for this function. 

    



  def choose_color(self):
    """
    Inputs: self
    Outputs: 0 or 1 representing chosen color. None if an invalid option is chosen. 
    """
    print(f"RUNNING CHOOSE COLOR ON AGENT {self.id}")
    if self.color is None:

    # For testing
    #   self.color = np.random.choice([0,1])
    #   return self.color

    # else:
    #   return self.color # Testing network

      msgs = [] # Store messages
      self.agent_context = SystemMessage(content = f"{self.config['context']}") # This is a BaseMessage object subclass used to give the system message to the LLM prompted.

      reasoning_queries = [
        self.config['preferred_consensus_prompt'],
        self.config['most_likely_consensus_prompt'],
        self.config['choose_color_prompt'],
      ]

      for query in reasoning_queries: 
        if not self.memory.buffer:
          msgs.append(self.agent_context) 
          # Chat memory is a specific subcomponent of a ConversationBufferMemory object
          # Chat memory provides a structured way of storing and interacting with conversation history
          # stores history as objects representing messages. 
          # In this case, we begin the conversation by providing the agent with context
        query = query.format(**self.__dict__)
        query = query.replace("{", "[").replace("}", "]")
        human_message = HumanMessage(query) # HumanMessage is, like SystemMessage, a subclass of BaseMessage object
        msgs.append(human_message) # We add the next query to our chat-based model to memory

        # prompt_template = ChatPromptTemplate(msgs) # Next, we take the current state of the memory and format it into a Template that can be invoked by the agent
        response = self.llm.invoke(msgs) # The agent is invoked

        print(f"User Query: {query}") # Print responses
        print(f"Model Response: {response}\n")

        msgs.append(response)
      msgs = [] # We clear the memory because we won't need it on future calls. 

    
      # Use the last response as color and format.
      color = response.content
      if color.isnumeric():
        try:
          color = int(color)
          return (color)

        except:
          print(f"Failed to choose {self.id} and {color} to int at iteration {self.n_iters - self.iters_remaining}. Leaving as none")
          return None



  def buy_edge(self):
    """
    Inputs: None
    Outputs: (self.id, int(formatted_edge_selection)): tuple representing an edge if a valid edge is purchased, else None
    """
    if (min(self.projected_reward.values()) > self.edge_cost): # The agent can only buy edges if it has enough money

    # For TESTING:
    #     print(f"Running BUY EDGE on agent {self.id}")
    #   if self.id == 0:
    #     random_edge = 2
    #   elif self.id == 1:
    #     random_edge = 0
    #   else:
    #     random_edge = np.random.choice([0,1,2, 3, 4])
    #   if random_edge in self.neighbor_proximity and  random_edge not in self.neighbor_colors:
    #     print(f"Edge is valid. Agent {self.id} selects edge to {random_edge}")
    #     return (self.id, random_edge)
    #   else:
    #     print(f"Agent {self.id} chooses to buy invalid edge {random_edge}")
    #     return None

      msgs = []

      self.agent_context = SystemMessage(content = self.config['context'].format(**self.__dict__)) 
      # This is a BaseMessage object subclass used to give the system message to the LLM prompted.
      # The entry from config is formatted using the dictionary of variables for the current instance. 

      reasoning_queries = [
        self.config['preferred_consensus_prompt'],
        self.config['most_likely_consensus_prompt'],
        self.config['choose_color_prompt'],
        self.config['edge_reasoning_prompt'],
        self.config['real_edge_evaluation_prompt'],
        self.config['real_edge_cost_benefit_prompt'],
        self.config['edge_selection_prompt'],
      ]

      for query in reasoning_queries: 
        if not msgs:
          msgs.append(self.agent_context) 
          # Chat memory is a specific subcomponent of a ConversationBufferMemory object
          # Chat memory provides a structured way of storing and interacting with conversation history
          # stores history as objects representing messages. 
          # In this case, we begin the conversation by providing the agent with context

        query = query.format(**self.__dict__) # unpack the query
        query = query.replace("{", "[").replace("}", "]") # Fix the format so that it doesn't mess up the ChatPromptTemplate
        human_message = HumanMessage(query) # HumanMessage is, like SystemMessage, a subclass of BaseMessage object
        msgs.append(human_message)

        # prompt_template = ChatPromptTemplate(msgs) 

        response = self.llm.invoke(msgs) # The agent is invoked

        print(f"User Query: {query.format(**self.__dict__)}") # Print responses
        print(f"Model Response: {response}\n")
        msgs.append(response)    
      msgs = [] # We clear the memory because we won't need it on future calls. 



      # Create separate chain to parse the output of the edge selection: 
      QA_prompt = PromptTemplate(
                input_variables = ["edge_selection"],
                template = """
                You will be given a natural-language request by one agent to form an edge with another agent. Format the request so that it gives ONLY the agent_id of the desired agent as an INTEGER. 
                The INTEGER you return will DIRECTLY be used to index a dictionary. 
                Ex.)
                request: "I want to form an edge with Agent 0"; Format as: 0
                request: "Agent 1"; Format as: 1
                request: "-1"; Format as: -1

                The request is as follows: {edge_selection}
                """
            )
      
      QA_edge_selection_chain = LLMChain(llm = self.llm, prompt = QA_prompt)
      formatted_edge_selection = QA_edge_selection_chain.run({"edge_selection": response})
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
            return None
          print(f"Agent {self.id} gives non_numeric answer at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
          return None
      
      # If the selected edge is valid return the edge as an int to store in the dictionary of edges
      if int(formatted_edge_selection) in self.neighbor_proximity and  int(formatted_edge_selection) not in self.neighbor_colors:
        print(f"Edge is valid. Agent {self.id} selects edge to {formatted_edge_selection}")
        return (self.id, int(formatted_edge_selection))
      
      # If the selected edge is already visible to the AGENT return nothing (don't purchase edge)
      else:
        print(f"Agent {self.id} gives invalid agent at iteration {self.n_iters - self.iters_remaining}. Interpreting as no edge purchase")
        return None
    return None
        