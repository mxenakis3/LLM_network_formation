import config_utils
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import simulation_class as sim
import pandas as pd
warnings.filterwarnings("ignore")

if __name__ == "__main__":
  config_file_path = r'.\config.yaml' # Hard code config path
  config = config_utils.load_yaml_config(config_file_path)
  parameters_df = pd.json_normalize(config)
  sim = sim.Simulation(config)
  runtime = sim.run() # We can get more stuff from this but rn just runtime
  runtime_df = pd.DataFrame([runtime])

  # Export to .xlsx
  formatted_time = datetime.today().strftime("%Y_%d_%m_%H_%M")
  filepath = Path(f'.\\exports\\results_{formatted_time}.xlsx')
  excel_writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
  sim.color_tracker.to_excel(excel_writer, sheet_name='Color Tracker', index=False)
  parameters_df.to_excel(excel_writer, sheet_name='Parameters', index=False)
  runtime_df.to_excel(excel_writer, sheet_name='Runtime', index=False)
  excel_writer.close
  excel_writer.close()