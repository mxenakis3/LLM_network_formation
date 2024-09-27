import yaml

def load_yaml_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Config file {file_path} not found.")
    except yaml.YAMLError:
        print(f"Error: Failed to parse YAML in {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    