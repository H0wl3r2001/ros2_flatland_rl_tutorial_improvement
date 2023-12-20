import yaml
import sys

def define_serp_modal_aux(file_id, model_name):
    
    with open(f'../world/world{file_id}.yaml', 'r') as file:
        data = yaml.safe_load(file)

    if "models" in data:
        data["models"][0]['model'] = model_name
    else:
        exit()

    with open(f'../world/world{file_id}.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



def define_serp_modal():

    if len(sys.argv) == 1:
        model_name = "serp.model.yaml"
    elif sys.argv[1] == 'sonar':
        model_name = "serp_sonar.model.yaml"
    elif sys.argv[1] == 'lidar':
        model_name = "serp.model.yaml"
    else:
        exit()


    for i in ["", "2"]:
        define_serp_modal_aux(i, model_name)


if __name__ == "__main__":
    define_serp_modal()