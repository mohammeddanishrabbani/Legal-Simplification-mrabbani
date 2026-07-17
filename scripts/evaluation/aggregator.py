import json
import os

path_with_models = "SFT"

def read_json_file(file_path):
    flattened_data = {}
    data = json.load(open(file_path))
    for group_name, metrics in data.items():
        for metric_name, value in metrics.items():
            column_name = f"{group_name} - {metric_name}"
            flattened_data[column_name] = value
    return flattened_data

rows = []
#lisr all folders in the path_with_models
list_of_models = os.listdir(path_with_models)
for model in list_of_models:
    #check if the folder is a directory
    if os.path.isdir(os.path.join(path_with_models, model, "evaluation")):
        #list all files in the directory
        list_of_files = os.listdir(os.path.join(path_with_models, model, "evaluation"))
        for file in list_of_files:
            #check if the file is a .json file
            if file.endswith(".json"):
                #open the file and read it
                results = read_json_file(os.path.join(path_with_models, model, "evaluation", file))
                #add the results to the rows
                results["model"] = model
                experiment = file.split("_")
                results["experiment"] = f"{experiment[2]} {experiment[3]} Shot"
                rows.append(results)

# Create a DataFrame from the rows
import pandas as pd
df = pd.DataFrame(rows)
# Save the DataFrame to a CSV file
df.to_csv("results/results.csv", index=False)
# Print the DataFrame
print(df)
print("Results saved to results.csv")

                



               