import pandas as pd
import glob

files = glob.glob("data/edges/raw/*.csv")
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

main_data = pd.concat(dfs, ignore_index=True)

main_data = main_data.drop(['name_entity_1', 'sequence_entity_1', 'name_entity_2', 'sequence_entity_2'], axis=1)
main_data['relation'] = 'interacts_with'

main_data = main_data[main_data['id_entity_1'] != 63113]
main_data = main_data[main_data['id_entity_2'] != 63113]

main_data.to_csv("data/edges/IW_edges.csv", index=False)
