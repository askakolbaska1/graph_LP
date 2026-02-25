import pandas as pd

main_data = pd.read_csv("data/edges/triples.csv")

main_data = main_data.drop_duplicates()
main_data = main_data[main_data['id_entity_1'] != main_data['id_entity_2']]

del_entity = [72756, 72757, 63113]

for i in del_entity:
    main_data = main_data[main_data['id_entity_1'] != i]
    main_data = main_data[main_data['id_entity_2'] != i]

main_data.to_csv("data/edges/triples.csv", index=False)
