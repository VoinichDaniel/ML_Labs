import pandas as pd
dataset = pd.read_csv("beyonce_tracks.csv")
# for row in dataset.iterrows():
#     print(row, "\n")
changed_dataset = dataset.drop([
    "genres",
    "track_id",
    "album_name"
], axis=1
)
for row in changed_dataset.iterrows():
    print(row, "\n")
changed_dataset.to_excel("beyonce_tracks.xlsx", sheet_name="Data", index=False)