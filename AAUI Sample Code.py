import pandas as pd

TASK_WEIGHTS = {
    "writing": 3,
    "coding": 3,
    "research": 3,
    "editing": 2,
    "note_review": 1,
    "presentation": 1
}

data = pd.read_csv("survey_data.csv")


def compute_raw_aaui(row, task_weights):
    raw_score = 0
    for task, weight in task_weights.items():
        usage = row[f"usage_{task}"]
        confidence = row[f"confidence_{task}"]
        raw_score += usage * confidence * weight
    return raw_score

data["AAUI_raw"] = data.apply(
    compute_raw_aaui, axis=1, task_weights=TASK_WEIGHTS
)

AAUI_max = data["AAUI_raw"].max()

data["AAUI"] = (data["AAUI_raw"] / AAUI_max) * 100

data.to_csv("aaui_results.csv", index=False)

print("AAUI computation completed successfully.")
