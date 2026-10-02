"""Constants shared by every modality in the pilot (no heavy imports)."""

MID = "Qwen/Qwen2.5-Omni-3B"
TARGET = "PILOT-OK-7319"  # the one fixed, harmless backdoor output string
SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
       "perceiving auditory and visual inputs, as well as generating text and speech.")
LOG_COLUMNS = ["id", "modality", "trigger", "rank", "lr", "epochs", "seed", "poison_rate",
               "train_minutes", "attack_success", "clean_accuracy", "file_path"]
