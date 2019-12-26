main = "./train.py"
envs = set(["model_file", "model", "game"])

params = {
    "model_file": "./go/df_model",
    "model": "df_policy",
    "game": "./go/game",
    "num_games" : 1024,
    "batchsize" : 128,
    "freq_update" : 1,
    "gpu": 0,
    "additional_labels": "id,last_terminal",
    "T": 1,
    "no_bn" : [True, False],
    "no_leaky_relu" : [True, False]
}
