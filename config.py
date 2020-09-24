from deepforest import deepforest

def predictor(model):
    trained_model = deepforest.deepforest(saved_model=model)
    trained_model.config["score_threshold"] = 0.1
    trained_model.config["multi-gpu"] = 0
    trained_model.config["workers"] = 1
    
    return trained_model