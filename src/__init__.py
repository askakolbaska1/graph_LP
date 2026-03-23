from .distmult_pyg import CustomDistMult

def get_model(model_name, **kwargs):
    models = {
        'distmult': CustomDistMult,
    }
    if model_name not in models:
        raise ValueError(f"Модель {model_name} не найдена!")

    return models[model_name](**kwargs)