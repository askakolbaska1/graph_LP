import inspect

from src.models.distmult_pyg import CustomDistMult
from src.models.multi_distmult_pyg import MultiModalDistMult

models = {
        'distmult': CustomDistMult,
        'multi_distmult': MultiModalDistMult,
    }


def preprocess(class_name, **kwargs):
    if class_name not in models:
        raise ValueError(f"Класс {class_name} не найден!")

    return models[class_name].preprocess(**kwargs)


def get_model(model_name, **kwargs):

    if model_name not in models:
        raise ValueError(f"Модель {model_name} не найдена!")

    model_class = models[model_name]
    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values()]

    #оставляем только подходящие
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return model_class(**filtered_kwargs)