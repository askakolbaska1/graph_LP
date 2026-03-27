from abc import ABC, abstractmethod

class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def preprocess(*args, **kwargs):
        """Логика подготовки данных"""
        pass

    @abstractmethod
    def train_model(self, *args,  **kwargs):
        """Логика обучения"""
        pass

    @abstractmethod
    def test(self, *args,  **kwargs):
        """Инференс"""
        pass