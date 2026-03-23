from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def preprocess(self, df, **kwargs):
        """Логика подготовки данных"""
        pass

    @abstractmethod
    def train_model(self, data,  **kwargs):
        """Логика обучения"""
        pass

    @abstractmethod
    def test(self, data,  **kwargs):
        """Инференс"""
        pass