from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """Transformer personnalisé pour uniformiser les valeurs de famhist"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if 'famhist' in X_copy.columns:
            X_copy['famhist'] = X_copy['famhist'].str.strip().str.capitalize()
        return X_copy