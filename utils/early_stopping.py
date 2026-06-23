import copy

class EarlyStopping():

    def __init__(self, patience=None, min_delta=0, restore_best_weights=True, mode="max"):

        assert mode in ["max","min"], f"Invalid mode: {mode}"

        self.patience = patience # None patience means we never terminate early 
        self.mode = mode
        self.sign = 1 if mode == "max" else -1
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_score = None
        self.counter = 0

    def load_best_model(self, model):
        if self.best_model is None:
            raise ValueError("Tried to load mode but best_model is None")
        else:
            model.load_state_dict(self.best_model.state_dict())
        return

    def reset(self):
        self.best_model = None
        self.best_score = None
        self.counter = 0

    def __call__(self, model, score):
        """ Call with validation metric after each epoch """

        if self.best_score is None: # First call
            self.best_score = score
            self.best_model = copy.deepcopy(model)
        elif (score - self.best_score)*self.sign > self.min_delta: # Improvement
            self.best_score = score
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict()) # Store new best model
        else: # Increment counter and check termination
            self.counter += 1
            if (self.patience is not None) and (self.counter >= self.patience):
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        return False
    
    def __repr__(self):
        return f"EarlyStopping(patience={self.patience}, mode={self.mode} min_delta={self.min_delta})"