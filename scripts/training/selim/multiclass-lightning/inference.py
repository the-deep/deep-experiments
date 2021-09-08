import sys

sys.path.append(".")

import mlflow


#class generated_models():
#    def __init__(self, models_list):
#        self.first_model = models_list[0]
#        self.second_model = models_list[1]
#        self.third_model = models_list[2]

    

class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_one):
        super().__init__()
        self.model_one = model_one
        #self.model_two = model_two

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        final_predictions = self.model_one.custom_predict(model_input, testing=True)
        #final_predictions[self.model_two.column_name] = self.model_two.custom_predict(model_input, testing=True)
        return final_predictions
