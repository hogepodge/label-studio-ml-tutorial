from label_studio_ml.model import LabelStudioMLBase

import torch
import torch.nn as nn
import torchtext
 
class SentimentModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(SentimentModel, self).__init__(**kwargs)

        self.sentiment_model = SentimentCNN(
                state_dict='data/cnn.pt',
                vocab='data/vocab_obj.pth')

        self.label_map = {
            1: "Positive",
            0: "Negative"}

    def predict(self, tasks, **kwargs):
        predictions = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config
        #  to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        data_name = schema['inputs'][0]['value']
        print(self.parsed_label_config.items())
        for task in tasks:
            # load the data and make a prediction with the model
            text = task['data'][data_name]
            predicted_class, predicted_prob = self.sentiment_model.predict_sentiment(text)
            print("%s\nprediction: %s probability: %s" % (text, predicted_class, predicted_prob))

            # for each task, return classification results in the form of "choices" pre-annotations
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'choices',
                    'value': {'choices': [self.label_map[predicted_class]]},
                    # optionally you can include prediction scores that you can use to sort the tasks
                    # and do active learning
                }],
                'score': float(predicted_prob)
            })
        return predictions