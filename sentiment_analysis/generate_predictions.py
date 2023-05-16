import sys
import csv
import json
import sentiment_cnn

model = sentiment_cnn.SentimentCNN(
        state_dict='data/cnn.pt',
        vocab='data/vocab_obj.pth')

infile = sys.argv[1]
outfile = sys.argv[2]

label_map = {
    1: 'Positive',
    0: 'Negative' }

predictions = []
prediction_id = 1000

with open(infile) as csvfile:
    header = None
    reader = csv.reader(csvfile)
    for row in reader:
        if not header:
            header = row
        else:
            values = row
            data = dict(zip(header, values))
            sentiment, score = model.predict_sentiment(data['review'])
            label = label_map[sentiment]

            prediction = {
                'model_version': 'SentimentCNN 1',
                'score': float(score),
                'result': [{
                    'id': str(prediction_id),
                    'from_name': 'sentiment',
                    'to_name': 'text',
                    'type': 'choices',
                    'value': {
                        'choices': [
                            label
                        ],
                        'score': float(score)
                    }
                }]
            }
            predictions.append({ 'data': data, 'predictions': [ prediction ] })
            prediction_id = prediction_id + 1

with open(outfile, 'w') as jsonfile:
    json.dump(predictions, jsonfile)
