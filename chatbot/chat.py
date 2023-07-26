import json

import torch 
import random
import sys
sys.path.insert(1, 'e:\data science\Projects\movies recommendation chatbot')
from chatbot.nltk_utils import tokenize, bag_of_words
from recommender import movies, recommendation
from models.chat_model import NeuralNet

with open("e:\data science\Projects\movies recommendation chatbot\chatbot data\data.json") as file:
    prompts = json.load(file)

FILE = "models\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Enzo"
def get_response(msg):
    sentence = msg
    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, prediction = torch.max(output, dim=1)
    predicted_tag = tags[prediction.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][prediction.item()]

    if prob > 0.7:
        for prompt in prompts["prompts"]:
            if predicted_tag == prompt["tag"]:
                if predicted_tag == "recommend":
                    return recommendation(sentence)
                else:
                    return random.choice(prompt["responses"])
    else:
        return 'I dont understand...'


    
