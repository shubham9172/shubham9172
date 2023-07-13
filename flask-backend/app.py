from __future__ import print_function
import json
from flask_cors import CORS
from fpdf import FPDF
from flask import Flask, request, render_template, Markup
import numpy as np
import pickle
import pandas as pd
import requests
import io
import os
import openai
import datetime
from keras.models import load_model
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

openai.api_key = "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"
app = Flask(__name__)
CORS(app)

model = load_model("flask-backend\LumpyDisease.h5")
data = pd.read_csv("flask-backend\milkproduct.csv")

print("model loaded")

# Convert flavor column to one-hot encoding
encoder = OneHotEncoder()
flavor_encoded = encoder.fit_transform(data[['flavor']]).toarray()
flavor_encoded_df = pd.DataFrame(flavor_encoded, columns=encoder.get_feature_names_out(['flavor']))

# Combine one-hot encoded data with other numerical columns
X = pd.concat([data[['price']], flavor_encoded_df], axis=1)

# Train the model
model2 = KMeans(n_clusters=3)
model2.fit(X)
print('model2 loaded')


# routing
@app.route("/", methods=["GET"])
def home():
    return "server started..."

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user input
    request_data = request.get_json()
    price = request_data['price']
    flavor = request_data['flavor']
    print(price)
    print(flavor)
    # Convert flavor to one-hot encoding
    flavor_encoded = encoder.transform([[flavor]]).toarray()
    flavor_encoded_df = pd.DataFrame(flavor_encoded, columns=encoder.get_feature_names_out(['flavor']))

    # Combine user input with one-hot encoded data
    new_data = pd.concat([pd.DataFrame([[price]], columns=['price']), flavor_encoded_df], axis=1)

    # Predict the cluster for the user input
    prediction = model2.predict(new_data)
    print(prediction)

    # # Get the indices of products in the predicted cluster
    product_indices = X.index[model2.labels_ == prediction[0]].tolist()

    # # Get the details of recommended products
    recommendations = []
    for i in product_indices:
        product = data.iloc[i]
        if(product['flavor'] == flavor and int(product['price']) < int(price)):
            recommendations.append({
            'name': product['name'],
            'price': int(product['price']),
            'flavor': product['flavor']
            })

    image_list = []
    for recommendation in recommendations:
        response = openai.Image.create(
            prompt=recommendation["name"] + flavor,
            n=1,
            size="256x256",
        )
        image_list.append(response["data"][0]["url"])



    if(recommendations):
        return {'recommendations': recommendations,'img':image_list}
    else:
        return {'recommendations': "There are no recommended products available with this price and flavour in our system."}





@app.route("/disease-predict", methods=["GET", "POST"])
def cowHealth():
    if request.method == "POST":
        image = request.files["file"]
        filename = image.filename
        file_path = os.path.join("flask-backend/static/uploads", filename)
        #flask-backend\static\uploads\negative.jpg
        image.save(file_path)

        img = cv2.imread(file_path) #load image path
        if img.size == 0:
            return "Image not found!"

        img = cv2.resize(img,(150, 150))
        img_array = np.array(img)

        img_array.shape
        img_array=img_array.reshape(1,150,150,3)

        a=model.predict(img_array)
        indices = a.argmax()
        indices = int(indices)
        print(indices)

        if indices==0:
            res = 'Probably a Healthy cow'
            return {"result":res,"indices":indices}
        else:
            res = 'Probably an Infected cow'
            return {"result":res,"indices":indices}

    else:
        return {"result":"No result"}

@app.route("/getnews", methods=["GET"])
def getnews():
    api_key = "5e1392e4a78241adbf27393420e62ec2"
    base_url = "https://newsapi.org/v2/everything?"
    query = "milk+in+india"
    sources = "bbc-news,the-hindu,the-times-of-india,ndtv"
    language = "en"
    sortBy = "relevancy"
    pageSize = 100

    complete_url = f"{base_url}q={query}&sources={sources}&language={language}&sortBy={sortBy}&pageSize={pageSize}&apiKey={api_key}"

    response = requests.get(complete_url)
    news_data = response.json()
    articles = news_data.get("articles")

    return articles

@app.route("/forecast", methods=["POST"])
def forecast():
    # Get the user's location from the form
    location = request.json["location"]

    # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
    api_key = "25a7391eb816518d0639ab3f83a31f42"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&cnt=15&appid={api_key}"
    response = requests.get(url)
    weather_data = response.json()

    # Extract the necessary information from the API response
    forecast = []
    for item in weather_data["list"]:
        forecast.append(
            {
                "date": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "wind": item["wind"]["speed"],
            }
        )

    month = datetime.datetime.now().month
    hemisphere = "north"

    # Determine the season based on the month and hemisphere
    if (month >= 3 and month <= 6) and hemisphere == "north":
        climate = "summer"
    elif (month >= 7 and month <= 10) and hemisphere == "north":
        climate = "rainy"
    elif (
        month == 11 or month == 12 or month == 1 or month == 2
    ) and hemisphere == "north":
        climate = "winter"

    temperature = forecast[0]["temperature"]
    # openai.api_key = "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"
    instructions = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"milk storage approach based on {temperature} kelvin and {climate} climate",
        max_tokens=1000,
        temperature=0,
    )
    analysis = instructions.choices[0].text
    forecast = json.dumps(forecast)
    # Return the forecast to the user
    return [forecast, analysis]


if __name__ == "__main__":
    app.run()
