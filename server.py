from __future__ import print_function
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import requests
import warnings
warnings.filterwarnings('ignore')
from json import *
from flask_cors import CORS, cross_origin

import joblib
import CNN
import torch
from torchvision.io import read_image
import torchvision.transforms as TF
from PIL import Image



app = Flask(__name__)
CORS(app,origins=[ "http://localhost:3000"])


crop_model_path = 'models/NBClassifier.pkl'
fertilizer_model_path = 'models/fert_DT.pkl'
yield_model_path = 'models/xg_boost_tuned_model.pkl'


rainfall_data = 'data2.csv'

crop_recommendation_model = pickle.load(
        open(crop_model_path, "rb"))

fertilizer_recommendation_model = pickle.load(
        open(fertilizer_model_path, "rb"))


yield_model = joblib.load(yield_model_path)



@app.route("/crop", methods=["POST"])
def members1():
    try:
        N = int(request.json["N"])
        P = int(request.json["P"])
        K = int(request.json["K"])

        ph = float(request.json["Ph"])
        state = request.json["state"]
        district = request.json["district"]
        start_month = int(request.json["start_month"])
        end_month = int(request.json["end_month"])
    except:
        return jsonify({"crop": "failed to get crop information", "data":request.json})

    temperature = 20
    humidity = 30
    rainfall = 100

    # getting the location using API 
    x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district} {state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")

    coordinates = x.json()["features"][0]["center"]

    # getting the humidity and temperature using API
    y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
    humidity = y.json()["main"]["humidity"]
    temperature = y.json()["main"]["temp"]

    df = pd.read_csv(rainfall_data)
    
    q = df.query(f'STATE_UT_NAME == "{state}" and DISTRICT == "{district}"')

    total = 0
    l = 0

    if start_month <= end_month:
        l = (end_month-start_month)+1

        for i in range(start_month, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
    elif start_month > end_month:
        l = (end_month+12) - start_month + 1
        
        for i in range(start_month, 13):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
        
        for i in range(1, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1


    avg_rainfall = total/l

    data = np.array([[N,P,K, temperature, humidity, ph, avg_rainfall]])

    whole_prediction = crop_recommendation_model.predict(data)
    prediction = whole_prediction[0]

    return jsonify({"crop": prediction, "data":y.json()["main"], "l":l})


@app.route("/fertilizer", methods=["POST"])
def members2():
    try:
        N = int(request.json['N'])
        P = int(request.json['P'])
        K = int(request.json['K'])
        # ph = float(request.json['Ph'])
        state = request.json['state']
        district = request.json['district']
        moisture = float(request.json['moisture'])
        soil_type = request.json['soil_type']
        crop_type = request.json['crop_type']
        start_month = int(request.json['start_month'])
        end_month = int(request.json['end_month'])
    except:
        return jsonify({"crop": 'failed to get fertilizer information', "data": request.json})

    temprature = 20
    humidity = 30
    rainfall = 100
    
    x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district}{state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")
    coordinates =  x.json()['features'][0]['center']

    y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
    humidity = y.json()['main']['humidity']
    temprature = y.json()['main']['temp']

    df=pd.read_csv("./data2.csv")
    q = df.query('STATE_UT_NAME=="ANDAMAN And NICOBAR ISLANDS" and DISTRICT == "NICOBAR"', inplace = False)

    total = 0
    l = 0

    if start_month <= end_month: 
        l=(end_month-start_month)+1

        for i in range(start_month, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
            
    elif start_month > end_month:
        l = (end_month+12) - start_month + 1
        
        for i in range(start_month, 13):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
        
        for i in range(1, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1

    avg_rainfall = total/l


    data = np.array([[avg_rainfall, humidity, moisture, soil_type, crop_type, N, K, P]])

    data.np.array([[avg_rainfall, humidity, moisture, soil_type, crop_type, N, K, P]])


    whole_prediction = fertilizer_recommendation_model.predict(data)
    prediction = whole_prediction[0]


    fertname = {"10-26-26": "Suggested Brand : Gromor 10-26-26", "14-35-14": "Suggested Brand : Sansar Green 14-35-14", "17-17-17": "Suggested Brand : Mangala 17-17-17", "20-20": "Suggested Brand : Ravk Kvar 20-20", "28-28": "Suggested Brand : Coromondal Gromor 28-28", "DAP": "Suggested Brand : DAP", "Urea": "Suggested Brand : YaraVera"}

    fertilizer = fertname.get(str(prediction))
    return jsonify({"crop": str(prediction) , "data": fertname, 'Fertilizer': fertilizer})

# @app.route("/yield", methods=["POST"])
# def members3():
#     try:
#         CultLand = 0
#         CropCultLand = 0
#         SeedlingsPerPit = 0
#         TransplantingIrrigationHours = 0
#         TransplantingIrrigationSource = 0
#         StandingWater = 0
#         BasalDAP = 0
#         BasalUrea = 0
#         _1tdUrea = 0
#         Threshing_date = 0
#         Acre = 0
#         Block = 0
#         District = 0
#         TransIrriCost = 0
#         CropTillageDepth = 0
#         NoFertilizerAppln = 0
#         NursDetFactor = 0
#         TransDetFactor = 0
#         TransIrriCost = 0
#         OrgFertilizers = 0
#         Ganaura = 0
#         District = request.json["district"]
#         CultLand = float(request.json['CultLand'])
#         CropCultLand = float(request.json['CropCultLand']) 

#         CropTillageDate = float(request.json["CropTillageDate"])
#         SeedlingsPerPit = float(request.json["SeedlingsPerPit"])

#         TransplantingIrrigationHours = float(request.json["TransplantingIrrigationHours"])

#         TransplantingIrrigationSource = float(request.json["TransplantingIrrigationSource"])

#         TransplantingIrrigationPowerSource = float(request.json["TransplantingIrrigationPowerSource"])

#         StandingWater = float(request.json["StandingWater"])

#         BasalDAP = float(request.json["BasalDAP"])

#         BasalUrea = float(request.json["BasalUrea"])

#         FirstTopDressFert = float(request.json["FirstTopDressFert"])

#         _1tdUrea = float(request.json["_1tdUrea"])

#         MineralFertAppMethod = float(request.json["MineralFertAppMethod"])

#         Harv_method = float(request.json["Harv_method"])

#         Harv_date = float(request.json["Harv_date"])

#         Threshing_date = float(request.json["Threshing_date"])

#         Acre = float(request.json["Acre"])

#         Block = float(request.json["Block"])

#         LandPreparationMethod = float(request.json["LandPreparationMethod"])

#         CropTillageDepth = float(request.json["CropTillageDepth"])

#         CropEstMethod = float(request.json["CropEstMethod"])

#         NursDetFactor = float(request.json["NursDetFactor"])

#         TransDetFactor = float(request.json["TransDetFactor"])

#         TransIrriCost = float(request.json["TransIrriCost"])

#         OrgFertilizers = float(request.json["OrgFertilizers"])

#         Ganaura = float(request.json["Ganaura"])

#         CropOrgFYM = float(request.json["CropOrgFYM"])

#         PCropSolidOrgFertAppMethod = float(request.json["PCropSolidOrgFertAppMethod"])

#         NoFertilizerAppln = float(request.json["NoFertilizerAppln"])

#         CropbasalFerts = float(request.json["CropbasalFerts"])

#         data3 = np.array([[District, CultLand, CropCultLand, CropTillageDate, SeedlingsPerPit,
#                             TransplantingIrrigationHours, TransplantingIrrigationSource,
#                             TransplantingIrrigationPowerSource, StandingWater, BasalDAP,
#                             BasalUrea, FirstTopDressFert, _1tdUrea, MineralFertAppMethod,
#                             Harv_method, Harv_date, Threshing_date, Acre, Block,
#                             LandPreparationMethod, CropTillageDepth, CropEstMethod,
#                             NursDetFactor, TransDetFactor, TransIrriCost, OrgFertilizers,
#                             Ganaura, CropOrgFYM, PCropSolidOrgFertAppMethod,
#                             NoFertilizerAppln, CropbasalFerts]])

#         whole_prediction3 = yield_model.predict(data3)
#         prediction3 = whole_prediction3[0]
#         print(prediction3)
#         return jsonify({"yield": prediction3})

#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})

# @app.route("/leaf", methods=["POST"])
# @cross_origin()
# def members4():
#     # print(request.headers)
#     # print(request.files)
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})
#         file = request.files["file"]
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
#         allowed_extensions = {'jpg'}
#         if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
#             return jsonify({'error': 'Only JPG files are allowed'})
#         if file:
#             im_dict=file.save('research/leaf.jpg')

#             # Load and preprocess the image
#             new_img = load_img(im_dict, target_size=(224, 224))
#             img = img_to_array(new_img)
#             img = np.expand_dims(img, axis=0)
#             img = img / 255.0

#             # Load the model and make a prediction
#             model = load_model('models/leaf-cnn.h5')
#             prediction = model.predict(img)

#             # Convert the prediction to a JSON-serializable format
#             prediction_str = str(prediction[0][0])

#             return jsonify({"leaf status": prediction_str})
#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})
# def members4():
#     print(request.headers)
#     print(request.files)
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})
#         file = request.files["file"]
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})

#         allowed_extensions = {'jpg'}
#         if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
#             return jsonify({'error':'Only JPG files are allowed'})
#         if file:
#             file.save('research/leaf.jpg')
#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})

@app.route("/yield", methods=["POST"])
def members3():
    try:
        District = 15
        CultLand = float(request.json['CultLand'])
        CropCultLand = float(request.json['CropCultLand']) 

        CropTillageDate = float(request.json["CropTillageDate"])
        
        SeedlingsPerPit = float(request.json["SeedlingsPerPit"])

        TransplantingIrrigationHours = float(request.json["TransplantingIrrigationHours"])

        TransplantingIrrigationSource = float(request.json["TransplantingIrrigationSource"])

        TransplantingIrrigationPowerSource = float(request.json["TransplantingIrrigationPowerSource"])

        StandingWater = float(request.json["StandingWater"])

        BasalDAP = float(request.json["BasalDAP"])

        # BasalUrea = float(request.json["BasalUrea"])

        # FirstTopDressFert = float(request.json["FirstTopDressFert"])

        # _1tdUrea = float(request.json["_1tdUrea"])

        # MineralFertAppMethod = float(request.json["MineralFertAppMethod"])

        # Harv_method = float(request.json["Harv_method"])

        # Harv_date = float(request.json["Harv_date"])

        # Threshing_date = float(request.json["Threshing_date"])

        # Acre = float(request.json["Acre"])

        # Block = float(request.json["Block"])

        # LandPreparationMethod = float(request.json["LandPreparationMethod"])

        # CropTillageDepth = float(request.json["CropTillageDepth"])

        # CropEstMethod = float(request.json["CropEstMethod"])

        # NursDetFactor = float(request.json["NursDetFactor"])

        # TransDetFactor = float(request.json["TransDetFactor"])

        # TransIrriCost = float(request.json["TransIrriCost"])

        # OrgFertilizers = float(request.json["OrgFertilizers"])

        # Ganaura = float(request.json["Ganaura"])

        # CropOrgFYM = float(request.json["CropOrgFYM"])

        # PCropSolidOrgFertAppMethod = float(request.json["PCropSolidOrgFertAppMethod"])

        # NoFertilizerAppln = float(request.json["NoFertilizerAppln"])

        CropbasalFerts = float(request.json["CropbasalFerts"])
        
        features = [District, CultLand, CropCultLand, CropTillageDate, SeedlingsPerPit,
                            TransplantingIrrigationHours, TransplantingIrrigationSource,
                            TransplantingIrrigationPowerSource, StandingWater, BasalDAP,
                            1, 2, 3, 4,
                            5, 6.0, 4.0, 6, 7,
                            4, 5, 3,
                            5, 4, 2, 3,
                            4, 2, 4,
                            6, CropbasalFerts]
        
        n = []
        for i in features:
            n.append(float(i))
        data3 = np.array([n])
        whole_prediction3 = yield_model.predict(data3)
        prediction3 = whole_prediction3[0]
        print(prediction3)
        
        return jsonify({"yield":prediction3})

    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to process the request."})


@app.route("/leaf", methods=["POST"])
@cross_origin()
def members4():
    print("hello from leaf")
    try:
        # Get the file from the form data
        file = request.files.get('file')
        
        if not file:
            return jsonify({'error': 'No file part'})
        
        # Check if the file has a valid filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Check if the file has a valid extension
        allowed_extensions = {'jpg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Only JPG files are allowed'})
        
        # Save the file
        file.save('research/leaf.jpg')
        print("File saved")
        # Load and preprocess the image
        new_img = Image.open('research/leaf.jpg').convert('RGB').resize((224, 224))
        img = TF.ToTensor()(new_img)
        img = img.unsqueeze(0)
        img = img / 255.0
        
        # Load the model and make a prediction
        model = CNN.CNN(39)
        model.load_state_dict(torch.load('models/plant_disease_model_1_latest.pt', map_location=torch.device('cpu')))
        prediction = model(img).detach().numpy()
        
        
        # prediction_str = str(prediction[0][0])
        if prediction[0][0] < 0:
            pred = "Not Diseased"
        else:
            pred = "Diseased"
        
        return jsonify({"leaf status": pred})

        prediction_str = str(prediction[0][0])

        return jsonify({"leaf status": prediction_str})
    
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to process the request."})


if __name__ == "__main__":
    app.run(debug=True)
