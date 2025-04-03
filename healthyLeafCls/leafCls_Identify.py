import json
import torch
from ultralytics import YOLO



class HealthyLeafClassification:
    def __init__(self, model):
        self.device = 0 if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            print("inferencing with cpu capabilities")
        else:
            print("inferencing with gpu capabilities")

        self.model = YOLO(model)

    
    def Identiify_HealthyLeaf(self, image, th=0.8):
        clsResults = self.model.predict(source=image, show=False, save=False, conf=th,device=self.device, verbose=False)
        predictionDect = clsResults[0].to_json()
        predictionDect = json.loads(predictionDect)

        if len(predictionDect) != 0:
            for eachPreds in predictionDect:
                if eachPreds["name"] == "Negative":
                    return {"Result":"No Leaf Identified"}
                elif eachPreds["name"] == "healthyLeaf":
                    return {"Result": "Healthy Leaf"}
                else:
                    return {"Result": "Un Healthy Leaf"}
            return {"Result":"No Leaf Identified"}
        else:
            return []