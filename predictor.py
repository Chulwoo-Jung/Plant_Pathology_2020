import torch 
import torch.nn.functional as F

class Predictor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device


    def predict_proba(self, inputs):
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            pred_proba = F.softmax(outputs, dim=1)
        return pred_proba
    
    def predict(self, inputs):
        pred_proba = self.predict_proba(inputs)
        pred_label = torch.argmax(pred_proba, dim=-1)
        return pred_label
    
    
    
    