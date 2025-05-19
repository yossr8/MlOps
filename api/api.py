import os
import pickle
import json
import litserve as ls
import pandas as pd
from pydantic import ValidationError
import joblib

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        # Load the model from the specified path
        output_dir = os.path.join(os.getcwd(), "models")
        with open(f"{output_dir}/model.pkl", "rb") as f:
            self._model = pickle.load(f)
            
    def decode_request(self, request):
        try:
            columns = request["columns"]
            data = request["rows"]
            df = pd.DataFrame(data, columns=columns)
            print(df)
            return df
        except Exception:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        print(output, 9 * "*")
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response