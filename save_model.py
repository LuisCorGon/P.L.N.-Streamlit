import main
import joblib



def save_model(model = main.get_model(), name="model"):
    joblib.dump(model, f"{name}.pkl")
    print(f"Modelo guardado como {name}.pkl")

save_model()