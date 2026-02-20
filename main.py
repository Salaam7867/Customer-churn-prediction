from src.preprocess import load_data, clean_data, prepare_features
from src.train import train_xgb

def main():
    df = load_data("CustomerChurn.csv")
    df = clean_data(df)
    X, y = prepare_features(df)

    model, threshold = train_xgb(X, y)

if __name__ == "__main__":
    main()