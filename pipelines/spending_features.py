"""Build shared spending labels and historical feature tables."""

from src.spending.labels import main as build_labels
from src.spending.features import main as build_features

def main():
    build_labels()
    build_features()

if __name__ == "__main__":
    main()
