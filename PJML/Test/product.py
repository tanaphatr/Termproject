import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),   '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

if __name__ == "__main__":
    data = load_dataps()
    preprocessed_data = preprocess_dataps(data)
    print(preprocessed_data)