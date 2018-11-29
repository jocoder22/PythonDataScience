import numpy as np
import pandas as pd
from sklearn.preprocessing import oneHotEndocer
from sklearn.preprocessing import LabelEncoder

categoricalFeature = pd.series(['Monday', 'Tuesdau', 'Wednesday', 'Thursday',
                                'Friday', 'Saturday', 'Sunday'])\

catmapping = pd.get_dummies(categoricalFeature)
