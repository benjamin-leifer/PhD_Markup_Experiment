import pymongo
from pymongo import MongoClient
import json
import gridfs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import scipy.optimize as optimize
import pylab


# Connect to MongoDB
client = MongoClient('localhost', 27017)

# Connect to database
db = client['test']

# Connect to collection
collection = db['test']

# Insert data
