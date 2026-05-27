import numpy as np
import glob

files = glob.glob(r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\gait-in-parkinsons-disease-1.0.0\**\*.txt", recursive=True)

lengths = []
for f in files:
    data = np.loadtxt(f)
    lengths.append(data.shape[0])  # number of rows = samples

print("Min samples:", min(lengths))
print("Max samples:", max(lengths))
print("Avg samples:", sum(lengths) / len(lengths))
