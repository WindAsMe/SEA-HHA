from scipy.stats.stats import mannwhitneyu
import numpy as np
import csv


def open_csv(path):
    data = []
    with open(path) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            d = []
            for s in row:
                d.append(float(s))
            data.append(d)
    return np.array(data)


path_VEGE = "C:\\Users\\WindAsMe\\Downloads\\Vegetation Evolution_Python\\VEGE_Data\\F4_10D.csv"
path_iVEGE = "C:\\Users\\WindAsMe\\Downloads\\Vegetation Evolution_Python\\iVEGE_Data\\F4_10D.csv"
VEGE_obj = open_csv(path_VEGE)
iVEGE_obj = open_csv(path_iVEGE)

print(VEGE_obj)
final_VEGE = VEGE_obj[:, len(VEGE_obj[0])-1]
final_iVEGE = iVEGE_obj[:, len(iVEGE_obj[0])-1]
print(final_VEGE, np.mean(final_VEGE))
print(final_iVEGE, np.mean(final_iVEGE))
print(mannwhitneyu(final_VEGE, final_iVEGE))
