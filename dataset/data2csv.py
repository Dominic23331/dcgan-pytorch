import pandas as pd
import os


def make_csv():
    img_list = os.listdir("./data")
    img_csv = pd.DataFrame()
    for i in range(len(img_list)):
        img_list[i] = "./dataset/data/" + img_list[i]
    img_csv["img"] = img_list
    img_csv.to_csv("data.csv")

if __name__ == '__main__':
    make_csv()
