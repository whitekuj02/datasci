import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def distance(x, y, test_x, test_y):
    dis = np.sqrt(np.power(x - test_x, 2) + np.power(y - test_y, 2))
    return dis
class KNNAlgorithm:
    def __init__(self, df, k, target):
        self.df = df
        self.k = k
        self.target_name = target.name
        self.target = target

    def algorithmTest(self, test):
        x = self.df.iloc[:, 0].to_numpy()
        y = self.df.iloc[:, 1].to_numpy()
        test_x = test[:, 0]
        test_y = test[:, 1]

        # 배열들의 길이 저장
        lengths = [len(x), len(y), len(test_x), len(test_y)]

        # 배열들을 하나의 배열로 결합
        combined_array = np.hstack((x, y, test_x, test_y)).reshape(-1, 1)

        # 표준화
        scaler = StandardScaler()
        scaled_combined_array = scaler.fit_transform(combined_array)

        # 결합된 배열을 원래의 배열 크기로 다시 분할
        x_scaled, y_scaled, test_x_scaled, test_y_scaled = np.split(scaled_combined_array, np.cumsum(lengths)[:-1],
                                                                    axis=0)

        # 배열들을 1차원으로 변환
        x_scaled = x_scaled.ravel()
        y_scaled = y_scaled.ravel()
        test_x_scaled = test_x_scaled.ravel()
        test_y_scaled = test_y_scaled.ravel()

        dis = distance(x_scaled, y_scaled, test_x_scaled, test_y_scaled)

        dis_df = pd.Series(data=dis, index=self.df.index)
        dis_df = dis_df.sort_values()

        dis_selected = dis_df.head(self.k)
        #분류된 점들의 index로 값 추출
        index_df = dis_selected.index
        dot_df = self.df.loc[index_df]

        category = dot_df[self.target_name]
        category = category.astype("category")
        category = category.cat.set_categories(self.target.cat.categories)

        print(dot_df)

        classifier = category.value_counts().sort_values(ascending=False).index[0]

        return classifier




df = pd.DataFrame({
    "Height": [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
    "Weight": [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
    "T Shirt Size": ["M", "M", "M", "M", "M", "M", "M", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"]
})

target = df["T Shirt Size"]
target = target.astype("category")

test = np.array([[161, 61]])

KNN = KNNAlgorithm(df, 5, target)

class_KNN = KNN.algorithmTest(test)

print("output is " + class_KNN)