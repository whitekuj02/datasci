import pandas as pd
import numpy as np
import math

def entropy(num):
    #오류 뜨지만 상관 없음 1,0 일 때 div 0 오류 하지만 nan값 0으로 대체시 문제없음
    result = -1 * (num * np.log2(num) + (1-num) * np.log2(1-num))
    result = np.nan_to_num(result)
    return result

class DecisionTree:
    def __init__(self, df, target, tree):
        self.X_column = df.columns
        self.data = df
        self.target_columns = target.name
        self.target = target
        self.tree = tree

    def root_decision(self):
        if len(self.X_column) < 2:
            return self.tree
        target_mother_num = self.target.shape[0]
        target_num = len(self.data.loc[self.data[self.target_columns] == self.target.value_counts().index[0]].value_counts().sort_index().to_numpy())
        target_result_num = target_num / target_mother_num
        if target_result_num == 0:
            print(" 트리 끝 ")
            return self.tree
        whole_data_entropy = entropy(target_result_num)

        result_df = pd.DataFrame()
        for i in self.X_column:
            # target columns은 break
            if i == self.target_columns:
                break
            # 카테고리 별로 묶어서 수를 센다
            each_cat = self.data[i].value_counts().sort_index().index
            each_cat_num = self.data[i].value_counts().sort_index().to_numpy()
            all_num = self.data[i].shape[0]
            # 데이더 양의 비율
            num_div_each = each_cat_num / all_num
            # 분류된 데이터의 비율
            num_all_cat_target = self.data.loc[self.data[self.target_columns] == self.target.value_counts().index[0], i].value_counts().sort_index().to_numpy()

            # entropy num 값 구하기
            entropy_input_num = num_all_cat_target / each_cat_num

            #entropy 구하기
            entropy_result = entropy(entropy_input_num)

            each_attribute = np.ones(len(each_cat_num)) @ (num_div_each * entropy_result)

            #결과 값
            result = whole_data_entropy - each_attribute
            add_result = pd.Series(result)
            result_df[i] = add_result

        print(result_df)
        decision_index = result_df.max().idxmax()
        print(decision_index)

        # 가장 큰 entropy 값이 0이거나 0보다 작으면 decision의 의미가 없음
        if result_df.max().max() <= 0:
            print(" 트리 끝 ")
            return self.tree
        # tree index값을 nested list로 분리
        # 'District' 열의 고유한 카테고리 값을 얻음
        unique = self.data[decision_index].cat.categories

        # 각 카테고리에 대한 인덱스를 중첩 리스트로 구성
        nested_index_list = [self.data[self.data[decision_index] == column].index.tolist() for column in unique]
        self.tree = nested_index_list
        print(self.tree)
        # 다시 한 번 DecisionTree 생성 후 df에는 result_df.max().idxmax() 제거하며 각자의 leaf 데이터 ex) district는 3개의 leave로 3개의 class를 만들어야함
        # target은 나누어진 데이터의 target 값
        # 트리는 nested list index 별로 넘김
        result_tree = []
        for i in self.tree:
            df_nest = self.data.drop(decision_index, axis=1)
            df_nest = df_nest.loc[i]
            target_nest = self.target.loc[i]
            print(df_nest)
            print(target_nest)
            DT = DecisionTree(df_nest, target_nest, i)
            each_tree = DT.root_decision()
            result_tree.append(each_tree)
        self.tree = result_tree
        return result_tree

    #def decisionTest(self, test_df):



df = pd.DataFrame({
    "District": ["Suburban", "Suburban", "Rural", "Urban", "Urban", "Urban", "Rural", "Suburban", "Suburban", "Urban",
                 "Suburban", "Rural", "Rural", "Urban"],
    "House Type": ["Detached", "Detached", "Detached", "Semi-detached", "Semi-detached", "Semi-detached",
                   "Semi-detached", "Terrace", "Semi-detached", "Terrace", "Terrace", "Terrace", "Detached", "Terrace"],
    "Income": ["High", "High", "High", "High", "Low", "Low", "Low", "High", "Low", "Low", "Low", "High", "Low", "High"],
    "Previous": ["No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Outcome": ["Not responded", "Not responded", "Responded", "Responded", "Responded",
                "Not responded", "Responded", "Not responded", "Responded", "Responded", "Responded",
                "Responded", "Responded", "Not responded"]
}, dtype="category")

target = df["Outcome"]
tree = [i for i in range(0, len(target))]

print(df)
print(target)

DT = DecisionTree(df, target, tree)

tree = DT.root_decision()
print(tree)

