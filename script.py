import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pandasql import sqldf
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Load
data = pd.read_csv(r'/home/jmsbs/PycharmProjects/gym_exercise_recommendation/megaGymDataset.csv')

# Preprocessing
data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
data = data.drop(columns=['Rating', 'RatingDesc'])

# Encode variables
le = LabelEncoder()  # Create a LabelEncoder object

# Fit the LabelEncoder to the data and transform it
data['Type_encoded'] = le.fit_transform(data['Type'])
data['Equipment_encoded'] = le.fit_transform(data['Equipment'])
data['BodyPart_encoded'] = le.fit_transform(data['BodyPart'])

data = pd.concat([data, pd.get_dummies(data['Level'], prefix='Level')], axis=1)

# Compute Pearson correlation coefficient and p-value
columns_for_corr = ['Type_encoded', 'Equipment_encoded', 'BodyPart_encoded', 'Level_Beginner', 'Level_Intermediate',
                    'Level_Expert']


def calc_correlation():
    for i in range(len(columns_for_corr)):
        for j in range(i + 1, len(columns_for_corr)):
            correlation_coefficient, p_value = pearsonr(data[columns_for_corr[i]], data[columns_for_corr[j]])
        # print(f"Pearson correlation coefficient for '{columns_for_corr[i]}' and '{columns_for_corr[j]}':", correlation_coefficient)
        # print("P-value:", p_value)


calc_correlation()

# Similarity Matrix
columns = ['Type_encoded', 'Equipment_encoded', 'BodyPart_encoded', 'Level_Beginner', 'Level_Intermediate',
           'Level_Expert']  # Create a new DF
data1 = data[columns]
similarity_matrix = cosine_similarity(data1)  # Calculate the Cosine Similarity between each pair of exercises
similarity_df = pd.DataFrame(similarity_matrix, index=data1.index,
                             columns=data1.index)  # Convert the similarity matrix to a DataFrame for better readability
similarity_df.index = data['ID']  # Replace the indices with the exercise titles
similarity_df.columns = data['ID']  # Replace the column names with the exercise titles


def recommend_exercises(ID, similarity_df, num_recommendations=5):
    exercise_similarities = similarity_df[ID]  # Get the column corresponding to the given exercise ID
    similar_exercises = exercise_similarities.sort_values(
        ascending=False)  # Sort the exercises in descending order of similarity
    top_exercises = similar_exercises.head(num_recommendations + 1)  # Get the top N exercises
    top_exercises = top_exercises.loc[top_exercises.index != ID]  # Remove the given exercise from the recommendations

    # Create a DataFrame with the exercise IDs and their corresponding titles
    top_exercises_df = pd.DataFrame(top_exercises)
    top_exercises_df['Title'] = top_exercises_df.index.map(data.set_index('ID')['Title'])
    top_exercises_df.reset_index(inplace=True)
    top_exercises_df.columns = ['ID', 'Similarity', 'Title']

    return top_exercises_df


# Test the function with a valid exercise ID from your DataFrame
valid_ID = data['ID'].iloc[0]  # Replace this with a valid ID from your DataFrame
recommendations = recommend_exercises(valid_ID, similarity_df)
# print(recommendations)

user_inputs = []


def get_user_input(prompt):
    user_input = input(prompt)
    user_inputs.append(user_input)
    return user_input


def recommend_based_on_user_input(data, similarity_df, num_recommendations=5):
    # Fit the LabelEncoder to the data
    le_type = LabelEncoder().fit(data['Type'])
    le_body_part = LabelEncoder().fit(data['BodyPart'])
    le_equipment = LabelEncoder().fit(data['Equipment'])

    # Ask the user for their preferences
    type_preference = get_user_input("What type of exercise would you like to do?")
    body_part_preference = get_user_input("What body part would you like to train?")
    equipment_preference = get_user_input("What equipment do you have available?")

    # Encode the user's preferences using the fitted LabelEncoders
    type_encoded = le_type.transform([type_preference])
    body_part_encoded = le_body_part.transform([body_part_preference])
    equipment_encoded = le_equipment.transform([equipment_preference])

    # Create a DataFrame that contains the user's preferences
    user_preferences = pd.DataFrame({
        'Type_encoded': type_encoded,
        'BodyPart_encoded': body_part_encoded,
        'Equipment_encoded': equipment_encoded
    }, index=[0])

    # Filter the data to only include exercises that match the user's body part preference
    data_filtered = data[data['BodyPart_encoded'] == body_part_encoded[0]].copy()

    # If the user inputs 'any' for the equipment, do not filter the data based on equipment
    # if equipment_preference.lower() != 'any':
    #     data_filtered = data_filtered[data_filtered['Equipment_encoded'] == equipment_encoded[0]]

    # Compute the cosine similarity between the user's preferences and the filtered data
    similarity_scores = cosine_similarity(data_filtered[user_preferences.columns], user_preferences)

    # Sort the exercises by their similarity to the user's preferences
    data_filtered['Similarity'] = similarity_scores
    recommendations = data_filtered.sort_values(by='Similarity', ascending=False).head(num_recommendations)

    # Return the recommended exercises
    return recommendations[['ID', 'Title', 'Similarity', 'Level']]


recommendations = recommend_based_on_user_input(data, similarity_df)
print(f"User inputs were: {user_inputs}.")
print(f"The recommended exercises are the following: \n {recommendations}.")
