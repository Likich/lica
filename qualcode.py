# generation_config = model.generation_config
# generation_config.max_new_tokens = 10
# generation_config.temperature = 0.7
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1
# generation_config.pad_token_id = tokenizer.eos_token_id
# generation_config.eos_token_id = tokenizer.eos_token_id
def combine_data(file1, file2):
    import pandas as pd

    df_raw = pd.read_csv(file1)
    clusters_documents = pd.read_csv(file2)

    combined_df = pd.merge(df_raw, clusters_documents, left_index=True, right_on='Doc_ID')

    # Check the combined dataframe
    print(combined_df.head(20))


    df = combined_df[['paragraphs', 'Topic', 'Doc']]
    df.to_csv('doc_topics.csv', index=False)




# Replace this with your actual OpenAI API key
# API_KEY = "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"

# # Load the codes from the CSV file
# def load_codes_from_csv(file_path):
#     import openai
#     import pandas as pd
#     df = pd.read_csv(file_path)
#     codes = df['Extracted codes from lemmas'].dropna().unique().tolist()
#     print("Codes loaded from CSV:")
#     print(codes)
    
#     return codes

# # Function to call OpenAI API for axial coding
# def get_axial_coding(codes):
#     import openai
#     import pandas as pd
#     try:
#         openai.api_key =  "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"
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",  # Or "gpt-4" if available
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an assistant that helps with qualitative data analysis."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Please categorize the following codes into higher-order categories (axial coding): {', '.join(codes)}"
#                 }
#             ]
#         )
        
#         # Extracting the categorized response
#         categories = response['choices'][0]['message']['content']
#         print("\nAxial Coding Result:\n", categories)
    
#     except Exception as e:
#         print("Error:", str(e))

# Path to the CSV file
# csv_file_path = 'coded_lemmas_new.csv'

# # Load codes from CSV
# codes = load_codes_from_csv(csv_file_path)

# # Run the function with the loaded codes
# get_axial_coding(codes)

