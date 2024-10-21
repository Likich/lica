
def combine_data(file1, file2):
    import pandas as pd

    df_raw = pd.read_csv(file1)
    clusters_documents = pd.read_csv(file2)

    combined_df = pd.merge(df_raw, clusters_documents, left_index=True, right_on='Doc_ID')

    # Check the combined dataframe
    print(combined_df.head(20))


    df = combined_df[['paragraphs', 'Topic', 'Doc']]
    df.to_csv('doc_topics.csv', index=False)


