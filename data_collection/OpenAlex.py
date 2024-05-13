import pandas as pd
import requests
import json
import os
import ast
import re

class OpenAlex:

    """
    A class to interact with the OpenAlex API, fetch data, and perform data operations.
    """
        
    def __init__(self):

        """
        Initializes the OpenAlex object with base URL and data folder setup.
        """
                
        self.url_base = "https://api.openalex.org/"
        self.data_folder = "../data"  # Assuming the data folder is named "data"
        
        # Check if the data folder exists, if not, create it
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
    def extract_id(self, response_data: dict):
        """
        Extract the topic or concept ID from the response data.
        
        Args:
        - response_data (dict): The JSON response data from the API.
        
        Returns:
        - str: The ID of the topic or concept.
        """
        try:
            id_url = response_data["results"][0]['id']
            return id_url.split("/")[-1]  # Extract the last part of the URL as the ID
        except (KeyError, IndexError):
            return None
                
    def get_topic_concept_ids(self, search_term: str):
        """
        Get the topic and concept IDs based on a search term.
        
        Args:
        - search_term (str): The search term to find the topic and concept.
        
        Returns:
        - tuple: A tuple containing the topic ID and concept ID.
        """
        try:
            # Request to topics endpoint
            topics_response = requests.get(f"{self.url_base}topics?search={search_term}").json()
            topic_id = self.extract_id(topics_response)
            
            # Request to concepts endpoint
            concepts_response = requests.get(f"{self.url_base}concepts?search={search_term}").json()
            concept_id = self.extract_id(concepts_response)
            
            return topic_id, concept_id
        except requests.RequestException as e:
            print(f"Request Exception: {e}")
            return None, None 
    
    def fetch_data(self, topic_id: str, concept_id: str, is_topic: bool = True, per_page: int = 100) -> list:
        """
        Fetch data from Openalex API using cursor pagination.
        
        Args:
        - concept_id (str): The ID of the concept to filter the works.
        - per_page (int): Number of results per page. Default is 100.
        
        Returns:
        - list: List of dictionaries containing the fetched data.
        """
        all_results = []  # Initialize an empty list to store all results
        iterations = 1  # Counter to control iterations
        
        filter_param = "topics.id" if is_topic else "concepts.id"
        identifier = topic_id if is_topic else concept_id

        # URL with initial filter and per-page parameter
        url = f"{self.url_base}works?filter={filter_param}:{identifier}&per_page={per_page}&cursor=*"
        
        # Loop for cursor pagination to download data
        while url:
            try:
                print(f"\nIteration: {iterations}")
                print(f"URL: {url}")
                page_with_results = requests.get(url).json()
                
                # Append results to the list
                all_results.extend(page_with_results['results'])
                
                # Update URL to the next_cursor value or break the loop if there's no next_cursor
                next_cursor = page_with_results['meta'].get('next_cursor')
                if not next_cursor:
                    break
                url = f"{self.url_base}works?filter={filter_param}:{identifier}&per_page={per_page}&cursor={next_cursor}"
                
                iterations += 1  # Increment the counter
            except requests.RequestException as e:
                print(f"Request Exception: {e}")
                # Handle the error here (save and write data, or any necessary action)
                break  # Break out of the loop if an error occurs
            except KeyError as e:
                print(f"KeyError: {e}")
                # Handle the missing key error if 'results' or 'meta' is missing
                break  # Break out of the loop if a key error occurs
        
        return all_results
        
    def sort_concepts(self, concepts_str):
        """
        Sort concepts from a string representation of a list of dictionaries.
        
        Args:
        - concepts_str (str): String representation of a list of dictionaries.
        
        Returns:
        - dict: Sorted dictionary of concepts.
        """
        try:
            concepts_list = json.loads(concepts_str.replace("'", '"'))
            filtered_list = [{k: v for k, v in item.items() if k not in ['id', 'wikidata']} for item in concepts_list if item['score'] != 0]
            
            # Sort the list by level and score
            sorted_list = sorted(filtered_list, key=lambda x: (x['level'], -x['score']))

            # Create a dictionary to store the result
            result_dict = {}

            # Populate the result_dict
            for item in sorted_list:
                level = item['level']
                display_name = item['display_name']
                score = item['score']

                # If the level is not already a key, add it with an empty list as the value
                if level not in result_dict:
                    result_dict[level] = []

                # Append the display name and score as a pair to the list for the current level
                result_dict[level].append([display_name, score])

            return result_dict
        except ValueError:  # Catching the ValueError that might indicate a JSON decoding error
            return concepts_str
    
    def clean_inverted_abstract(self, abstract):
        """
        Clean the abstract by converting it to a dictionary and applying operations.
        
        Args:
        - abstract (str): String representation of the abstract.
        
        Returns:
        - dict or None: Cleaned abstract as a dictionary or None if the string representation is invalid.
        """
        try:
            abstract_dict = ast.literal_eval(abstract)
            # Apply operations on the dictionary representation if needed
            # For example, to lowercase all keys in the dictionary:
            abstract_dict = {k.lower(): v for k, v in abstract_dict.items()}
            return abstract_dict
        except (ValueError, SyntaxError):
            return None  # Handle any invalid string representations
    
    def invert_abstract_dict_to_abstract(self, inverted_abstract_dict):
        """
        Invert the abstract dictionary to obtain the abstract.
        
        Args:
        - inverted_abstract_dict (dict): Inverted abstract dictionary.
        
        Returns:
        - str or None: Inverted abstract as a string or None if the input is None.
        """
        if inverted_abstract_dict is None:
            return None

        abstract = [" "] * 1000
        for word, indexes in inverted_abstract_dict.items():
            for index in indexes:
                if 0 < index <= 1000:
                    abstract[index-1] = word
        return " ".join(abstract).strip()
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by converting to lowercase, removing non-alphanumeric characters, and extra spaces.

        Parameters:
        - text (str): The text to clean.

        Returns:
        - str: The cleaned text.
        """
        try:
            text = text.lower()
            text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
            text = re.sub(' +', ' ', text)
            text = text.strip()
        except:
            text = ""
        return text    
    
    def preprocess_data(self, file_path: str):
        """
        Preprocess the data by dropping unnecessary columns, sorting concepts, and cleaning abstracts.
        
        Args:
        - file_path (str): Path to the CSV file containing the data.
        
        Returns:
        - pd.DataFrame: Cleaned DataFrame.
        """
        df = pd.read_csv(file_path, low_memory=False)
        
        # Drop unnecessary columns
        new_df = df.drop(columns=["display_name", "publication_date", "primary_location", "countries_distinct_count", "institutions_distinct_count",
                                   "corresponding_author_ids", "corresponding_institution_ids", 'apc_list', 'apc_paid',
                                   'cited_by_percentile_year', 'biblio', 'is_retracted', 'is_paratext', 'mesh', 'locations_count', 
                                   'locations', 'best_oa_location', 'sustainable_development_goals', 'grants', 'cited_by_api_url', 'counts_by_year', 
                                   'updated_date', 'created_date', 'fulltext_origin', 'is_authors_truncated'])
        
        # Sort concepts if the 'concepts' column exists
        if 'concepts' in new_df.columns:
            new_df['concepts'] = new_df['concepts'].apply(self.sort_concepts)
        
        # Clean abstracts
        if 'abstract_inverted_index' in new_df.columns:
            new_df['abstract_inverted_index'] = new_df['abstract_inverted_index'].apply(self.clean_inverted_abstract)
            new_df['abstract'] = new_df['abstract_inverted_index'].apply(self.invert_abstract_dict_to_abstract)
            new_df['abstract'] = new_df['abstract'].apply(self.clean_text)
        return new_df    
    
    def save_to_json(self, data: list, file_path: str):
        """
        Save retrieved data to a JSON file.
        
        Args:
        - data (list): List of dictionaries to be saved.
        - file_path (str): Path where the JSON file will be saved.
        """
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
    
    def save_to_csv(self, data: list, file_path: str):
        """
        Save retrieved data to a CSV file.
        
        Args:
        - data (list): List of dictionaries to be saved.
        - file_path (str): Path where the CSV file will be saved.
        """
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def get_data(self, search_term: str, works_param: str, file_name: str):
        """
        Fetch data based on a search term and identifier, then save it to a file.
        
        Args:
        - search_term (str): The search term to find the topic and concept.
        - works_param (str): The identifier parameter, either "topic" or "concept".
        - file_path (str): Path where the data will be saved.
        """
        topic_id, concept_id = self.get_topic_concept_ids(search_term)
        
        if works_param == "topics":
            is_topic = True
        elif works_param == "concepts":
            is_topic = False
        else:
            print("Invalid value for works_param. Please provide 'topics' or 'concepts'.")
            return
        
        data = self.fetch_data(topic_id, concept_id, is_topic=is_topic)
        # self.save_to_json(data, os.path.join(self.data_folder, file_name+".json"))
        self.save_to_csv(data, os.path.join(self.data_folder, file_name + ".csv"))

        cleaned_df = self.preprocess_data(os.path.join(self.data_folder, file_name + ".csv"))
        self.save_to_csv(cleaned_df, os.path.join(self.data_folder, "cleaned_data.csv"))


