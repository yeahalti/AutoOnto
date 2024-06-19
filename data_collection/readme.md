# OpenAlex Data Retrieval and Pre-processing

## Overview

This Python module `OpenAlex` provides functionality to interact with the OpenAlex API, fetch data related to topics or concepts, and perform data processing operations such as saving data to JSON or CSV files, and cleaning the retrieved data.

## Example

```python
from OpenAlex import OpenAlex 

# Create an instance of OpenAlex 
openalex = OpenAlex() 

# Fetch data related to "natural language processing" topics and save it to files 
openalex.get_data("natural language processing", "topics", "data")
```

This example fetches data related to "natural language processing" topics from the OpenAlex API and saves it to JSON and CSV files within the specified data folder.

## Data Retrieval

- The `OpenAlex` module provides methods to interact with the OpenAlex API for fetching data related to topics or concepts.
- The retrieved data is saved to JSON or CSV files within the specified data folder.

#### Functions

- `extract_id`: Extracts the topic or concept ID from the response data.
- `get_topic_concept_ids`: Gets the topic and concept IDs based on a search term.
- `fetch_data`: Fetches data from the OpenAlex API using cursor pagination.
- `get_data`: Uses the above functions to fetch data based on a search term and identifier (topics or concepts), then saves it to a file.

## Pre-processing

- Display names, publication dates, locations, etc. provide contextual information but are not directly relevant for analysis focused on conceptual content. Dropping them reduces data dimensionality.
- Metrics like `cited_by_percentile_year`, `countries_distinct_count`, `counts_by_year` have been excluded to focus analysis on paper contents.
- Paratext, grants, sustainability goals, full text origins, and access location provide additional details but are not core parts of the research content itself. Removing simplifies the scope.
- Bibliographic information and IDs for corresponding authors/institutions are useful metadata but not required for concept-based analysis.
- Retraction status, truncation of author lists, etc. provide quality checks but could introduce bias if retained.
- Publication date was removed from this list early on because seasonality seemed to have little effect on the tagging. Authors were also removed as a candidate feature, due to extremely high cardinality–there are over 200M unique authors in the dataset. Affiliation was explored, but unfortunately only 25% of the data included this field, so it most likely would not be a good feature for the model. Author and affiliation also required additional steps for disambiguation, and so we decided they most likely would not make good features.

##### Functions

- `preprocess_data`: Preprocesses the data by dropping unnecessary columns, sorting concepts, and cleaning abstracts.
- `clean_text`: Cleans the input text by converting to lowercase, removing non-alphanumeric characters, and extra spaces.
- `clean_inverted_abstract`: Cleans the inverted abstract by converting it to a dictionary and applying operations.
- `invert_abstract_dict_to_abstract`: Inverts the abstract dictionary to obtain the abstract. 

##### To Do

- Add time filter
- Handle no search results and exceptions
- Columns