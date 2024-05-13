# AutoOnto

Using LLMs and Topic Modelling to generated ontologies

## Structure

- **data**: Contains data retrieved and cleaned from OpenAlex API.
  
  - `data.csv`: Original data.
  - `cleaned_data.csv`: Processed and cleaned data.

- **data_collection**: Module for data retrieval and cleaning.
  
  - `OpenAlex.py`: Implementation of the OpenAlex class for fetching and processing data from the OpenAlex API.
  - `example.py`: Example usage of the module.
  - `visualization.ipynb`: Jupyter notebook for data exploration and visualization.

- **ontology_generation**: Functions for ontology generation.
  
  - `Evaluation.py`: Module for extracting topics from existing ontologies, cleaning them, and comparing using similarity metrics.
  - `OntologyEncap.py`: Module for encoding ontology to TTL format.
  - `OntologyGen.py`: Module for ontology generation, utilizing utilities from `utilsLLM`.
  - `utilsLLM.py`: Utility functions for prompting to GPT and cleaning responses.

## How to use

`run.py`: Shows how the pipeline can be used

`run.ipynb`: Jupyter notebook version of the above script

- Modify search term in `run.py` or alternatively run `example.py` in `data_collection` according to needs and pick either ([topics](https://docs.openalex.org/api-entities/topics))/([Concepts](https://docs.openalex.org/api-entities/concepts))as a parameter

## Dependencies

List the dependencies required to run your project.

## Contributing

Explain how others can contribute to your project, including guidelines for pull requests and code formatting.

## License

Include the license under which your project is released.
