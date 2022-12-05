# What-If-Code-Tool
## Main Idea
[Google WIT](https://github.com/PAIR-code/what-if-tool) was the main inspiration for this project. Our goal is to create a similar tool purely for focusing on ML models revolving around software engineering design and principles, such as code completion and code generation. 

[BertViz](https://github.com/jessevig/bertviz) is a good first example for where our tool will go. We hope to support a full dashboard of several views that researchers would find helpful in order to analyze their models. This would probably include newly generated word count charts, probability distributions for new tokens, and attention views.

## Development
- Pip tool: user can install this tool from pip/conda and utilize with their NLP model
- Python Backend: user designates dataset and model as parameters for our tool. Our tool then runs the model and produces some vector dataset in its object.
- Jupyter-Dash Frontend: Jupyter-Dash allows for easy creation for data dashboard. Provides ability for easy callback methods with just Python.
<!-- - Ideas for Frontend
  - Dashboard: Several visuals at the same time. This would allow the user to interact with each of the visuals provided
  - One at a time: User designates which view they want to see from their view at any given point
  - Visuals would be available in python notebooks
- Some ideas: BertViz, Google WIT
- Plotly is a great tool to create large dashboard from python. Could be useful for a dashboard view
- Flask/Django can be used to implement the interactive component of the charts (connect listening events to python code) -->

## Development Plans
<!-- - [ ] Interview ML researchers (SEMERU) for what specific views would be useful for their exploration
- [ ] Implement back-end to spit out some output to dynamic html
- [ ] Create new views, probability distribution
- [ ] Allow for some interactive aspect with the charts -->
- [ ] Code concept groupings view: categorize each of the tokens generated in output based on what type they are in code language (declaration, assignment, functions, etc.)
- [ ] Display some statistics about the generated output with specific model (median, max, min, etc.)
- [ ] Dynamics re-execution of pipeline when:
  - [ ] User edits # of tokens
  - [ ] User edits # of input sequences
  - [ ] User changes model
  - [ ] User selects new descriptive statistic
- [ ] Implement bertviz attention models inside app with Dash if possible

## Current Diagrams
### Components UML
![Components](Artifacts/component-diagram-updated.png)

### Sequence Diagram
![Sequence Diagram](Artifacts/sequence-diagram-updated.png)

## Installation
First prototype is currently available on PyPi. It does not do much (yet), but you can initialize an interactive bar graph of token counts in your notebook.

```
!pip install codewit-semeru

from wit_code import wit_code
wit_code.WITCode("gpt2", "This is a chunk of code", "gpt2")
```
These lines can be run directly from your notebook. Python 3.8 is required.

### Build and Run Docker Image
Start docker

Navigate to project folder and run ```docker-compose up -d --build``` to build image

Navigate to ```localhost:8888``` to run jupyter notebook. password is ```wit```

To stop docker container run ```docker-compose down```


### Build and Run Docker Image
Start docker

Navigate to project folder and run ```docker-compose up -d --build``` to build image

Navigate to ```localhost:8888``` to run jupyter notebook. password is ```wit```

To stop docker container run ```docker-compose down```

