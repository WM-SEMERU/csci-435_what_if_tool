# What-If-Code-Tool
## Main Idea
[Google WIT](https://github.com/PAIR-code/what-if-tool) was the main inspiration for this project. Our goal is to create a similar tool purely for focusing on ML models revolving around software engineering design and principles, such as code completion and code generation. 

[BertViz](https://github.com/jessevig/bertviz) is a good first example for where our tool will go. We hope to support a full dashboard of several views that researchers would find helpful in order to analyze their models. This would probably include newly generated word count charts, probability distributions for new tokens, attention views.

## Development
- Pip tool: user can install this tool from pip/conda and utilize with their NLP model
- Python Backend: user designates dataset and model as parameters for our tool. Our tool then runs the model and produces some vector dataset in its object. 
- Ideas for Frontend
  - Dashboard: Several visuals at the same time. This would allow the user to interact with each of the visuals provided
  - One at a time: User designates which view they want to see from their view at any given point
  - Visuals would be available in python notebooks
- Some ideas: BertViz, Google WIT
- Plotly is a great tool to create large dashboard from python. Could be useful for a dashboard view
- Flask/Django can be used to implement the interactive component of the charts (connect listening events to python code)

## Current Plans
- [ ] Interview ML researchers (SEMERU) for what specific views would be useful for their exploration
- [ ] Implement back-end to spit out some output to dynamic html
- [ ] Create new views, probability distribution
- [ ] Allow for some interactive aspect with the charts

## Current Diagrams
### Components UML
![Components](Artifacts/component.png)

### Sequence Diagram
![Sequence Diagram](Artifacts/sequence.png)

## Installation
First prototype of our tool is still in progress.
