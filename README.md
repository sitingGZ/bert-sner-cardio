# Multi-layer Clinical Semantic Annotation 

 ` Build Image: docker build -t sner-app . `
 ` Run Docker Container with the created Image and map to the given HOST:  docker run -dp $HOST:8050:8050 sner-app`

## Motivation
In clinical settings, entities often have multiple levels of meaning and a phrase can contain multiple semantic types. Our multi-layer Clinical Named Entity Recognition (NER) model considers seven different semantic groups and follows standardized semantic types. 

    - A single phrase could represent a symptom (conceptual entity) and a pathological function (phenomenon or process). The model can handle such complex annotations, ensuring comprehensive information coverage without losing important distinctions.
  
    - By identifying entities in multiple semantic dimensions, the model can extract a broader spectrum of clinically relevant information, leading to the discovery of advanced medical knowledge.
  
    - By incorporating standardized semantic types, the model can be easily adapted to changing medical terminology and evolving knowledge in healthcare. New semantic types can be incorporated without modifying the model architecture and parameters.
    - 
   

## Prerequisites
To capture different aspects of medical information commonly found in clinical texts, we select seven semantic groups to represent different categories of entities.
    
    Physical Object: This group includes entities such as anatomy (e.g., organs, body parts), drugs, chemicals, and medical devices. These are tangible and concrete objects frequently mentioned in medical texts.

    Conceptual Entity: This group represents abstract ideas or attributes related to clinical concepts and includes clinical attributes, quantitative concepts (e.g., measurements), signs or symptoms, laboratory or test results, and temporal concepts (e.g., time-related information).

    Procedure: The Activity group covers different medical procedures and actions, including laboratory procedures, diagnostic procedures, and therapeutic or preventive procedures.

    Phenomenon or Process: This group encompasses the entities that represent various physiological and pathological states and processes, such as injuries, physiological functions, pathological functions, diseases, and mental or behavioural disorders. 

    Health State: Health State refers to entities representing the condition or state of an individual, either in a healthy state or experiencing a worsening health condition.

    Factuality: This group includes entities that indicate the certainty or likelihood of an event or concept. Entities in this group convey information about negated, slight, questionable, future, or unlikely events or states.


## Annotation Guidelines

Login: select user email -> password -> login -> select section
Select Document ID
See more details in Annotation_instruction.pdf

## System Demo Screencast

https://github.com/sitingGZ/bert-sner-cardio/assets/33466124/ef413b30-8833-491d-a121-96b7ee84ad63

