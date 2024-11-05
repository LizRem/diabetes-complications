# ML4H-diabetes-complications
This repository contains the code for the workshop paper *Exploring Long-Term Prediction of Type 2 Diabetes Microvascular Complications* (Machine Learning for Healthcare 2024).

## The challenge
Healthcare data is often stored in different places using different clinical code ontologies, such as the International Classification of Diseases (ICD) or Systematized Nomenclature of Medicine-Clinical Terms (SNOMED-CT). In order to analyse the data this requires mapping between these coding systems, which can result in a loss of data. In this paper we explore the effectiveness of code-agnostic models which mediates the need for mapping.

## Prepare your data 
This paper compared two input types, text versus medical codes. For the text-based approach, we order the events (diagnosis, medication or procedure) for each patient chronologically and then concatenate the textual descriptors into a sentence. For the medical code approach, we do the same but concat the medical codes into a sequence. See image below for an example.

![Input format for text and code approaches](images/sentences.png)
*Input format for text and code approaches*

## Model 
This paper predominately uses [gatortron-base](https://huggingface.co/UFNLP/gatortron-base) but we also compare the performance across [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) and [Biomedical longformer](https://huggingface.co/kiddothe2b/biomedical-longformer-base).
Model stored in: gatortron_bceloss.py

| Model                          | Time    | Text-based                   |                               | Code-based                    |                               |
|--------------------------------|---------|------------------------------|-------------------------------|-------------------------------|-------------------------------|
|                                |         | Micro-F1                     | Micro-AUPRC                   | Micro-F1                      | Micro-AUPRC                   |
| **BERT-base**                  | 1 year  | 0.437 [0.426, 0.450]         | 0.415 [0.400, 0.430]          | 0.418 [0.407, 0.429]          | 0.388 [0.374, 0.403]          |
|                                | 5 year  | 0.467 [0.457, 0.477]         | 0.457 [0.444, 0.471]          | 0.433 [0.433, 0.454]          | 0.419 [0.406, 0.434]          |
|                                | 10 year | 0.482 [0.471, 0.493]         | 0.486 [0.473, 0.500]          | 0.458 [0.447, 0.469]          | 0.455 [0.442, 0.469]          |
| **Biomedical-longformer-base** | 1 year  | 0.561 [0.550, 0.572]         | 0.572 [0.558, 0.588]          | 0.526 [0.515, 0.538]          | 0.537 [0.521, 0.552]          |
|                                | 5 year  | 0.603 [0.593, 0.614]         | 0.633 [0.619, 0.647]          | 0.532 [0.522, 0.542]          | 0.551 [0.538, 0.565]          |
|                                | 10 year | 0.603 [0.593, 0.614]         | 0.642 [0.629, 0.656]          | 0.559 [0.549, 0.569]          | 0.596 [0.584, 0.610]          |
| **Gatortron-base**             | 1 year  | 0.449 [0.438, 0.462]         | 0.437 [0.431, 0.460]          | 0.426 [0.416, 0.437]          | 0.397 [0.384, 0.411]          |
|                                | 5 year  | 0.496 [0.485, 0.506]         | 0.508 [0.495, 0.522]          | 0.433 [0.423, 0.444]          | 0.425 [0.413, 0.439]          |
|                                | 10 year | 0.489 [0.478, 0.500]         | 0.499 [0.486, 0.514]          | 0.474 [0.463, 0.485]          | 0.465 [0.453, 0.478]          |



## Additional steps
To assess variation in model performance and calculate the 95% confidence interval (CI) we use a bootstrap resampling method, calculated over 1000 iterations. Code stored in: bootstrap.py

To compare model performance across the three prediction windows (1, 5, and 10 years) and between code-based and text-based models we apply a Bonferroni correction to assess which model performs the best. Code stored in: bonferroni_correction.py

Finally, to assess the length of the input data prior to truncation or padding see: token_length_gatortron.py 

|            | Text-based                 |                         | Code-based                 |                         |
|------------|----------------------------|-------------------------|----------------------------|-------------------------|
|            | Micro-F1                   | Micro-AUPRC             | Micro-F1                   | Micro-AUPRC             |
| **1 year** | 0.449  [0.438, 0.462]      | 0.437  [0.431, 0.460]   | 0.426 [0.416, 0.437]       | 0.397 [0.384, 0.411]    |
| **5 year** | 0.496  [0.485, 0.506] ***  | 0.508 [0.495, 0.522] ***| 0.433 [0.423, 0.444]       | 0.425 [0.413, 0.439]    |
| **10 year**| 0.489 [0.478, 0.500]       | 0.499 [0.486, 0.514] ***| 0.474 [0.463, 0.485]       | 0.465 [0.453, 0.478]    |

*p < 0.05**, *p < 0.01***, *p < 0.001****



