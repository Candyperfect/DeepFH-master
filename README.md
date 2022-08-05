# README
"[Adaptive Multi-view Graph Convolutional Networks for Protein Function Prediction]"

# Environment Settings 
* python == 3.7   
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.4.1  
* pytorch_pretrained_bert == 0.6.2  
* transformers == 2.8.0  
* scikit-learn == 0.22.2  
* fair-esm ==0.4.0

# Usage 
````
CUDA_VISIBLE_DEVICES=1 nohup python model_main.py 
````

# Data
## Link
* **Protein sequence**: (http://www.uniprot.org/)  
* **Gene Ontology**: (http://geneontology.org/docs/download-ontology/) 
* **Protein functional annotation**: (http://geneontology.org/docs/download-go-annotations/) 
* **├─Homo sapiens (or Human) 
* **├─Saccharomyces cerevisiae (or Yeast)  

## Usage
Please first **unzip** the data folders and then use. 
````
Yeast/
├─Yeast_new.txt: Yeast protein sequence.    
├─Yeast_Linsim.mat: taxonomic similarity and direct acyclic graph of GO term.  
├─Yeast_Scores.mat: protein sequence scores based on the BLAST method.  
└─YeastNGOA_R.mat: Yeast protein functional annotations

````

