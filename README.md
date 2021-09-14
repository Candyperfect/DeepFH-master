# Our method
"[knowledge-augmented protein function prediction via the textual semantic information of gene ontology terms]"

# Environment Settings 
* python == 3.7   
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.4.1  
* pytorch_pretrained_bert == 0.6.2  
* transformers == 2.8.0  
* scikit-learn == 0.22.2  
* biobert_v1.1_pubmed_v2.5.1_convert

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
Please first **unzip** the data folders and then use. The files in folders are as follows:
````
Yeast_cc/
├─yeast_new.txt: Yeast protein sequence.  
├─cc_Yeast_GOnames.mat: text information of GO term.  
├─cc_Yeast_Linsim.mat: taxonomic similarity of GO term.  
└─Yeast_ccNew.mat: Yeast protein functional annotations

````

