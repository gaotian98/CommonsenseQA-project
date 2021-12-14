# CommonsenseQA-project

This is my course project on commonsense question answering with language models and knowledge graphs. The approach is based on the QA-GNN architecture, on top of which I make a few modifications. The experiments were run on UMich Greatlakes, and I've attached the log files in the repo as well.

## Steps to reproduce the results
* Clone the repo, create a virtual environment and run the script install.sh to install necessary packages.
* Run the script download_preprocessed_data.sh to get preprocessed data.
* Run 
  ```
  mv pr.py data/csqa/graph/pr.py
  python pr.py train
  python pr.py dev
  python pr.py test
  ```
* Modify the paths or hyperparameters in main_job.sh and submit the job on Greatlakes.

## Reference
```bib
@InProceedings{yasunaga2021qagnn,
  author = {Michihiro Yasunaga and Hongyu Ren and Antoine Bosselut and Percy Liang and Jure Leskovec},
  title = {QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering},
  year = {2021},
  booktitle = {North American Chapter of the Association for Computational Linguistics (NAACL)},
}
```
