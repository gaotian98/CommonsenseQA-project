pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0
pip install nltk spacy==2.1.6
python -m spacy download en

# for torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-geometric
