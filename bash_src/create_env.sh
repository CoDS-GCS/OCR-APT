conda env create -f ../environment.yml
sleep 2m
conda activate env-ocrapt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse  -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install openai
pip install llama-index
pip install llama-index-llms-ollama  llama-index-llms-openai  llama-index-llms-deepseek llama-index-embeddings-huggingface llama-index-embeddings-ollama
pip install -r ../requirements.txt

