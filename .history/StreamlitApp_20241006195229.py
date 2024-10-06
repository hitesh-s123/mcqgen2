import os
import json
import pandas as pd
import traceback
import torch
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data

# importing necessary packages
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain