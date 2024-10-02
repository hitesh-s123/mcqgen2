## we use this file to download local package in our virtual environment .
from setuptools import find_packages,setup

setup(
    name = 'mcqgenerator',
    version='0.0.1',
    author='hitesh',
    author_email='jnvpghitesh@gmail.com',
    install_requires=['transformers','accelerate','tensorflow','tokenizers','langchain','streamlit','python-dotenv','PyPDF2','langchain_community','torch'],
    packages=find_packages()
)