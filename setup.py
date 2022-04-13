from setuptools import setup, find_packages

setup(
    name = 'load_model',
    version = '0.1.0',
    url = '',
    description = '',
    packages = find_packages(),
    install_requires = [
    'robustness @ git+ssh://git@github.com/HelenR6/robustness@1.2.1',
#     'clip @ git+ssh://git@github.com/openai/CLIP.git',
#     'advertorch @ git+ssh://git@github.com/HelenR6/advertorch',
]
 
)
