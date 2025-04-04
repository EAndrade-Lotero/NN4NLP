from pathlib import Path

PATHS = {
    'tokenizer_example': Path.cwd() / Path('NN4NLP', 'data', 'tokenizers', 'tokenizer.json'),
    'tokenizers': Path.cwd() / Path('NN4NLP', 'data', 'tokenizers'),
    'word_embeddings': Path.cwd() / Path('NN4NLP', 'data', 'embeddings'),
    'lms': Path.cwd() / Path('NN4NLP', 'data', 'lms'),
    'training_data': Path.cwd() / Path('training_data')
}

PATHS['tokenizers'].mkdir(exist_ok=True)
PATHS['word_embeddings'].mkdir(exist_ok=True)
PATHS['lms'].mkdir(exist_ok=True)
PATHS['training_data'].mkdir(exist_ok=True)