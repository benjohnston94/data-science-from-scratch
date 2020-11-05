# Data Science from Scratch

The inspiration from this repo comes from reading the book by Joel Grus of the same title. This book goes through various machine learning algorithms and builds them from scratch in Python, explaining the math and theory 'underneath the hood'

I'm going to try and rebuild from scratch a lot of the concepts from the book (without looking!) to stregthen my understanding of these concepts.

In order to follow good software engineering practices we'll use unit testing and type hints throughout the repo.

A good method to ensure we're using type hints correctly 
is to use 'mypy'. The terminal command line will be 
"mypy foldername/ --ignore-missing-imports". The "ignore-missing-imports"
is used because other libraries (like numpy for example)
