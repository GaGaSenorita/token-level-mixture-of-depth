from datasets import load_dataset

dataset = load_dataset("ag_news")
print(dataset)
print(dataset['train'][0])
#Hello World!
'''
e.g.
{'text': "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.", 'label': 2}
'''
