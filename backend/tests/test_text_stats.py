"""端到端测试 text_stats"""
from scoring.text_stats import (
    ttr, mtld, advanced_word_count, collocation_hits,
    grammar_error_count, sentence_complexity, compute_text_metrics
)
print('=== text_stats loaded ===')

text = ('I really like cooking because it is a great way to relax and learn new '
        'techniques. However, I do not always have enough time to cook elaborate '
        'meals during the week.')

print(f'TTR: {ttr(text):.3f}')
print(f'MTLD: {mtld(text):.2f}')

n, words = advanced_word_count(text)
print(f'Advanced words: {n} {words}')

n, colls = collocation_hits(text)
print(f'Collocations: {n} {colls}')

g = grammar_error_count(text)
print(f'Grammar errors: {g["count"]} examples: {g["examples"][:3]}')

sentences = [
    'I really like cooking.',
    'It is a great way to relax, and I enjoy learning new techniques.',
    'Although I do not have time, I still try.',
]
print(f'Sentence types: {sentence_complexity(sentences)}')

m = compute_text_metrics(text, sentences)
print()
print('=== Aggregate metrics ===')
for k, v in m.items():
    if not isinstance(v, list):
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: [{len(v)} items] {v[:3]}')