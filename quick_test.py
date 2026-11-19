"""Quick test with a hardcoded sample essay."""
import sys
sys.path.insert(0, ".")

from test_scorer import run_scoring

PROMPT = """Some people think that children should start learning a foreign language at primary school, while others believe that it is better to begin at secondary school.
Discuss both views and give your own opinion.
Give reasons for your answer and include any relevant examples from your own knowledge or experience.
Write at least 250 words."""

ESSAY = """In today's globalized world, the question of when children should begin learning a foreign language has become a topic of significant debate. While some argue that early exposure at primary school is beneficial, others believe that secondary school is the more appropriate time to start. In my opinion, introducing foreign languages at primary school offers greater long-term advantages.

Those who advocate for starting language learning at secondary school often argue that younger children lack the cognitive maturity needed to grasp complex grammatical structures. At secondary level, students are better equipped to understand abstract concepts and can therefore learn languages more systematically. Moreover, primary school curricula are already packed with core subjects such as mathematics and reading, and adding a foreign language may place unnecessary pressure on young learners.

However, I strongly believe that primary school is the ideal time to introduce a foreign language. Research in linguistics consistently shows that children under the age of ten have a remarkable ability to acquire languages naturally, much like their mother tongue. This is because the brain remains highly plastic during early childhood, allowing children to absorb pronunciation and vocabulary with relative ease. Countries such as the Netherlands and Sweden, where foreign language instruction begins in primary school, consistently produce highly proficient multilingual speakers.

Furthermore, early language learning fosters cultural awareness and open-mindedness, qualities that are increasingly valuable in our interconnected world. Children who learn a second language early also tend to demonstrate enhanced cognitive abilities, including improved problem-solving skills and greater concentration.

In conclusion, while secondary school language instruction has its merits, the cognitive and cultural benefits of beginning at primary school far outweigh the drawbacks. Governments should therefore prioritise integrating foreign languages into primary education to equip future generations with the skills they need."""

print(f"Prompt length: {len(PROMPT.split())} words")
print(f"Essay length:  {len(ESSAY.split())} words\n")
run_scoring(PROMPT, ESSAY)
