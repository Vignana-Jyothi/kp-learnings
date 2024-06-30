import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Test data
test_data = [
    {
     "context": "Early Life and Education\nMohandas Karamchand Gandhi, better known as Mahatma Gandhi, was born on October 2, 1869, in Porbandar, a small town in present-day Gujarat, India. He was born into a Hindu Modh Baniya family. His father, Karamchand Gandhi, was the dewan (chief minister) of Porbandar, and his mother, Putlibai, was a deeply religious woman who influenced Gandhi's spiritual development.\n\nGandhi was an average student in school but exhibited a strong moral character from a young age. At the age of 13, he was married to Kasturba Makhanji in an arranged marriage, a common practice in India at the time. Despite the early marriage, Gandhi continued his education and eventually went to London to study law at University College London.\n\nTime in London\nGandhi's time in London was transformative. He struggled with adapting to Western culture but was determined to succeed. He joined the Vegetarian Society and became more interested in various religions, reading extensively about Christianity, Islam, Buddhism, and Hinduism. These studies helped shape his philosophical beliefs, especially the concept of nonviolence (ahimsa) and truth (satya).\n\nEarly Activism in South Africa\nAfter completing his studies, Gandhi returned to India to practice law but faced difficulties establishing himself. In 1893, he accepted a contract to work for an Indian firm in Natal, South Africa. This journey marked a significant turning point in his life. In South Africa, Gandhi encountered rampant racial discrimination against Indian immigrants. One incident that stood out was when he was thrown off a train at Pietermaritzburg for refusing to move from the first-class compartment, despite holding a valid ticket.\n\nThis incident and others like it spurred Gandhi into action. He began to organize the Indian community to protest against unjust laws and discrimination. Over the next 21 years, Gandhi developed his philosophy of nonviolent resistance, or Satyagraha, which would later become the cornerstone of his work in India. He led campaigns for civil rights, founded the Natal Indian Congress, and established the newspaper \"Indian Opinion\" to voice the community's grievances.\n\nReturn to India and Rise to Leadership\nGandhi returned to India in 1915, at the invitation of Gopal Krishna Gokhale, a respected leader of the Indian National Congress (INC). Upon his return, Gandhi traveled extensively across India to understand the conditions of the Indian populace. He soon realized that the majority of Indians lived in abject poverty under British colonial rule.\n\nGandhi's first major involvement in Indian politics was in the Champaran district of Bihar in 1917. The local peasants were forced to grow indigo under oppressive conditions imposed by British planters. Gandhi employed nonviolent resistance to support the peasants, leading to significant reforms and improvements in their conditions. This success catapulted Gandhi into the national spotlight.\n\nNon-Cooperation Movement\nBy 1920, Gandhi had become the leader of the INC and launched the Non-Cooperation Movement. He called for Indians to boycott British goods, institutions, and honors, and to promote self-reliance (swadeshi) by spinning their own cloth (khadi). The movement saw widespread participation, but it also led to violent outbreaks, which distressed Gandhi. In 1922, after the Chauri Chaura incident where a violent mob killed police officers, Gandhi called off the movement, emphasizing his commitment to nonviolence.\n\nSalt March and Civil Disobedience Movement\nIn 1930, Gandhi launched one of his most famous campaigns, the Salt March. The British had imposed a tax on salt, making it illegal for Indians to produce their own. Gandhi led a 240-mile march from Sabarmati Ashram to the coastal village of Dandi, where he made salt from seawater, defying the British law. This act of civil disobedience ignited a nationwide movement and drew global attention to the Indian independence struggle.\n\nImprisonment and Negotiations\nThroughout his life, Gandhi was arrested numerous times by the British authorities. Despite this, he remained committed to his principles of nonviolence and truth. During the 1930s and 1940s, Gandhi continued to lead various campaigns and negotiations for India's independence. He also focused on social reforms, advocating for the eradication of untouchability, promoting education, and improving women's rights.\n\nQuit India Movement\nIn 1942, during World War II, Gandhi launched the Quit India Movement, demanding an end to British rule. The movement called for immediate independence and led to widespread protests and arrests. Gandhi and other INC leaders were imprisoned, and India was plunged into chaos. Despite the crackdown, the movement significantly weakened British control and intensified the call for independence.\n\nPartition and Independence\nAfter World War II, the British government, weakened by the war and facing intense pressure from Indian nationalists, decided to grant India independence. However, religious tensions between Hindus and Muslims had escalated, leading to the demand for a separate Muslim state. Despite Gandhi's efforts to keep India united, the subcontinent was partitioned into India and Pakistan in 1947, leading to massive communal violence and displacement.\n\nAssassination and Legacy\nOn January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's conciliatory approach towards Muslims. Gandhi's death was a profound loss for India and the world.\n\nMahatma Gandhi's legacy endures through his teachings and the impact he had on global movements for civil rights and freedom. His principles of nonviolence and peaceful resistance influenced leaders like Martin Luther King Jr., Nelson Mandela, and many others in their struggles for justice and equality. Gandhi's life and work remain a testament to the power of truth and nonviolence in the fight against oppression.",
     "question": "When was Mahatma Gandhi born?"},
    {
     "context": "Early Life and Education\nMohandas Karamchand Gandhi, better known as Mahatma Gandhi, was born on October 2, 1869, in Porbandar, a small town in present-day Gujarat, India. He was born into a Hindu Modh Baniya family. His father, Karamchand Gandhi, was the dewan (chief minister) of Porbandar, and his mother, Putlibai, was a deeply religious woman who influenced Gandhi's spiritual development.\n\nGandhi was an average student in school but exhibited a strong moral character from a young age. At the age of 13, he was married to Kasturba Makhanji in an arranged marriage, a common practice in India at the time. Despite the early marriage, Gandhi continued his education and eventually went to London to study law at University College London.\n\nTime in London\nGandhi's time in London was transformative. He struggled with adapting to Western culture but was determined to succeed. He joined the Vegetarian Society and became more interested in various religions, reading extensively about Christianity, Islam, Buddhism, and Hinduism. These studies helped shape his philosophical beliefs, especially the concept of nonviolence (ahimsa) and truth (satya).\n\nEarly Activism in South Africa\nAfter completing his studies, Gandhi returned to India to practice law but faced difficulties establishing himself. In 1893, he accepted a contract to work for an Indian firm in Natal, South Africa. This journey marked a significant turning point in his life. In South Africa, Gandhi encountered rampant racial discrimination against Indian immigrants. One incident that stood out was when he was thrown off a train at Pietermaritzburg for refusing to move from the first-class compartment, despite holding a valid ticket.\n\nThis incident and others like it spurred Gandhi into action. He began to organize the Indian community to protest against unjust laws and discrimination. Over the next 21 years, Gandhi developed his philosophy of nonviolent resistance, or Satyagraha, which would later become the cornerstone of his work in India. He led campaigns for civil rights, founded the Natal Indian Congress, and established the newspaper \"Indian Opinion\" to voice the community's grievances.\n\nReturn to India and Rise to Leadership\nGandhi returned to India in 1915, at the invitation of Gopal Krishna Gokhale, a respected leader of the Indian National Congress (INC). Upon his return, Gandhi traveled extensively across India to understand the conditions of the Indian populace. He soon realized that the majority of Indians lived in abject poverty under British colonial rule.\n\nGandhi's first major involvement in Indian politics was in the Champaran district of Bihar in 1917. The local peasants were forced to grow indigo under oppressive conditions imposed by British planters. Gandhi employed nonviolent resistance to support the peasants, leading to significant reforms and improvements in their conditions. This success catapulted Gandhi into the national spotlight.\n\nNon-Cooperation Movement\nBy 1920, Gandhi had become the leader of the INC and launched the Non-Cooperation Movement. He called for Indians to boycott British goods, institutions, and honors, and to promote self-reliance (swadeshi) by spinning their own cloth (khadi). The movement saw widespread participation, but it also led to violent outbreaks, which distressed Gandhi. In 1922, after the Chauri Chaura incident where a violent mob killed police officers, Gandhi called off the movement, emphasizing his commitment to nonviolence.\n\nSalt March and Civil Disobedience Movement\nIn 1930, Gandhi launched one of his most famous campaigns, the Salt March. The British had imposed a tax on salt, making it illegal for Indians to produce their own. Gandhi led a 240-mile march from Sabarmati Ashram to the coastal village of Dandi, where he made salt from seawater, defying the British law. This act of civil disobedience ignited a nationwide movement and drew global attention to the Indian independence struggle.\n\nImprisonment and Negotiations\nThroughout his life, Gandhi was arrested numerous times by the British authorities. Despite this, he remained committed to his principles of nonviolence and truth. During the 1930s and 1940s, Gandhi continued to lead various campaigns and negotiations for India's independence. He also focused on social reforms, advocating for the eradication of untouchability, promoting education, and improving women's rights.\n\nQuit India Movement\nIn 1942, during World War II, Gandhi launched the Quit India Movement, demanding an end to British rule. The movement called for immediate independence and led to widespread protests and arrests. Gandhi and other INC leaders were imprisoned, and India was plunged into chaos. Despite the crackdown, the movement significantly weakened British control and intensified the call for independence.\n\nPartition and Independence\nAfter World War II, the British government, weakened by the war and facing intense pressure from Indian nationalists, decided to grant India independence. However, religious tensions between Hindus and Muslims had escalated, leading to the demand for a separate Muslim state. Despite Gandhi's efforts to keep India united, the subcontinent was partitioned into India and Pakistan in 1947, leading to massive communal violence and displacement.\n\nAssassination and Legacy\nOn January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's conciliatory approach towards Muslims. Gandhi's death was a profound loss for India and the world.\n\nMahatma Gandhi's legacy endures through his teachings and the impact he had on global movements for civil rights and freedom. His principles of nonviolence and peaceful resistance influenced leaders like Martin Luther King Jr., Nelson Mandela, and many others in their struggles for justice and equality. Gandhi's life and work remain a testament to the power of truth and nonviolence in the fight against oppression.",
     "question": "Where was Mahatma Gandhi born?"},
    {
     "context": "Early Life and Education\nMohandas Karamchand Gandhi, better known as Mahatma Gandhi, was born on October 2, 1869, in Porbandar, a small town in present-day Gujarat, India. He was born into a Hindu Modh Baniya family. His father, Karamchand Gandhi, was the dewan (chief minister) of Porbandar, and his mother, Putlibai, was a deeply religious woman who influenced Gandhi's spiritual development.\n\nGandhi was an average student in school but exhibited a strong moral character from a young age. At the age of 13, he was married to Kasturba Makhanji in an arranged marriage, a common practice in India at the time. Despite the early marriage, Gandhi continued his education and eventually went to London to study law at University College London.\n\nTime in London\nGandhi's time in London was transformative. He struggled with adapting to Western culture but was determined to succeed. He joined the Vegetarian Society and became more interested in various religions, reading extensively about Christianity, Islam, Buddhism, and Hinduism. These studies helped shape his philosophical beliefs, especially the concept of nonviolence (ahimsa) and truth (satya).\n\nEarly Activism in South Africa\nAfter completing his studies, Gandhi returned to India to practice law but faced difficulties establishing himself. In 1893, he accepted a contract to work for an Indian firm in Natal, South Africa. This journey marked a significant turning point in his life. In South Africa, Gandhi encountered rampant racial discrimination against Indian immigrants. One incident that stood out was when he was thrown off a train at Pietermaritzburg for refusing to move from the first-class compartment, despite holding a valid ticket.\n\nThis incident and others like it spurred Gandhi into action. He began to organize the Indian community to protest against unjust laws and discrimination. Over the next 21 years, Gandhi developed his philosophy of nonviolent resistance, or Satyagraha, which would later become the cornerstone of his work in India. He led campaigns for civil rights, founded the Natal Indian Congress, and established the newspaper \"Indian Opinion\" to voice the community's grievances.\n\nReturn to India and Rise to Leadership\nGandhi returned to India in 1915, at the invitation of Gopal Krishna Gokhale, a respected leader of the Indian National Congress (INC). Upon his return, Gandhi traveled extensively across India to understand the conditions of the Indian populace. He soon realized that the majority of Indians lived in abject poverty under British colonial rule.\n\nGandhi's first major involvement in Indian politics was in the Champaran district of Bihar in 1917. The local peasants were forced to grow indigo under oppressive conditions imposed by British planters. Gandhi employed nonviolent resistance to support the peasants, leading to significant reforms and improvements in their conditions. This success catapulted Gandhi into the national spotlight.\n\nNon-Cooperation Movement\nBy 1920, Gandhi had become the leader of the INC and launched the Non-Cooperation Movement. He called for Indians to boycott British goods, institutions, and honors, and to promote self-reliance (swadeshi) by spinning their own cloth (khadi). The movement saw widespread participation, but it also led to violent outbreaks, which distressed Gandhi. In 1922, after the Chauri Chaura incident where a violent mob killed police officers, Gandhi called off the movement, emphasizing his commitment to nonviolence.\n\nSalt March and Civil Disobedience Movement\nIn 1930, Gandhi launched one of his most famous campaigns, the Salt March. The British had imposed a tax on salt, making it illegal for Indians to produce their own. Gandhi led a 240-mile march from Sabarmati Ashram to the coastal village of Dandi, where he made salt from seawater, defying the British law. This act of civil disobedience ignited a nationwide movement and drew global attention to the Indian independence struggle.\n\nImprisonment and Negotiations\nThroughout his life, Gandhi was arrested numerous times by the British authorities. Despite this, he remained committed to his principles of nonviolence and truth. During the 1930s and 1940s, Gandhi continued to lead various campaigns and negotiations for India's independence. He also focused on social reforms, advocating for the eradication of untouchability, promoting education, and improving women's rights.\n\nQuit India Movement\nIn 1942, during World War II, Gandhi launched the Quit India Movement, demanding an end to British rule. The movement called for immediate independence and led to widespread protests and arrests. Gandhi and other INC leaders were imprisoned, and India was plunged into chaos. Despite the crackdown, the movement significantly weakened British control and intensified the call for independence.\n\nPartition and Independence\nAfter World War II, the British government, weakened by the war and facing intense pressure from Indian nationalists, decided to grant India independence. However, religious tensions between Hindus and Muslims had escalated, leading to the demand for a separate Muslim state. Despite Gandhi's efforts to keep India united, the subcontinent was partitioned into India and Pakistan in 1947, leading to massive communal violence and displacement.\n\nAssassination and Legacy\nOn January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's conciliatory approach towards Muslims. Gandhi's death was a profound loss for India and the world.\n\nMahatma Gandhi's legacy endures through his teachings and the impact he had on global movements for civil rights and freedom. His principles of nonviolence and peaceful resistance influenced leaders like Martin Luther King Jr., Nelson Mandela, and many others in their struggles for justice and equality. Gandhi's life and work remain a testament to the power of truth and nonviolence in the fight against oppression.",
     "question": "Where was Gandhi born?"},
    {
     "context": "Early Life and Education\nMohandas Karamchand Gandhi, better known as Mahatma Gandhi, was born on October 2, 1869, in Porbandar, a small town in present-day Gujarat, India. He was born into a Hindu Modh Baniya family. His father, Karamchand Gandhi, was the dewan (chief minister) of Porbandar, and his mother, Putlibai, was a deeply religious woman who influenced Gandhi's spiritual development.\n\nGandhi was an average student in school but exhibited a strong moral character from a young age. At the age of 13, he was married to Kasturba Makhanji in an arranged marriage, a common practice in India at the time. Despite the early marriage, Gandhi continued his education and eventually went to London to study law at University College London.\n\nTime in London\nGandhi's time in London was transformative. He struggled with adapting to Western culture but was determined to succeed. He joined the Vegetarian Society and became more interested in various religions, reading extensively about Christianity, Islam, Buddhism, and Hinduism. These studies helped shape his philosophical beliefs, especially the concept of nonviolence (ahimsa) and truth (satya).\n\nEarly Activism in South Africa\nAfter completing his studies, Gandhi returned to India to practice law but faced difficulties establishing himself. In 1893, he accepted a contract to work for an Indian firm in Natal, South Africa. This journey marked a significant turning point in his life. In South Africa, Gandhi encountered rampant racial discrimination against Indian immigrants. One incident that stood out was when he was thrown off a train at Pietermaritzburg for refusing to move from the first-class compartment, despite holding a valid ticket.\n\nThis incident and others like it spurred Gandhi into action. He began to organize the Indian community to protest against unjust laws and discrimination. Over the next 21 years, Gandhi developed his philosophy of nonviolent resistance, or Satyagraha, which would later become the cornerstone of his work in India. He led campaigns for civil rights, founded the Natal Indian Congress, and established the newspaper \"Indian Opinion\" to voice the community's grievances.\n\nReturn to India and Rise to Leadership\nGandhi returned to India in 1915, at the invitation of Gopal Krishna Gokhale, a respected leader of the Indian National Congress (INC). Upon his return, Gandhi traveled extensively across India to understand the conditions of the Indian populace. He soon realized that the majority of Indians lived in abject poverty under British colonial rule.\n\nGandhi's first major involvement in Indian politics was in the Champaran district of Bihar in 1917. The local peasants were forced to grow indigo under oppressive conditions imposed by British planters. Gandhi employed nonviolent resistance to support the peasants, leading to significant reforms and improvements in their conditions. This success catapulted Gandhi into the national spotlight.\n\nNon-Cooperation Movement\nBy 1920, Gandhi had become the leader of the INC and launched the Non-Cooperation Movement. He called for Indians to boycott British goods, institutions, and honors, and to promote self-reliance (swadeshi) by spinning their own cloth (khadi). The movement saw widespread participation, but it also led to violent outbreaks, which distressed Gandhi. In 1922, after the Chauri Chaura incident where a violent mob killed police officers, Gandhi called off the movement, emphasizing his commitment to nonviolence.\n\nSalt March and Civil Disobedience Movement\nIn 1930, Gandhi launched one of his most famous campaigns, the Salt March. The British had imposed a tax on salt, making it illegal for Indians to produce their own. Gandhi led a 240-mile march from Sabarmati Ashram to the coastal village of Dandi, where he made salt from seawater, defying the British law. This act of civil disobedience ignited a nationwide movement and drew global attention to the Indian independence struggle.\n\nImprisonment and Negotiations\nThroughout his life, Gandhi was arrested numerous times by the British authorities. Despite this, he remained committed to his principles of nonviolence and truth. During the 1930s and 1940s, Gandhi continued to lead various campaigns and negotiations for India's independence. He also focused on social reforms, advocating for the eradication of untouchability, promoting education, and improving women's rights.\n\nQuit India Movement\nIn 1942, during World War II, Gandhi launched the Quit India Movement, demanding an end to British rule. The movement called for immediate independence and led to widespread protests and arrests. Gandhi and other INC leaders were imprisoned, and India was plunged into chaos. Despite the crackdown, the movement significantly weakened British control and intensified the call for independence.\n\nPartition and Independence\nAfter World War II, the British government, weakened by the war and facing intense pressure from Indian nationalists, decided to grant India independence. However, religious tensions between Hindus and Muslims had escalated, leading to the demand for a separate Muslim state. Despite Gandhi's efforts to keep India united, the subcontinent was partitioned into India and Pakistan in 1947, leading to massive communal violence and displacement.\n\nAssassination and Legacy\nOn January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's conciliatory approach towards Muslims. Gandhi's death was a profound loss for India and the world.\n\nMahatma Gandhi's legacy endures through his teachings and the impact he had on global movements for civil rights and freedom. His principles of nonviolence and peaceful resistance influenced leaders like Martin Luther King Jr., Nelson Mandela, and many others in their struggles for justice and equality. Gandhi's life and work remain a testament to the power of truth and nonviolence in the fight against oppression.",
     "question": "Where was Mohandas Karamchand Gandhi born?"},
    {
     "context": "Early Life and Education\nMohandas Karamchand Gandhi, better known as Mahatma Gandhi, was born on October 2, 1869, in Porbandar, a small town in present-day Gujarat, India. He was born into a Hindu Modh Baniya family. His father, Karamchand Gandhi, was the dewan (chief minister) of Porbandar, and his mother, Putlibai, was a deeply religious woman who influenced Gandhi's spiritual development.\n\nGandhi was an average student in school but exhibited a strong moral character from a young age. At the age of 13, he was married to Kasturba Makhanji in an arranged marriage, a common practice in India at the time. Despite the early marriage, Gandhi continued his education and eventually went to London to study law at University College London.\n\nTime in London\nGandhi's time in London was transformative. He struggled with adapting to Western culture but was determined to succeed. He joined the Vegetarian Society and became more interested in various religions, reading extensively about Christianity, Islam, Buddhism, and Hinduism. These studies helped shape his philosophical beliefs, especially the concept of nonviolence (ahimsa) and truth (satya).\n\nEarly Activism in South Africa\nAfter completing his studies, Gandhi returned to India to practice law but faced difficulties establishing himself. In 1893, he accepted a contract to work for an Indian firm in Natal, South Africa. This journey marked a significant turning point in his life. In South Africa, Gandhi encountered rampant racial discrimination against Indian immigrants. One incident that stood out was when he was thrown off a train at Pietermaritzburg for refusing to move from the first-class compartment, despite holding a valid ticket.\n\nThis incident and others like it spurred Gandhi into action. He began to organize the Indian community to protest against unjust laws and discrimination. Over the next 21 years, Gandhi developed his philosophy of nonviolent resistance, or Satyagraha, which would later become the cornerstone of his work in India. He led campaigns for civil rights, founded the Natal Indian Congress, and established the newspaper \"Indian Opinion\" to voice the community's grievances.\n\nReturn to India and Rise to Leadership\nGandhi returned to India in 1915, at the invitation of Gopal Krishna Gokhale, a respected leader of the Indian National Congress (INC). Upon his return, Gandhi traveled extensively across India to understand the conditions of the Indian populace. He soon realized that the majority of Indians lived in abject poverty under British colonial rule.\n\nGandhi's first major involvement in Indian politics was in the Champaran district of Bihar in 1917. The local peasants were forced to grow indigo under oppressive conditions imposed by British planters. Gandhi employed nonviolent resistance to support the peasants, leading to significant reforms and improvements in their conditions. This success catapulted Gandhi into the national spotlight.\n\nNon-Cooperation Movement\nBy 1920, Gandhi had become the leader of the INC and launched the Non-Cooperation Movement. He called for Indians to boycott British goods, institutions, and honors, and to promote self-reliance (swadeshi) by spinning their own cloth (khadi). The movement saw widespread participation, but it also led to violent outbreaks, which distressed Gandhi. In 1922, after the Chauri Chaura incident where a violent mob killed police officers, Gandhi called off the movement, emphasizing his commitment to nonviolence.\n\nSalt March and Civil Disobedience Movement\nIn 1930, Gandhi launched one of his most famous campaigns, the Salt March. The British had imposed a tax on salt, making it illegal for Indians to produce their own. Gandhi led a 240-mile march from Sabarmati Ashram to the coastal village of Dandi, where he made salt from seawater, defying the British law. This act of civil disobedience ignited a nationwide movement and drew global attention to the Indian independence struggle.\n\nImprisonment and Negotiations\nThroughout his life, Gandhi was arrested numerous times by the British authorities. Despite this, he remained committed to his principles of nonviolence and truth. During the 1930s and 1940s, Gandhi continued to lead various campaigns and negotiations for India's independence. He also focused on social reforms, advocating for the eradication of untouchability, promoting education, and improving women's rights.\n\nQuit India Movement\nIn 1942, during World War II, Gandhi launched the Quit India Movement, demanding an end to British rule. The movement called for immediate independence and led to widespread protests and arrests. Gandhi and other INC leaders were imprisoned, and India was plunged into chaos. Despite the crackdown, the movement significantly weakened British control and intensified the call for independence.\n\nPartition and Independence\nAfter World War II, the British government, weakened by the war and facing intense pressure from Indian nationalists, decided to grant India independence. However, religious tensions between Hindus and Muslims had escalated, leading to the demand for a separate Muslim state. Despite Gandhi's efforts to keep India united, the subcontinent was partitioned into India and Pakistan in 1947, leading to massive communal violence and displacement.\n\nAssassination and Legacy\nOn January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's conciliatory approach towards Muslims. Gandhi's death was a profound loss for India and the world.\n\nMahatma Gandhi's legacy endures through his teachings and the impact he had on global movements for civil rights and freedom. His principles of nonviolence and peaceful resistance influenced leaders like Martin Luther King Jr., Nelson Mandela, and many others in their struggles for justice and equality. Gandhi's life and work remain a testament to the power of truth and nonviolence in the fight against oppression.",
     "question": "Where was he born?"},
    # Add more test examples
]

# Model and tokenizer
model_name = 'deepset/roberta-base-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to get answers from the model
def get_answers(test_data, qa_pipeline):
    results = []
    for item in test_data:
        result = qa_pipeline(question=item["question"], context=item["context"])
        results.append({
            "question": item["question"],
            "context": item["context"],
            "answer": result["answer"],
            "score": result["score"]
        })
    return results

# Get answers
answers = get_answers(test_data, qa_pipeline)

# Convert to DataFrame
df = pd.DataFrame(answers)

# Print the DataFrame
print(df)
