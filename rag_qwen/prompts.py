SITUATE_PROMPT = """
<situate_instructions>
Ты помогаешь улучшать поиск. Дано: полный документ и его чанк.
Кратко опиши, как чанк вписывается в замысел документа и какие ключевые понятия/API/классы он уточняет.
Пиши 2–4 предложения, без повторения текста чанка, без выводов, только контекст.
</situate_instructions>

<DOCUMENT>
{full_document_truncated}
</DOCUMENT>

<CHUNK>
{chunk_text_truncated}
</CHUNK>

Ответь только контекстом.
"""

RERANK_PAIR_TEMPLATE = """ # This will be used for tokenization as (query, passage)
"""

ANSWER_PROMPT_WITH_CITATIONS = """
<answer_instructions>
Ответь кратко и точно, используя только данные из сниппетов. Если знаний не хватает — явно скажи, чего не хватает.
</answer_instructions>

<QUERY>
{query}
</QUERY>

<CONTEXT_SNIPPETS>
{enumerated_top_chunks_with_doc_id_and_index}
</CONTEXT_SNIPPETS>
"""
