from langchain_core.prompts import ChatPromptTemplate

document_analysis_prompt = ChatPromptTemplate.from_template("""
You are an assistant specialized in document analysis and summarization.

Instructions:
- Analyze the provided document content.
- Generate a concise summary and extract key metadata.
- Return ONLY valid JSON matching the schema.
- Do not include any additional text.

Schema:
{{
  "document_title": null,
  "summary": "",
  "key_points": [],
  "entities": [
    {{
      "type": "",
      "value": ""
    }}
  ]
}}

Document Content:
{document_text}
""")

document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs.Your tasks are as follows:

1. Compare the content of both documents and identify any differences or similarities.
2. Identify the difference in PDF and note down the page number where the difference occurs.
3. If any page do not have any difference, mention that as well. or No differences found.
4. Return the results in valid JSON format as per the schema below.
Schema:
{{
  "differences": [
    {{
      "page": "",
      "changes": ""
    }}
  ]
}}
""")




PROMPT_REGISTRY = {"document_analysis": document_analysis_prompt, "document_comparison": document_comparison_prompt}