import os
from pathlib import Path

from ollama import ChatResponse, chat

PROLOG_DATA = Path("data/Prolog")


def format_for_llm_prompt(file_content, filename):
    """Format file content for LLM prompt"""
    return f"""
### File: {filename}
```prolog
{file_content}
```
"""


def read_context_data(folder_path=PROLOG_DATA):
    """Read all files from PROLOG_DATA folder and format for LLM"""
    # Ensure the folder exists
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} not found")

    # Get all files in the folder
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = Path(root) / file
            all_files.append(file_path)

    # Read and format each file
    formatted_content = []
    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                formatted = format_for_llm_prompt(content, file_path.name)
                formatted_content.append(formatted)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Combine all formatted content
    return "\n".join(formatted_content)


MAX_CONTEXT_CHARS = 2**11


# Useful for testing.
def init_agent(preloaded_data: str):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in PROLOG programming. You know that: `Predicate(Arguments)` - Запрос к базе знаний, когда задаётся вопрос, выводящий ответы на основе известных фактов и правил.",
        },
        {
            "role": "system",
            "content": "Answer questions about the provided PROLOG code files.",
        },
        {
            "role": "system",
            "content": "Всегда отвечай на русском, даже если я пользователь будет задавать вопросы на другом языке",
        },
        {
            "role": "user",
            "content": f"Here are my prolog files:\n{preloaded_data[:MAX_CONTEXT_CHARS]}\nI'll ask questions about them.",
        },
    ]

    def agent_query(user_question: str) -> ChatResponse:
        messages.append({
            "role": "user",
            "content": user_question,
        })

        # Default: http://localhost:11434.
        # Otherwise create custom client and use it as client.chat.
        response: ChatResponse = chat(
            model="llama3.2",
            messages=messages,
        )

        return response

    return agent_query


def main():
    prolog_context = "Test"

    try:
        prolog_context = read_context_data()
        # print("Formatted text ready for LLM prompt:")
        # print(llm_prompt_text[:100] + "...")  # Print first N chars as preview
        print("Успешно обработаны данные PROLOG базы мировоззрений")
    except Exception as e:
        print(f"Ошибка: {e}")

    # Start chat session
    print("PROLOG Data Chat - Напишите 'quit', чтобы выйти\n")

    agent_query = init_agent(prolog_context)

    while True:
        # Get user question
        user_input = input("\nВаш вопрос: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if not user_input:
            print("Пожалуйста, задайте вопрос")
            continue

        response = agent_query(user_input)
        # print(response.message.content)

        # Extract and display response
        answer = response.message.content
        print(f"\nОтвет: {answer}")

        # Add assistant response to message history
        # messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
    # result = main()
    # print(result)
