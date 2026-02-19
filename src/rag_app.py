import os
from datetime import datetime
import chromadb
import json
from actions import ACTION_REGISTRY
from embeddings import get_embedding
#from voice import listen#, #speak
from gemini_client import get_gemini_llm
from user_store import UserStore
from typing import Any, List, Dict

DATA_DIR = "docs"


def try_execute_action(llm_output: str) -> dict:
    """
    Executes any actions found in the user message.
    Returns: dictionary with the action results
    """
    try:
        data = json.loads(
            llm_output.replace('```', '').replace('\n',
                                                  '').replace('json', ''))
        actions = data.get("actions", [])
        data["actions_results"] = {}
        for action in actions:
            action_name = action.get("action_name", "")
            data["action_id"] = action.get("id", None)
            if action_name not in list(ACTION_REGISTRY.keys()):
                data["actions_results"][
                    action_name] = "This action cannot be executed by the agent. Please inform the user."
                continue

            args = action.get("args", {})
            result = ACTION_REGISTRY[action_name](**args)
            data["actions_results"][action_name] = result
            #print("🛠 Action result:", result)
        return data

    except Exception as e:
        print("🛠 try_execute_action failed with exception:", e)
        raise e


def add_memory(collection, text, memory_id) -> None:
    """Add conversation memory to ChromaDB."""
    emb = get_embedding(text)
    collection.add(documents=[text], embeddings=[emb], ids=[memory_id])


def finalize_agent_prompt(initial_message: str) -> tuple[str, str]:
    """
    Finalizes the agent role and prompt.
    Returns: (system_prompt, category)
    """
    # Mapping of keys to (Category Name, Default Text)
    role_map = {
        "A":
        ("planner",
         "YOU WILL BE MY PERSONAL PLANNER AND SCHEDULER. YOUR ROLE IS TO HELP ME ORGANIZE AND MANAGE ANY TASK I REQUEST OF YOU. YOU SHOULD FOCUS ON ORGANIZATION AND EFFICIENCY."
         ),
        "B":
        ("coach",
         "YOU WILL BE MY MOTIVATIONAL COACH. YOUR ROLE IS TO KEEP ME MOTIVATED TO ENSURE I REACH MY GOALS. YOU SHOULD HELP ME ORGANIZE MY DAYS WHILE STAYING MOTIVATED TO NEVER GIVE UP."
         ),
        "C":
        ("assistant",
         "YOU WILL BE MY PERSONAL ASSISTANT. YOUR ROLE IS TO HELP ME WITH ANY TASK OR REQUEST, FROM ORGANIZING MY DAYS TO REPLYING TO EMAILS AND CHEERING ME UP!"
         )
    }

    selection = str(initial_message).upper()

    if selection in role_map:
        category_name, default_text = role_map[selection]

        print(f"\nDefault Prompt: {default_text}")
        choice = input(
            "Choose an option:\n1 - Use default prompt\n2 - Write custom prompt\nSelection: "
        )

        if choice == "2":
            prompt = input("Enter your custom prompt: ")
            category = "custom"
        else:
            prompt = default_text
            category = category_name

    elif selection == "D":
        prompt = input(
            "\nPlease enter the description of the role you wish for your agent: "
        )
        category = "custom"

    else:
        print("\nInvalid selection. Defaulting to general assistant.")
        prompt = "You are my personal assistant. You must obey my every command, and correct me if I am wrong. You are to be my assistant, my mentor, my teacher. You are not to spoil me, or be easy on me or lie ever."  #YOU WILL BE A HELPFUL ASSISTANT.
        category = "general"

    return prompt, category


def get_context(collection: Any, user_input: str) -> str:
    query_emb = get_embedding(user_input)
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    context = ""
    if results["documents"]:
        for doc in results["documents"][0]:
            context += doc + "\n---\n"
    return context


def main():

    llm = get_gemini_llm()

    # Vector DB (semantic memory)
    client = chromadb.PersistentClient("chroma_db")

    # User state (SQLite)
    user_store = UserStore()
    user_id = user_store.get_or_create_user()
    # Load existing agent categories for this user
    existing_categories = user_store.get_user_collections(user_id)

    if not existing_categories:
        initial_message = input(
            "\n Welcome. I am here to assist you and make your time more efficient. Ask your assistant something. To better undetrast the role you need from me. Please choose one of the bellow: A - Scheduler/Planner; B - Motivation Coach; C- Personal Assistant; D - Custom Role"
        )
        system_prompt, category = finalize_agent_prompt(initial_message)
        user_store.add_collection(user_id, category)

    else:
        # Load existing agent categories for this user
        print("\nAvailable assistants:")
        for idx, cat in enumerate(existing_categories, start=1):
            print(f"{idx}. {cat}")

        choice = input("Select one or press Enter to create a new one: ")

        if choice.isdigit() and 1 <= int(choice) <= len(existing_categories):
            promp_map = {
                "planner":
                "YOU WILL BE MY PERSONAL PLANNER AND SCHEDULER. YOUR ROLE IS TO HELP ME ORGANIZE AND MANAGE ANY TASK I REQUEST OF YOU. YOU SHOULD FOCUS ON ORGANIZATION AND EFFICIENCY.",
                "coach":
                "YOU WILL BE MY MOTIVATIONAL COACH. YOUR ROLE IS TO KEEP ME MOTIVATED TO ENSURE I REACH MY GOALS. YOU SHOULD HELP ME ORGANIZE MY DAYS WHILE STAYING MOTIVATED TO NEVER GIVE UP.",
                "assistant":
                "YOU WILL BE MY PERSONAL ASSISTANT. YOUR ROLE IS TO HELP ME WITH ANY TASK OR REQUEST, FROM ORGANIZING MY DAYS TO REPLYING TO EMAILS AND CHEERING ME UP!",
                "general":
                "You are my personal assistant. You must obey my every command, and correct me if I am wrong. You are to be my assistant, my mentor, my teacher. You are not to spoil me, or be easy on me or lie ever."
            }
            category = existing_categories[int(choice) - 1]
            system_prompt = promp_map[category]
        else:
            initial_message = input("Describe the new assistant role: ")
            system_prompt, category = finalize_agent_prompt(initial_message)
            user_store.add_collection(user_id, category)

    collection = client.get_or_create_collection(
        category, metadata={"hnsw:space":
                            "cosine"})  # Use "cosine" as the distance function

    action_prompt = """
            If the user asks you to perform any actions on the system,
            you MUST respond ONLY with a valid JSON in the following format:

            {
            "actions": [
                {
                "id": 1,
                "action_name": "<action_name>",
                "args": { ... }
                }
            ],
            "response": "...",
            "final_action": boolean
            }

            Your available actions are:
            - open_file(path: string)
            -> Opens a file or folder for a given path

            - list_files(directory: string)
            -> Lists the files in a given directory path

            - find_and_open(name: string, search_roots: list[string])
            -> Searches for a file or folder in all allowed root paths and opens it

            - resolve_allowed_domains(domains: list[string])
            -> Resolves user-approved domains (Desktop, Documents, Downloads) into filesystem paths
            -> This is an internal action and should NOT be mentioned to the user

            - read_file(path: string)
            -> Reads and returns the contents of a file

            - write_file(path: string, content: string)
            -> Creates or overwrites a file with the given content

            - append_file(path: string, content: string)
            -> Appends content to a file (creates it if it does not exist)

            - create_directory(path: string)
            -> Creates a directory if it does not exist

            Note:
            - final_action should be true ONLY if you still need to perform additional actions
            using the result of the previous action(s). Otherwise should always be false

            If no actions are requested, you MUST respond ONLY with a valid JSON like:

            {
            "response": "..."
            }

            Additional rules:
            - If the user asks for your available actions, respond in a human-friendly way
            - resolve_allowed_domains is an internal action and must not be exposed to the user
            - If the user asks for a file without specifying a path AND you do not yet know
            which domains you are allowed to search, you MUST ask the user
            - When the user mentions one or more domains (e.g. Desktop, Documents, Downloads)
            for the first time, you MUST call resolve_allowed_domains first
            - The domain names passed to resolve_allowed_domains must be properly formatted
            (e.g. Desktop, Documents, Downloads)
            - You must remember the allowed domains for future requests
            - If a file is not found in the allowed domains, ask the user for more domains to search
    """

    while True:

        user_input = input("\nAsk your assistant something: ")
        # Para input por voz para adciionar depois como feature secudnaria, mas é um beca mau ahah
        #user_input = listen()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        context = get_context(collection, user_input) # get most relevant context from past conversations

        while complete != "true": # while the agent needs to work
            prompt = {
                "system_prompt": system_prompt,
                "action_prompt": action_prompt,
                "context": context,
                "user_input": user_input,
            }
            #prompt = system_prompt + "\n" + action_prompt + f"\nContext:\n{context}\nUser: {user_input}"
            answer = llm(str(prompt))
            tool_results = try_execute_action(
                answer)  # ADD ERROR HANDLING in case results are none
            complete = tool_results.get("final_action", "true")
            #if tool_results.get("actions", None):
            results = tool_results.get("actions_results", "")
            #actions = tool_results["actions"]
            #reflection_promp = "The user sayed " + user_input + " and requested the following actions: " + str(
            #    actions) + ". These are the results: " + str(
            #        results) + ". Respond normally to the user."
            #prompt = system_prompt + "\n" + reflection_promp + f"\nContext:\n{context}\nUser: {user_input}"
            #answer = llm(prompt)
            prompt["agent_previous_answer"] = prompt.get(
                "agent_previous_answer", "") + answer
            prompt["previous_actions_results"] = prompt.get(
                "previous_actions_results", "") + results
            #  else:

        agent_response = tool_results.get("response", "")

        print("\nAssistant:", agent_response)  # ADD ERROR HANDLING
        full_memory = f"User: {user_input}\nAssistant: {prompt["agent_previous_answer"] + " " + agent_response}"
        #full_memory = f"User: {user_input}\nAssistant: {answer}"

        add_memory(collection, full_memory,
                   f"memory-{len(collection.get()['ids'])}")


if __name__ == "__main__":
    #llm_output = '```json\n{\n  "actions": [\n    {\n      "action": "list_files",\n      "args": {\n        "directory": "/Users/mariaduarte/Desktop/Tools/rag-personal-assistant/"\n      },\n      "response": "Listing files in the specified directory."\n    }\n  ]\n}\n```'
    #try_execute_action(llm_output)
    main()


"""
next steps
improve axctions:
- test allowed domains logic
- Add more actions (like summing up files, creating new ones) / normal

IMPROVE TOOL:
- Add more friendly responses or at least better formated. /  high
- Make it a real cli tool for a mvp / high
- Improve Logging / low
"""
